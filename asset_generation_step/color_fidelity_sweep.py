"""
Stage 3 experiment — color fidelity vs polygon budget.

Companion to fidelity_sweep.py, per supervisor request: the M2 v1 color pass
stores one RGB per VERTEX, so decimating the mesh also decimates the color
signal. This sweep paints each geometry LOD with the exact color_hull.py
projection (same functions, imported) and measures how far each level's
surface COLOR strays from the full-resolution colored reference — geometry
already flatlined at 20k faces; this asks where color flatlines.

Metric: symmetric chamfer as in fidelity_sweep.py, but comparing RGB at the
matched surface points instead of distance between them (barycentric-
interpolated vertex colors, so it measures what a renderer would show).
Reported as Euclidean RGB error on the 0-255 scale; the reference sampled
against itself gives the metric's noise floor.

Requires: work/<SUBJECT>_hull_colored.glb (the reference, from color_hull.py),
work/lods/<SUBJECT>_lod_<budget>.glb (from fidelity_sweep.py — re-decimated
from the hull if missing), frames + masks + calibration as for color_hull.py.

Usage:  SUBJECT=snowglobe venv/bin/python asset_generation_step/color_fidelity_sweep.py

Output: work/lods/<SUBJECT>_lod_<faces>_colored.glb
        work/lods/render_color_*.png               same-camera colored renders
        work/lods/color_sweep_results.json
"""

import glob
import json
import os
import sys
import time

import numpy as np
import trimesh
from scipy.spatial import KDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import color_hull                                    # noqa: E402
from fidelity_sweep import decimate, render          # noqa: E402

SUBJECT_ID  = os.environ.get('SUBJECT', 'product_000')
HULL_GLB    = os.path.join(BASE_DIR, 'work', f'{SUBJECT_ID}_hull.glb')
COLORED_REF = os.path.join(BASE_DIR, 'work', f'{SUBJECT_ID}_hull_colored.glb')
OUT_DIR     = os.path.join(BASE_DIR, 'work', 'lods')

FACE_BUDGETS = [int(v) for v in os.environ.get(
    'FACE_BUDGETS', '1000,5000,20000,80000,320000').split(',')]
N_SAMPLES = 50_000


def vertex_colors(mesh):
    vis = mesh.visual
    if not hasattr(vis, 'vertex_colors'):
        vis = vis.to_color()
    return np.asarray(vis.vertex_colors)


def paint(mesh, cam, dist, calib_wh):
    """color_hull.py's sample-and-blend pass, applied to an arbitrary mesh
    (mirrors its main(); same functions, same weights)."""
    verts, normals = mesh.vertices, mesh.vertex_normals
    lo, hi = mesh.bounds
    X, Y, Z = ((verts - lo) / (hi - lo)).T
    colors, weights = {}, {}
    nx, ny, nz = normals.T
    for view in color_hull.VIEWS:
        col = color_hull.sample_view(view, X, Y, Z, cam, dist, calib_wh)
        if col is None:
            continue
        colors[view] = col
        if view == 'front':
            has_back = 'back' in colors or glob.glob(
                os.path.join(color_hull.MASKS_DIR, 'back', '*.png'))
            weights[view] = (np.maximum(nz, 0) if has_back
                             else np.abs(nz)) ** color_hull.BLEND_POWER
        elif view == 'back':
            weights[view] = np.maximum(-nz, 0) ** color_hull.BLEND_POWER
        elif view == 'side':
            weights[view] = np.abs(nx) ** color_hull.BLEND_POWER
        elif view == 'top':
            weights[view] = np.maximum(ny, 0) ** color_hull.BLEND_POWER
        elif view == 'bottom':
            weights[view] = np.maximum(-ny, 0) ** color_hull.BLEND_POWER
    if 'front' not in colors:
        print("[!] Front view is required for the color pass.")
        sys.exit(1)
    total = sum(weights.values())
    total[total == 0] = 1
    blend = sum(colors[v] * (weights[v] / total)[:, None] for v in colors)
    rgba = np.concatenate([np.clip(blend, 0, 255).astype(np.uint8),
                           np.full((len(verts), 1), 255, np.uint8)], axis=1)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=rgba)
    return mesh


def colored_samples(mesh, n):
    """n surface points with barycentric-interpolated vertex colors —
    the color a renderer would actually show at that point."""
    pts, fidx = trimesh.sample.sample_surface(mesh, n)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[fidx], pts)
    vc = vertex_colors(mesh)[mesh.faces[fidx], :3].astype(np.float64)
    return pts, (vc * bary[..., None]).sum(axis=1)


def color_error(ref_pts, ref_cols, ref_tree, mesh):
    """Symmetric: RGB distance at nearest surface points, both directions."""
    pts, cols = colored_samples(mesh, N_SAMPLES)
    _, idx = KDTree(pts).query(ref_pts)
    _, idx2 = ref_tree.query(pts)
    diff = np.concatenate([
        np.linalg.norm(ref_cols - cols[idx], axis=1),
        np.linalg.norm(cols - ref_cols[idx2], axis=1)])
    return float(diff.mean()), float(np.percentile(diff, 95))


def render_colored(mesh, path, title):
    """fidelity_sweep.render, but faces take their vertices' painted colors."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = mesh.vertices[:, [0, 2, 1]] * [1, -1, 1]
    tris = verts[mesh.faces]
    normals = mesh.face_normals[:, [0, 2, 1]] * [1, -1, 1]
    light = np.array([0.4, -0.6, 0.7])
    light /= np.linalg.norm(light)
    shade = np.clip(normals @ light, 0, 1) * 0.45 + 0.62
    fc = vertex_colors(mesh)[mesh.faces, :3].mean(axis=1) / 255.0
    fc = np.clip(fc * shade[:, None], 0, 1)

    fig = plt.figure(figsize=(5, 5), dpi=110)
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(tris, facecolors=fc, linewidths=0))
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    center, radius = (lo + hi) / 2, (hi - lo).max() / 2 * 1.05
    for axis, c in zip('xyz', center):
        getattr(ax, f'set_{axis}lim')(c - radius, c + radius)
    ax.view_init(elev=14, azim=-65)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.1)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    print("=" * 60)
    print("STAGE 3 — COLOR FIDELITY SWEEP (vertex colors vs budget)")
    print("=" * 60)
    if not os.path.exists(COLORED_REF):
        print(f"[!] No colored reference at {COLORED_REF} — run color_hull.py first.")
        sys.exit(1)
    ref = trimesh.load(COLORED_REF, force='mesh')
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Subject: {SUBJECT_ID}   colored reference: {len(ref.faces):,} faces\n")

    print("[*] Sampling colored reference + measuring the metric's noise floor...")
    ref_pts, ref_cols = colored_samples(ref, N_SAMPLES)
    ref_tree = KDTree(ref_pts)
    floor_mean, floor_p95 = color_error(ref_pts, ref_cols, ref_tree, ref)
    print(f"    floor (reference vs itself): mean {floor_mean:.2f}   "
          f"p95 {floor_p95:.2f}   (RGB 0-255)")

    cam, dist, calib_wh = color_hull.load_calibration(color_hull.CALIB_FILE)
    hull = None
    rows = [{'level': 'reference', 'faces': len(ref.faces),
             'vertices': len(ref.vertices),
             'file_kb': round(os.path.getsize(COLORED_REF) / 1024, 1),
             'mean_rgb_err': 0.0, 'p95_rgb_err': 0.0, 'note': 'colored full-res hull'}]

    for budget in sorted(FACE_BUDGETS, reverse=True):
        lod_glb = os.path.join(OUT_DIR, f'{SUBJECT_ID}_lod_{budget}.glb')
        if os.path.exists(lod_glb):
            lod = trimesh.load(lod_glb, force='mesh')
        else:
            if hull is None:
                hull = trimesh.load(HULL_GLB, force='mesh')
            lod = decimate(hull, budget)
        print(f"[*] Painting {len(lod.faces):,}-face LOD "
              f"({len(lod.vertices):,} color samples)...")
        t0 = time.time()
        lod = paint(lod, cam, dist, calib_wh)
        out = os.path.join(OUT_DIR, f'{SUBJECT_ID}_lod_{budget}_colored.glb')
        lod.export(out)
        mean_e, p95_e = color_error(ref_pts, ref_cols, ref_tree, lod)
        rows.append({'level': f'{budget:,} budget', 'faces': len(lod.faces),
                     'vertices': len(lod.vertices),
                     'file_kb': round(os.path.getsize(out) / 1024, 1),
                     'mean_rgb_err': round(mean_e, 3),
                     'p95_rgb_err': round(p95_e, 3),
                     'note': f'painted in {time.time() - t0:.0f}s'})
        print(f"    mean RGB err {mean_e:.2f}   p95 {p95_e:.2f}   "
              f"{rows[-1]['file_kb']:,.0f} KB")
        render_colored(lod, os.path.join(OUT_DIR, f'render_color_{budget}.png'),
                       f'{len(lod.faces):,} faces, colored')

    results_path = os.path.join(OUT_DIR, 'color_sweep_results.json')
    with open(results_path, 'w') as f:
        json.dump({'subject': SUBJECT_ID, 'n_samples': N_SAMPLES,
                   'floor_mean_rgb': round(floor_mean, 3),
                   'floor_p95_rgb': round(floor_p95, 3), 'rows': rows}, f, indent=2)
    print(f"\n[*] Wrote {results_path}")


if __name__ == '__main__':
    main()
