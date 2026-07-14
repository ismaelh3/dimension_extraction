"""
Stage 3 experiment — polygon-budget (fidelity) sweep.

Takes the full-resolution hull (work/<SUBJECT>_hull.glb) as ground truth and
decimates it to a ladder of face budgets — the "initial amount" and the
progressively "more in depth" versions the supervisor asked for. For each
level it measures how far the simplified surface strays from the reference,
so the polygon count becomes a number you can reason about instead of a vibe.

Also includes one deliberately instructive row: the coarsest level
loop-SUBDIVIDED back up to a high face count. Subdivision multiplies
polygons but cannot re-create detail that decimation threw away — the error
stays at the coarse level. Fidelity comes from information kept, not from
triangle count alone.

Error metric: symmetric chamfer distance between dense surface samplings of
the reference and each level (mean + p95, reported in mm). No rtree needed —
plain KD-trees on sampled point clouds.

Usage:  SUBJECT=snowglobe venv/bin/python asset_generation_step/analysis/fidelity_sweep.py

Output: work/lods/<SUBJECT>_lod_<faces>.glb        one asset per level
        work/lods/render_*.png                     same-camera renders
        work/lods/sweep_results.json               the numbers
"""

import json
import os
import time

import numpy as np
import trimesh
from scipy.spatial import KDTree

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # asset_generation_step/
SUBJECT_ID  = os.environ.get('SUBJECT', 'product_000')
HULL_GLB    = os.path.join(BASE_DIR, 'work', f'{SUBJECT_ID}_hull.glb')
OUT_DIR     = os.path.join(BASE_DIR, 'work', 'lods')

# the ladder: initial coarse budget -> progressively more polygons
FACE_BUDGETS = [int(v) for v in os.environ.get(
    'FACE_BUDGETS', '1000,5000,20000,80000,320000').split(',')]
N_SAMPLES = 50_000   # surface points per mesh for the chamfer measurement


def decimate(mesh, target_faces):
    """Same recipe as build_silhouette_mesh.voxels_to_mesh: float32 vertices
    so fast-simplification converges in one pass, loop as a safety net."""
    m = mesh.copy()
    m.vertices = m.vertices.astype(np.float32).astype(np.float64)
    while len(m.faces) > target_faces:
        before = len(m.faces)
        m = m.simplify_quadric_decimation(face_count=target_faces)
        if len(m.faces) >= before:
            break
    return m


def chamfer_mm(ref_pts, ref_tree, mesh):
    """Symmetric point-cloud chamfer distance, reference <-> mesh, in mm."""
    pts = mesh.sample(N_SAMPLES)
    tree = KDTree(pts)
    d_ref_to_lod, _ = tree.query(ref_pts)     # where did the surface move to
    d_lod_to_ref, _ = ref_tree.query(pts)     # did the lod invent surface
    both = np.concatenate([d_ref_to_lod, d_lod_to_ref]) * 1000.0
    return float(both.mean()), float(np.percentile(both, 95))


def render(mesh, path, title, show_edges):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # mesh is +Y up / +Z front, matplotlib's 3D axes are +Z up:
    # remap (X, Y, Z) -> (X, -Z, Y) so the product stands upright
    verts = mesh.vertices[:, [0, 2, 1]] * [1, -1, 1]
    tris = verts[mesh.faces]
    normals = mesh.face_normals[:, [0, 2, 1]] * [1, -1, 1]
    # simple headlight lambert shading off the face normals
    light = np.array([0.4, -0.6, 0.7])
    light /= np.linalg.norm(light)
    shade = np.clip(normals @ light, 0, 1) * 0.75 + 0.22
    colors = np.repeat(shade[:, None], 3, axis=1)

    fig = plt.figure(figsize=(5, 5), dpi=110)
    ax = fig.add_subplot(111, projection='3d')
    coll = Poly3DCollection(tris, facecolors=colors,
                            edgecolors='#1a3550' if show_edges else None,
                            linewidths=0.15 if show_edges else 0)
    ax.add_collection3d(coll)
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


def measure(tag, mesh, ref_pts, ref_tree, glb_path, note=''):
    mesh.export(glb_path)
    mean_mm, p95_mm = chamfer_mm(ref_pts, ref_tree, mesh)
    row = {
        'level':     tag,
        'faces':     len(mesh.faces),
        'vertices':  len(mesh.vertices),
        'file_kb':   round(os.path.getsize(glb_path) / 1024, 1),
        'mean_err_mm': round(mean_mm, 4),
        'p95_err_mm':  round(p95_mm, 4),
        'note':      note,
    }
    print(f"    {tag:<22} {row['faces']:>9,} faces  {row['file_kb']:>9,.0f} KB"
          f"   mean {mean_mm:.3f} mm   p95 {p95_mm:.3f} mm")
    return row


def main():
    print("=" * 60)
    print("STAGE 3 — POLYGON-BUDGET FIDELITY SWEEP")
    print("=" * 60)
    ref = trimesh.load(HULL_GLB, force='mesh')
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Subject: {SUBJECT_ID}   reference: {len(ref.faces):,} faces\n")

    print("[*] Sampling reference surface...")
    ref_pts = ref.sample(N_SAMPLES)
    ref_tree = KDTree(ref_pts)

    rows = [{
        'level': 'reference', 'faces': len(ref.faces),
        'vertices': len(ref.vertices),
        'file_kb': round(os.path.getsize(HULL_GLB) / 1024, 1),
        'mean_err_mm': 0.0, 'p95_err_mm': 0.0, 'note': 'full-res hull',
    }]
    print(f"    {'reference':<22} {len(ref.faces):>9,} faces  "
          f"{rows[0]['file_kb']:>9,.0f} KB   (ground truth)")

    print("[*] Decimating down the ladder...")
    lods = {}
    for budget in sorted(FACE_BUDGETS, reverse=True):
        t0 = time.time()
        lod = decimate(ref, budget)
        lods[budget] = lod
        glb = os.path.join(OUT_DIR, f'{SUBJECT_ID}_lod_{budget}.glb')
        rows.append(measure(f'{budget:,} budget', lod, ref_pts, ref_tree, glb,
                            note=f'decimated in {time.time() - t0:.1f}s'))

    # the lesson row: subdivide the coarsest LOD back up — polygons without detail
    coarsest = min(FACE_BUDGETS)
    print("[*] Subdividing the coarsest level back up (the trap)...")
    sub = lods[coarsest].copy()
    while len(sub.faces) < max(FACE_BUDGETS) / 4:
        sub = sub.subdivide_loop(iterations=1)
    glb = os.path.join(OUT_DIR, f'{SUBJECT_ID}_lod_{coarsest}_subdivided.glb')
    rows.append(measure(f'{coarsest:,} subdivided', sub, ref_pts, ref_tree, glb,
                        note='loop-subdivided from coarsest LOD — polygons '
                             'went up, detail did not come back'))

    print("[*] Rendering previews (same camera for every level)...")
    for budget, lod in sorted(lods.items()):
        render(lod, os.path.join(OUT_DIR, f'render_{budget}.png'),
               f'{len(lod.faces):,} faces', show_edges=len(lod.faces) <= 6_000)
    render(sub, os.path.join(OUT_DIR, 'render_subdivided.png'),
           f'{len(sub.faces):,} faces (subdivided from {coarsest:,})',
           show_edges=False)

    results_path = os.path.join(OUT_DIR, 'sweep_results.json')
    with open(results_path, 'w') as f:
        json.dump({'subject': SUBJECT_ID, 'n_samples': N_SAMPLES,
                   'rows': rows}, f, indent=2)
    print(f"\n[*] Wrote {results_path}")
    print(f"    LOD assets + renders in {OUT_DIR}")


if __name__ == '__main__':
    main()
