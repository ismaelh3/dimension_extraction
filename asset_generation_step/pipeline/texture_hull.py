"""
Stage 3, M2 v2 — baked UV texture projected from the capture photos.

Decouples color resolution from polygon count: the mesh is decimated to a
delivery budget, UV-unwrapped with xatlas, and every TEXEL of a texture is
colored by projecting its 3D surface point back onto the capture photos —
the same tight-bbox orthographic projection, per-frame median, and
normal-weighted view blend as color_hull.py (whose functions are reused
directly). A 20k-face mesh then carries 4M+ color samples instead of ~10k
vertex colors.

Inputs match color_hull.py (masks/, frames/, calibration) plus the hull
master work/<SUBJECT>_hull.glb.

Usage:  SUBJECT=snowglobe venv/bin/python asset_generation_step/pipeline/texture_hull.py
        knobs: TARGET_FACES (default 20000), TEXTURE_SIZE (default 2048),
               plus color_hull's SIDE_FROM / FRAMES_DIR / BLEND_POWER

Output: work/<SUBJECT>_textured.glb   (+ _texture_debug.png of the bake)
"""

import glob
import os
import sys
from datetime import datetime

import numpy as np
import trimesh
import xatlas
from PIL import Image
from pygltflib import GLTF2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # asset_generation_step/
sys.path.insert(0, os.path.join(BASE_DIR, 'pipeline'))
sys.path.insert(0, os.path.join(BASE_DIR, 'analysis'))
import color_hull                          # noqa: E402
from fidelity_sweep import decimate        # noqa: E402

SUBJECT_ID   = os.environ.get('SUBJECT', 'product_000')
TARGET_FACES = int(os.environ.get('TARGET_FACES', '20000'))
TEXTURE_SIZE = int(os.environ.get('TEXTURE_SIZE', '2048'))
DESPECKLE    = float(os.environ.get('DESPECKLE', '0'))  # 0=off; >0 = blur radius
HULL_GLB     = os.path.join(BASE_DIR, 'work', f'{SUBJECT_ID}_hull.glb')
OUT_GLB      = os.path.join(BASE_DIR, 'work', f'{SUBJECT_ID}_textured.glb')
DEBUG_PNG    = os.path.join(BASE_DIR, 'work', f'{SUBJECT_ID}_texture_debug.png')


def _front_most(u, v, depth, res, tol):
    """Points at (or within tol of) the nearest depth for their photo pixel.

    The views are axis-aligned orthographic in normalized box coords, so a
    depth buffer needs no rasterizer: view_uv's (u, v) depend only on the two
    axes perpendicular to the view direction, so points sharing a cell lie on
    one camera ray. The texel set already covers the surface densely enough
    to serve as its own depth samples.
    """
    iu = np.clip((np.asarray(u) * (res - 1)).astype(np.int32), 0, res - 1)
    iv = np.clip((np.asarray(v) * (res - 1)).astype(np.int32), 0, res - 1)
    cell = iv * res + iu
    nearest = np.full(res * res, np.inf, np.float32)
    np.minimum.at(nearest, cell, depth.astype(np.float32))
    return depth <= nearest[cell] + tol


def visible_from(view, X, Y, Z, comp, two_sided, res=512, tol=0.01):
    """Can this view actually SEE each point, or is the surface in the way?

    Without this, a point was sampled from any view its NORMAL roughly faced
    whether or not that view could see it: the shoe's mouth interior was
    painted with the swoosh from the outer side wall, and the cap's far side
    sampled the back photo through the snapback's strap opening (which shows
    unlit interior — true black).

    `two_sided` is the subtlety. Some views are deliberately MIRRORED: with
    no back capture, front is weighted by |nz| so it also paints back faces,
    and side is always |nx| so one flank's photo serves both. Testing those
    against a single camera direction throws away exactly the samples the
    mirror is there to provide (it turned the shoe's far side into flood-fill
    blotches). So for a mirrored view, test each point against whichever of
    the two opposed directions it faces, given by `comp` (its normal's
    component along the view axis).
    """
    u, v = color_hull.view_uv(view, X, Y, Z)
    axis = {'front': Z, 'back': Z, 'top': Y, 'bottom': Y, 'side': X}[view]
    if two_sided:
        return np.where(comp >= 0,
                        _front_most(u, v, 1 - axis, res, tol),
                        _front_most(u, v, axis, res, tol))
    # single direction: which end of the axis the camera sits at
    near_high = view in ('front', 'top') or (view == 'side' and
                                             color_hull.SIDE_FROM == 'right')
    return _front_most(u, v, (1 - axis) if near_high else axis, res, tol)


def blend_points(X, Y, Z, normals, cam, dist, calib_wh, extents=None):
    """color_hull's sample-and-blend, for arbitrary surface points (the
    vertex pass runs this on vertices; the bake runs it on texel centers)."""
    colors, weights = {}, {}
    nx, ny, nz = normals.T
    for view in color_hull.VIEWS:
        col = color_hull.sample_view(view, X, Y, Z, cam, dist, calib_wh,
                                     extents=extents)
        if col is None:
            continue
        # NaN = this view has no product pixel for that point (see
        # color_hull.sample_view). Keep the color array finite and let the
        # weight carry the validity, so an off-product sample contributes
        # nothing instead of dragging the blend toward the background.
        seen = np.isfinite(col).all(axis=1)
        colors[view] = np.nan_to_num(col)
        # `comp` is the normal's component along the view axis; `two_sided`
        # marks the views weighted by |comp| — i.e. deliberately mirrored to
        # cover the faces we never photographed (see visible_from).
        if view == 'front':
            has_back = 'back' in colors or glob.glob(
                os.path.join(color_hull.MASKS_DIR, 'back', '*.png'))
            two_sided = not has_back
            comp = nz
            weights[view] = (np.maximum(nz, 0) if has_back
                             else np.abs(nz)) ** color_hull.BLEND_POWER
        elif view == 'back':
            two_sided, comp = False, -nz
            weights[view] = np.maximum(-nz, 0) ** color_hull.BLEND_POWER
        elif view == 'side':
            # comp is along +X (visible_from's convention); SIDE_FROM picks
            # which photo is mirrored, not which way the geometry faces
            two_sided, comp = True, nx
            weights[view] = np.abs(nx) ** color_hull.BLEND_POWER
        elif view == 'top':
            two_sided, comp = False, ny
            weights[view] = np.maximum(ny, 0) ** color_hull.BLEND_POWER
        elif view == 'bottom':
            two_sided, comp = False, -ny
            weights[view] = np.maximum(-ny, 0) ** color_hull.BLEND_POWER
        weights[view] = weights[view] * seen * visible_from(
            view, X, Y, Z, comp, two_sided)
    if 'front' not in colors:
        print("[!] Front view is required for the texture bake.")
        sys.exit(1)
    total = sum(weights.values())
    # DEPTH_CARVE cavities can expose a texel whose normal isn't favorably
    # aligned with ANY of the 4 orthogonal capture directions (e.g. a floor
    # patch occluded from every shot by the collar rim above it) — total=0
    # there. Report which: the caller must NOT mark these "filled", or the
    # forced blend=0 bakes a black hole (found on a shoe's mouth floor) that
    # dilate_texture's neighbor-flood never gets a chance to fix, since it
    # only patches texels outside the UV islands, not degenerate ones inside.
    has_data = total > 0
    total[~has_data] = 1
    blend = sum(colors[v] * (weights[v] / total)[:, None] for v in colors)
    return np.clip(blend, 0, 255), sorted(colors), has_data


def rasterize_uv(faces, uvs, tsize):
    """For every texel covered by a UV triangle, return the texel indices
    (ty, tx), the face it belongs to, and its barycentric weights."""
    tex_ty, tex_tx, tex_face, tex_bary = [], [], [], []
    uvpix = uvs * (tsize - 1)
    for f, (i0, i1, i2) in enumerate(faces):
        (x0, y0), (x1, y1), (x2, y2) = uvpix[[i0, i1, i2]]
        xmin, xmax = int(min(x0, x1, x2)), int(np.ceil(max(x0, x1, x2)))
        ymin, ymax = int(min(y0, y1, y2)), int(np.ceil(max(y0, y1, y2)))
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(det) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(xmin, xmax + 1),
                             np.arange(ymin, ymax + 1))
        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / det
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / det
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue
        tex_tx.append(gx[inside])
        tex_ty.append(gy[inside])
        tex_face.append(np.full(inside.sum(), f, np.int32))
        tex_bary.append(np.stack([w0[inside], w1[inside], w2[inside]], axis=1))
    return (np.concatenate(tex_ty), np.concatenate(tex_tx),
            np.concatenate(tex_face), np.concatenate(tex_bary))


def dilate_texture(tex, filled):
    """Flood every unfilled texel with its nearest filled color so bilinear
    sampling at island borders never picks up background."""
    from scipy.ndimage import distance_transform_edt
    _, (iy, ix) = distance_transform_edt(~filled, return_indices=True)
    return tex[iy, ix]


def despeckle_atlas(tex, blur):
    """Optional post-bake cleanup: replace ISOLATED near-black texels (the
    black/gold flecks at silhouette edges + neighbor-filled cavities) with their
    local median, leaving large contiguous dark regions (a real dark back)
    intact, then a light blur for a smooth figurine surface. A texel is a speck
    if it is near-black yet its neighborhood median is not."""
    from PIL import ImageFilter
    med = np.asarray(Image.fromarray(tex).filter(ImageFilter.MedianFilter(5)),
                     np.uint8)
    s = tex.astype(int).sum(2)
    med_s = med.astype(int).sum(2)
    speck = (s < 95) & (med_s > 150)
    out = tex.copy()
    out[speck] = med[speck]
    print(f"    despeckled {int(speck.sum()):,} texels, blur {blur}")
    return np.asarray(Image.fromarray(out).filter(ImageFilter.GaussianBlur(blur)),
                      np.uint8)


def main():
    print("=" * 60)
    print("STAGE 3 — UV TEXTURE BAKE (M2 v2)")
    print("=" * 60)
    print(f"Subject: {SUBJECT_ID}   target: {TARGET_FACES:,} faces, "
          f"{TEXTURE_SIZE}px texture\n")

    hull = trimesh.load(HULL_GLB, force='mesh')
    src_extras = GLTF2().load(HULL_GLB).scenes[0].extras or {}
    print(f"[*] Decimating {len(hull.faces):,} -> {TARGET_FACES:,} faces...")
    was_watertight = hull.is_watertight
    mesh = decimate(hull, TARGET_FACES)
    # A 100x quadric-decimation pass can lose manifoldness on thin,
    # high-curvature features it collapses too aggressively — seen in
    # practice on a DEPTH_CARVE cavity wall (one side of a shoe's mouth came
    # out with a hole after decimation, though the un-decimated hull was
    # watertight). fill_holes patches the small gaps decimation introduces;
    # it's a no-op when there's nothing to patch.
    if was_watertight and not mesh.is_watertight:
        mesh.fill_holes()
    mesh.fix_normals()
    if was_watertight and not mesh.is_watertight:
        print(f"    [!] decimation broke watertightness and fill_holes "
              f"couldn't fully repair it — inspect the bake for gaps")

    print("[*] UV-unwrapping with xatlas...")
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    verts = mesh.vertices[vmapping]
    vnormals = mesh.vertex_normals[vmapping]
    print(f"    {len(verts):,} UV vertices, {len(indices):,} faces")

    print("[*] Rasterizing UV islands...")
    ty, tx, tface, tbary = rasterize_uv(indices, uvs, TEXTURE_SIZE)
    tri = verts[indices[tface]]                       # (N, 3, 3)
    pos = (tri * tbary[..., None]).sum(axis=1)
    nrm = (vnormals[indices[tface]] * tbary[..., None]).sum(axis=1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    print(f"    {len(pos):,} texels to bake "
          f"({len(pos) / TEXTURE_SIZE**2:.0%} of the atlas)")

    # same normalized-box coordinates the carve and color pass use
    lo, hi = mesh.bounds
    X, Y, Z = ((pos - lo) / (hi - lo)).T

    print("[*] Projecting texels onto the capture photos...")
    cam, dist, calib_wh = color_hull.load_calibration(color_hull.CALIB_FILE)
    colors, views_used, has_data = blend_points(X, Y, Z, nrm, cam, dist, calib_wh,
                                                extents=hi - lo)
    if (~has_data).any():
        print(f"    [!] {(~has_data).sum():,} texel(s) not favorably seen by "
              f"any view (likely a DEPTH_CARVE cavity wall) — filled from "
              f"nearest neighbor instead of baking them black")

    tex = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), np.uint8)
    filled = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE), bool)
    # v axis: glTF UV origin is top-left, xatlas uvs are in [0,1] with v up
    row = (TEXTURE_SIZE - 1) - ty
    tex[row, tx] = colors.astype(np.uint8)
    filled[row, tx] = has_data
    print("[*] Dilating island borders...")
    tex = dilate_texture(tex, filled)
    if DESPECKLE > 0:
        print("[*] Despeckling atlas...")
        tex = despeckle_atlas(tex, DESPECKLE)
    Image.fromarray(tex).save(DEBUG_PNG)

    print("[*] Exporting glb...")
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(tex),
        metallicFactor=0.0, roughnessFactor=0.6)
    out = trimesh.Trimesh(vertices=verts, faces=indices, process=False,
                          visual=trimesh.visual.TextureVisuals(
                              uv=uvs, material=material))
    out.export(OUT_GLB)

    g = GLTF2().load(OUT_GLB)
    src_extras['color_pass'] = {
        'method':       'uv_texture_v2_median_projection',
        'views':        views_used,
        'blend_power':  color_hull.BLEND_POWER,
        'target_faces': TARGET_FACES,
        'texture_size': TEXTURE_SIZE,
        'colored_at':   datetime.now().isoformat(),
    }
    g.scenes[g.scene or 0].extras = src_extras
    g.save(OUT_GLB)

    size_mb = os.path.getsize(OUT_GLB) / 1024 / 1024
    print(f"\n[*] Wrote {OUT_GLB}  ({size_mb:.1f} MB)")
    print(f"    views used: {', '.join(views_used)}   bake atlas: {DEBUG_PNG}")
    print("\nDone. Inspect at https://gltf-viewer.donmccurdy.com.")


if __name__ == '__main__':
    main()
