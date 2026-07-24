"""
Stage 3, M2 v1 — vertex colors projected from the capture photos.

Paints work/<SUBJECT>_hull.glb by projecting each vertex back onto the
original photos, using the same tight-bbox normalization the carve used:
a vertex's position inside the mesh bounding box maps to the same relative
position inside each view's mask bounding box, and the photo is sampled
there. Per view, every frame is sampled and the per-vertex MEDIAN across
frames is kept — moving specular reflections (glass!) get voted out, same
idea as the mask voting in the carve. Views are then blended per vertex by
how squarely its surface normal faces each camera.

Frames must be the RAW capture photos (masks were computed on undistorted
frames, so the identical undistortion — mirrored from segmentation.py — is
applied here before sampling). Views with masks but no matching frames are
skipped with a warning. Missing back/left views mirror front/side colors
automatically (orthographic sampling ignores the depth axis).

Usage:  SUBJECT=snowglobe make color-asset
        FRAMES_DIR=path/to/frames SUBJECT=x venv/bin/python \
            asset_generation_step/pipeline/color_hull.py

Output: work/<SUBJECT>_hull_colored.glb — geometry untouched, provenance
        extras carried over from the input hull plus a color_pass block.
"""

import glob
import os
import pickle
import sys
import warnings
from datetime import datetime

import cv2
import numpy as np
import trimesh
from pygltflib import GLTF2

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # asset_generation_step/
sys.path.insert(0, os.path.join(BASE_DIR, 'pipeline'))
# single source of truth for the TOPDOWN_FRONT env knob (and its parsing):
# the carve rotates 90°-off top/bottom masks, so sampling must un-rotate
# with the exact same per-view direction or colors land transposed
from build_silhouette_mesh import TOPDOWN_FRONT_SIDES  # noqa: E402
SUBJECT_ID = os.environ.get('SUBJECT', 'product_000')
SIDE_FROM  = os.environ.get('SIDE_FROM', 'right')
FRAMES_DIR = os.environ.get('FRAMES_DIR', os.path.join(
    BASE_DIR, '..', 'instance_segmentation_step', 'frames'))
CALIB_FILE = os.path.join(BASE_DIR, '..', 'camera_calibration_step',
                          'output', 'calibration_data.pkl')
MASKS_DIR  = os.path.join(BASE_DIR, 'masks', SUBJECT_ID)
HULL_GLB   = os.path.join(BASE_DIR, 'work', f'{SUBJECT_ID}_hull.glb')
OUT_GLB    = os.path.join(BASE_DIR, 'work', f'{SUBJECT_ID}_hull_colored.glb')

# how sharply the view blend follows the normal: higher = crisper ownership
# per face, lower = softer transitions (and more cross-view ghosting)
BLEND_POWER = float(os.environ.get('BLEND_POWER', '2'))

VIEWS = ('front', 'side', 'back', 'top', 'bottom')


# ------------------------------------------------- undistortion (mirrors
# instance_segmentation_step/segmentation.py — masks were made on undistorted
# frames, so sampling must happen in the exact same pixel space)

def load_calibration(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return (data['camera_matrix'], data['distortion_coefficients'],
            data.get('image_size_wh'))


def scale_matrix_to_frame(camera_matrix, calib_size_wh, frame_size_wh):
    if calib_size_wh is None or tuple(frame_size_wh) == tuple(calib_size_wh):
        return camera_matrix
    sx = frame_size_wh[0] / calib_size_wh[0]
    sy = frame_size_wh[1] / calib_size_wh[1]
    scaled = camera_matrix.copy()
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


def undistort_frame(frame, camera_matrix, dist_coeffs):
    h, w = frame.shape[:2]
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1, newImgSize=(w, h)
    )
    return cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_matrix)


# ------------------------------------------------- projection

def view_uv(view, X, Y, Z):
    """Normalized [0,1] photo coordinates (u along cols, v along rows) for
    vertices at normalized box coordinates X (width), Y (height), Z (depth,
    1 = front face). Inverse of build_silhouette_mesh.orient_to_grid — see
    that docstring for the per-view camera conventions."""
    if view == 'front':
        return X, 1 - Y
    if view == 'back':
        return 1 - X, 1 - Y
    if view == 'side':
        return (1 - Z, 1 - Y) if SIDE_FROM == 'right' else (Z, 1 - Y)
    if view == 'top':
        return X, Z
    if view == 'bottom':
        return 1 - X, Z
    raise ValueError(view)


def sample_view(view, X, Y, Z, cam, dist, calib_wh, extents=None):
    """Median per-vertex color across all of a view's frames, or None.

    extents (mesh W/H/D in metres) enables the same transposed-photo
    detection the carve does: a top/bottom mask whose tight bbox matches the
    transposed aspect gets its sample coordinates un-rotated (the inverse of
    build_silhouette_mesh.fix_topdown_rotation, per TOPDOWN_FRONT)."""
    mask_paths = sorted(glob.glob(os.path.join(MASKS_DIR, view, '*.png')))
    if not mask_paths:
        return None
    u, v = view_uv(view, X, Y, Z)
    expected = None
    if extents is not None and view in ('top', 'bottom'):
        expected = extents[0] / extents[2]                # W/D, photo cols/rows
        if 0.8 < expected < 1.25:
            expected = None                               # near-square: ambiguous
    samples, unrotated = [], 0
    for mp in mask_paths:
        stem = os.path.basename(mp).replace('_product_mask.png', '')
        # recursive so FRAMES_DIR can be a per-view capture ROOT (front/, side/,
        # ...) OR a flat folder — either way each mask's frame is found by stem.
        hits = (glob.glob(os.path.join(FRAMES_DIR, stem + '.*'))
                or glob.glob(os.path.join(FRAMES_DIR, '**', stem + '.*'),
                             recursive=True))
        if not hits:
            print(f"    [!] no frame for {stem} — skipped")
            continue
        frame = cv2.imread(hits[0])
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None or frame.shape[:2] != mask.shape:
            print(f"    [!] {stem}: frame/mask unreadable or size mismatch — skipped")
            continue
        h, w = frame.shape[:2]
        matrix = scale_matrix_to_frame(cam, calib_wh, (w, h))
        frame = undistort_frame(frame, matrix, dist)
        x0, y0, bw, bh = cv2.boundingRect(cv2.findNonZero((mask > 127).astype(np.uint8)))
        uf, vf = u, v
        if expected is not None and \
                abs(np.log((bw / bh) / (1 / expected))) < abs(np.log((bw / bh) / expected)):
            # photo is 90° off convention — sample where the carve's rotation
            # would have read from: ccw rotation k=1 inverts as (1-v, u),
            # cw k=3 as (v, 1-u)
            if TOPDOWN_FRONT_SIDES[view] == 'left':
                uf, vf = 1 - v, u
            else:
                uf, vf = v, 1 - u
            unrotated += 1
        cols = np.clip(x0 + uf * (bw - 1), 0, w - 1).astype(np.int32)
        rows = np.clip(y0 + vf * (bh - 1), 0, h - 1).astype(np.int32)
        samp = frame[rows, cols][:, ::-1].astype(np.float32)   # BGR -> RGB
        # Only PRODUCT pixels are a valid sample. Nothing constrained this
        # before, so a point projecting off the product took whatever was
        # behind it — and FILL_HOLES makes that routine: it seals a gap in
        # the MASK (a snapback's strap opening) while the PHOTO still shows
        # the dark table through it, so cavity texels sampling there baked
        # black. NaN marks "no data from this view"; callers must drop it
        # from the blend rather than average it in.
        samp[mask[rows, cols] <= 127] = np.nan
        samples.append(samp)
    if not samples:
        return None
    note = (f"  ({unrotated} transposed frame(s) un-rotated, "
            f"TOPDOWN_FRONT {view}:{TOPDOWN_FRONT_SIDES[view]})" if unrotated else '')
    print(f"    {view:<6} — {len(samples)} frame(s) sampled{note}")
    with warnings.catch_warnings():                # all-NaN = seen by no frame
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(np.stack(samples), axis=0).astype(np.float32)


# ------------------------------------------------- main

def main():
    print("=" * 60)
    print("STAGE 3 — VERTEX COLOR PASS (M2 v1)")
    print("=" * 60)
    print(f"Subject: {SUBJECT_ID}\n")

    if not os.path.exists(HULL_GLB):
        print(f"[!] No hull at {HULL_GLB} — run build_silhouette_mesh.py first.")
        sys.exit(1)
    source_extras = GLTF2().load(HULL_GLB).scenes[0].extras or {}

    mesh = trimesh.load(HULL_GLB, force='mesh')
    verts, normals = mesh.vertices, mesh.vertex_normals
    lo, hi = mesh.bounds
    X, Y, Z = ((verts - lo) / (hi - lo)).T
    print(f"[*] {len(verts):,} vertices to paint")

    cam, dist, calib_wh = load_calibration(CALIB_FILE)

    print("[*] Sampling photos (undistorted, median across frames)...")
    colors, weights = {}, {}
    nx, ny, nz = normals.T
    for view in VIEWS:
        col = sample_view(view, X, Y, Z, cam, dist, calib_wh, extents=hi - lo)
        if col is None:
            continue
        # NaN = no product pixel for that vertex in this view; the weight
        # carries the validity so it contributes nothing (see sample_view).
        seen = np.isfinite(col).all(axis=1)
        colors[view] = np.nan_to_num(col)
        if view == 'front':
            # |nz|: back faces mirror the front photo unless a back set exists
            weights[view] = (np.maximum(nz, 0) if 'back' in colors or
                             glob.glob(os.path.join(MASKS_DIR, 'back', '*.png'))
                             else np.abs(nz)) ** BLEND_POWER
        elif view == 'back':
            weights[view] = np.maximum(-nz, 0) ** BLEND_POWER
        elif view == 'side':
            weights[view] = np.abs(nx) ** BLEND_POWER
        elif view == 'top':
            weights[view] = np.maximum(ny, 0) ** BLEND_POWER
        elif view == 'bottom':
            weights[view] = np.maximum(-ny, 0) ** BLEND_POWER
        weights[view] = weights[view] * seen
    if 'front' not in colors:
        print("[!] Front view is required for the color pass.")
        sys.exit(1)

    print("[*] Blending views by vertex normal...")
    total = sum(weights.values())
    total[total == 0] = 1
    blend = sum(colors[v] * (weights[v] / total)[:, None] for v in colors)
    rgba = np.concatenate([np.clip(blend, 0, 255).astype(np.uint8),
                           np.full((len(verts), 1), 255, np.uint8)], axis=1)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=rgba)

    mesh.export(OUT_GLB)
    g = GLTF2().load(OUT_GLB)
    source_extras['color_pass'] = {
        'method':      'vertex_colors_v1_median_projection',
        'views':       sorted(colors),
        'blend_power': BLEND_POWER,
        'colored_at':  datetime.now().isoformat(),
    }
    g.scenes[g.scene or 0].extras = source_extras
    g.save(OUT_GLB)

    print(f"\n[*] Wrote {OUT_GLB}")
    print(f"    views used: {', '.join(sorted(colors))}")
    print("\nDone. Inspect at https://gltf-viewer.donmccurdy.com.")


if __name__ == '__main__':
    main()
