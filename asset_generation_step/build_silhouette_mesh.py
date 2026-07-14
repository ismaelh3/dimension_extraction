"""
Stage 3, Route C — silhouette reconstruction (visual hull).

Builds a real-world-scaled 3D mesh (.glb) for a product by carving a voxel
grid — sized exactly to the Stage 2 measurements — with the product's
segmentation masks. Supports any subset of four views:

    front  (required)   side | back | top | bottom   (optional, each tightens the hull)

Note: top and bottom produce the SAME carving constraint (an object's silhouette
along the vertical axis is identical from above and below, mirrored) — one of the
two is enough for geometry; providing both just makes that silhouette more robust.

Masks are consumed from  masks/<SUBJECT>/<view>/*.png  (the 0/255 PNGs the
segmentation step writes). Copy each capture set's masks there after running
the segmentation step on it — extra views (back/top) only need segmentation,
not depth/measurement, since scale comes entirely from the measurements JSON.

Usage:  SUBJECT=snowglobe CROSS_SECTION=round make build-asset
        SUBJECT=snowglobe RESOLUTION=768 SIDE_FROM=left FILL_HOLES=all \
            venv/bin/python asset_generation_step/build_silhouette_mesh.py

Output: work/<SUBJECT>_hull.glb  — metres, +Y up, front facing +Z,
        origin at bottom-center, provenance embedded in glTF extras.
"""

import glob
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import trimesh
from pygltflib import GLTF2
from skimage import measure

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SUBJECT_ID = os.environ.get('SUBJECT', 'product_000')
RESOLUTION = int(os.environ.get('RESOLUTION', '512'))   # voxels along the longest axis
# Which side of the product the 'side' set was shot from. Getting this wrong
# only mirrors the depth profile front-to-back — fix it here, not by reshooting.
SIDE_FROM  = os.environ.get('SIDE_FROM', 'right')
# Reflective/transparent products (glass, gloss) make SAM punch holes in the
# middle of a mask; carving would extrude those holes as tunnels through the
# hull. Views listed here get every fully-enclosed hole filled per frame.
# Objects with a REAL through-hole visible in a view (e.g. a mug handle seen
# from the front) must exclude that view. Accepts a comma-separated subset of
# front,side,back,top,bottom, or 'all' / 'none'.
FILL_HOLES = os.environ.get('FILL_HOLES', 'top,bottom')
# Perpendicular silhouettes can't carve a round object round — three circles
# intersect as a prism that bulges up to ~22% proud of the true sphere on the
# diagonals. 'round' adds a lathe constraint for rotationally symmetric
# products (globes, bottles, jars): at each height, voxels must fall inside
# the ellipse spanned by that height's front half-width and side half-depth.
# Keep 'silhouette' (default) for box-like/asymmetric products — 'round'
# would shave their corners off.
CROSS_SECTION = os.environ.get('CROSS_SECTION', 'silhouette')

VOTE_FRACTION     = 0.5   # pixel kept if inside >= this fraction of a view's masks
# taubin iterations: 10 leaves faint voxel-staircase ridges visible on smooth
# curved products; ~30 flattens them (finalize()'s exact-bbox rescale undoes
# the extra shrinkage, so more iterations cost time, not size accuracy)
SMOOTH_ITERATIONS = int(os.environ.get('SMOOTH_ITERATIONS', '10'))
# CROSS_SECTION=round only: gaussian sigma (in voxel rows) applied to the
# lathe's per-height profile (centers + radii). Mask-edge noise jitters each
# row's span independently, which reads as horizontal rings on the surface;
# smoothing the profile removes the rings while following the real shape.
# 0 = off. ~2 is enough; big values start rounding real grooves away.
PROFILE_SMOOTH    = float(os.environ.get('PROFILE_SMOOTH', '0'))
# gaussian sigma (voxels) applied to each view's vote fraction BEFORE the
# 0.5 threshold. Regularizes the silhouette boundary at sub-voxel precision:
# kills the wiggles and small reflection artifacts that per-mask edge noise
# leaves after voting, without eroding the true outline (a blurred step
# edge still crosses 0.5 at the same place). 0 = off; ~2 is plenty.
SIL_SMOOTH        = float(os.environ.get('SIL_SMOOTH', '0'))
# gaussian sigma (voxels) applied to the 3D occupancy volume before marching
# cubes. On a hard 0/1 volume the iso-surface can only sit on voxel-cube
# boundaries, which leaves shallow terraces ("onion rings") that mesh
# smoothing never fully erases; a blurred volume lets the 0.5 iso-surface
# pass BETWEEN voxels, so the mesh comes out terrace-free at the source.
# 0 = off; ~1.5 is plenty (the blur support stays inside ~2 voxels ≈ the
# carve's own resolution, so no real detail is lost).
VOLUME_SMOOTH     = float(os.environ.get('VOLUME_SMOOTH', '0'))
MIN_MASK_AREA_PX  = 100
# quality-first: no decimation by default — set TARGET_FACES to cap triangle
# count only when a delivery target (e.g. web/AR) demands it
TARGET_FACES      = int(os.environ.get('TARGET_FACES', '0'))

MASKS_DIR         = os.path.join(BASE_DIR, 'masks', SUBJECT_ID)
WORK_DIR          = os.path.join(BASE_DIR, 'work')
MEASUREMENTS_JSON = os.path.join(BASE_DIR, '..', 'measurement_extraction_step',
                                 'output', f'measurements_{SUBJECT_ID}.json')

VIEWS = ('front', 'side', 'back', 'top', 'bottom')

_fill = FILL_HOLES.strip().lower()
FILL_HOLES_VIEWS = (set(VIEWS) if _fill == 'all'
                    else set() if _fill in ('', 'none')
                    else {v.strip() for v in _fill.split(',')})
_unknown = FILL_HOLES_VIEWS - set(VIEWS)
if _unknown:
    print(f"[!] FILL_HOLES names unknown view(s) {sorted(_unknown)} — "
          f"valid: {', '.join(VIEWS)}, or all/none.")
    sys.exit(1)
if CROSS_SECTION not in ('silhouette', 'round'):
    print(f"[!] CROSS_SECTION must be 'silhouette' or 'round', "
          f"not '{CROSS_SECTION}'.")
    sys.exit(1)


# ---------------------------------------------------------------- inputs

def load_measurements():
    if not os.path.exists(MEASUREMENTS_JSON):
        print(f"[!] No measurements JSON for '{SUBJECT_ID}' at {MEASUREMENTS_JSON}")
        print("    Run the Stage 2 pipeline (and merge-views) for this subject first.")
        sys.exit(1)
    with open(MEASUREMENTS_JSON) as f:
        meta = json.load(f)
    dims_cm = meta.get('measurements_cm', {})
    missing = [k for k in ('width', 'height', 'depth') if not dims_cm.get(k)]
    if missing:
        print(f"[!] measurements_cm is missing {missing} — the voxel grid needs all three.")
        print("    Depth requires a side-view capture set merged via merge_views.py.")
        sys.exit(1)
    cross = meta.get('height_cross_check', {})
    if cross and cross.get('consistent') is False:
        print("[!] Stage 2 height_cross_check failed for this subject — fix the")
        print("    measurement before generating an asset from it.")
        sys.exit(1)
    return dims_cm, meta


def tight_crop(mask_bool):
    """Crop a boolean mask to its tight bounding box (same idea as Stage 2's
    get_mask_tight_bbox). Returns None for empty/near-empty masks."""
    pts = cv2.findNonZero(mask_bool.astype(np.uint8))
    if pts is None or len(pts) < MIN_MASK_AREA_PX:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    return mask_bool[y:y + h, x:x + w]


def fill_holes(mask_bool):
    """Fill background regions fully enclosed by foreground (specular /
    reflection dropouts). Redraws each blob's outer contour solid, so no
    foreground pixel is ever removed."""
    cnts, _ = cv2.findContours(mask_bool.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask_bool.shape, np.uint8)
    cv2.drawContours(filled, cnts, -1, 1, cv2.FILLED)
    return filled.astype(bool)


def load_view_silhouette(view, rows, cols):
    """Combine every mask PNG for a view into one (rows, cols) silhouette.

    Each mask is hole-filled (if the view is in FILL_HOLES), tight-cropped,
    stretched onto the face resolution, then majority-voted across frames —
    normalising to the bounding box makes per-frame position/distance
    differences irrelevant.
    Returns a boolean array, or None if the view has no masks.
    """
    paths = sorted(glob.glob(os.path.join(MASKS_DIR, view, '*.png')))
    if not paths:
        return None
    acc, used = np.zeros((rows, cols), np.float32), 0
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"    [!] unreadable mask skipped: {os.path.basename(p)}")
            continue
        mask = img > 127
        if view in FILL_HOLES_VIEWS:
            mask = fill_holes(mask)
        cropped = tight_crop(mask)
        if cropped is None:
            print(f"    [!] empty mask skipped: {os.path.basename(p)}")
            continue
        acc += cv2.resize(cropped.astype(np.float32), (cols, rows),
                          interpolation=cv2.INTER_AREA)
        used += 1
    if used == 0:
        return None
    prob = acc / used
    if SIL_SMOOTH > 0:
        prob = cv2.GaussianBlur(prob, (0, 0), SIL_SMOOTH)
    sil = prob >= VOTE_FRACTION
    # small open+close pass kills speckles / hairline leaks without moving edges
    k = max(3, RESOLUTION // 86) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    sil = cv2.morphologyEx(sil.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, kernel).astype(bool)
    note = '  (holes filled)' if view in FILL_HOLES_VIEWS else ''
    print(f"    {view:<5} — {used} mask(s) combined{note}")
    return sil


# ---------------------------------------------------------------- carving

def orient_to_grid(sil, view):
    """Map a view's silhouette (image coords: row down, col right) onto grid
    axes. Grid is [ix, iy, iz] = [+X width, +Y up, +Z toward the front].

    Derived from camera position per view (image col = camera's rightward axis):
      front: col -> +X            back: col -> -X   (mirrored)
      side (from product's right): col -> -Z; from left: col -> +Z
      top  (shot standing at the front, looking straight down): col -> +X,
           row -> +Z (image bottom = product front)
      bottom (product ROLLED upside down about its front-back axis, so its
           front still faces the photographer; shot straight down): the roll
           mirrors left/right, so col -> -X, row -> +Z
    Vertical views (front/side/back): row -> -Y, so rows get flipped.
    """
    if view == 'front':
        return np.flipud(sil).T                     # [ix, iy]
    if view == 'back':
        return np.flipud(np.fliplr(sil)).T          # [ix, iy]
    if view == 'side':
        if SIDE_FROM == 'right':
            return np.flipud(np.fliplr(sil)).T      # [iz, iy]
        return np.flipud(sil).T                     # [iz, iy]
    if view == 'top':
        return sil.T                                # [ix, iz]
    if view == 'bottom':
        return np.fliplr(sil).T                     # [ix, iz]
    raise ValueError(view)


def _smooth_rows(vals, sigma):
    """Gaussian-smooth a per-row profile in place of its valid (non-NaN)
    rows. Row spacing is preserved by smoothing only the valid run — empty
    rows (outside the object) stay empty."""
    ok = ~np.isnan(vals)
    if sigma <= 0 or ok.sum() < 3:
        return vals
    from scipy.ndimage import gaussian_filter1d
    out = vals.copy()
    out[ok] = gaussian_filter1d(vals[ok], sigma, mode='nearest')
    return out


def lathe_constraint(grid_shape, faces):
    """CROSS_SECTION=round: per height, an elliptical disc whose semi-axes are
    that height's front half-width and side half-depth. Equivalent to carving
    with a continuum of views rotated about the vertical axis — removes the
    diagonal bulges the four perpendicular views can't see. Row centers come
    from each row's own extent, so an off-axis profile (spout, lean) follows
    the silhouette rather than snapping to the grid center.

    With PROFILE_SMOOTH > 0 the per-row centers/radii are gaussian-smoothed
    along the height axis before carving — mask-edge noise otherwise jitters
    each row independently and leaves horizontal rings on the surface."""
    nx, ny, nz = grid_shape
    occ = np.zeros(grid_shape, dtype=bool)
    xs, zs = np.arange(nx), np.arange(nz)
    side = faces['side'] if 'side' in faces else None       # [iz, iy]

    prof = {k: np.full(ny, np.nan) for k in ('cx', 'rx', 'cz', 'rz')}
    for iy in range(ny):
        span_x = np.flatnonzero(faces['front'][:, iy])
        if span_x.size == 0:
            continue
        prof['cx'][iy] = (span_x[0] + span_x[-1]) / 2
        prof['rx'][iy] = max((span_x[-1] - span_x[0]) / 2, 0.5)
        if side is not None:
            span_z = np.flatnonzero(side[:, iy])
            if span_z.size == 0:
                prof['cx'][iy] = prof['rx'][iy] = np.nan
                continue
            prof['cz'][iy] = (span_z[0] + span_z[-1]) / 2
            prof['rz'][iy] = max((span_z[-1] - span_z[0]) / 2, 0.5)
        else:
            # no side view: assume a circular section centered in depth
            prof['cz'][iy] = (nz - 1) / 2
            prof['rz'][iy] = prof['rx'][iy]

    if PROFILE_SMOOTH > 0:
        prof = {k: _smooth_rows(v, PROFILE_SMOOTH) for k, v in prof.items()}
        print(f"    lathe profile smoothed (sigma {PROFILE_SMOOTH:g} rows)")

    for iy in range(ny):
        if np.isnan(prof['cx'][iy]) or np.isnan(prof['cz'][iy]):
            continue
        cx, rx = prof['cx'][iy], prof['rx'][iy]
        cz, rz = prof['cz'][iy], prof['rz'][iy]
        occ[:, iy, :] = (((xs - cx) / rx) ** 2)[:, None] \
                      + (((zs - cz) / rz) ** 2)[None, :] <= 1.0
    return occ


def carve(grid_shape, faces):
    """AND together each available view's silhouette, extruded through the grid."""
    occ = np.ones(grid_shape, dtype=bool)
    if 'front' in faces:
        occ &= faces['front'][:, :, None]           # [ix, iy] -> broadcast over Z
    if 'back' in faces:
        occ &= faces['back'][:, :, None]
    if 'side' in faces:
        occ &= faces['side'].T[None, :, :]          # [iz, iy] -> [1, iy, iz]
    if 'top' in faces:
        occ &= faces['top'][:, None, :]             # [ix, iz] -> broadcast over Y
    if 'bottom' in faces:
        occ &= faces['bottom'][:, None, :]
    if CROSS_SECTION == 'round':
        occ &= lathe_constraint(grid_shape, faces)
        print("    round cross-section (lathe) applied")
    return occ


# ---------------------------------------------------------------- meshing

def voxels_to_mesh(occ, voxel_size_m):
    # zero-padding closes the surface at grid boundaries; when the volume is
    # blurred the pad must cover the blur support, or the smoothed surface
    # gets clipped by the array edge and the mesh comes out open (not
    # watertight)
    pad = max(1, int(np.ceil(3 * VOLUME_SMOOTH)) + 1)
    vol = np.pad(occ.astype(np.float32), pad)
    if VOLUME_SMOOTH > 0:
        from scipy.ndimage import gaussian_filter
        vol = gaussian_filter(vol, VOLUME_SMOOTH)
        print(f"    occupancy volume smoothed (sigma {VOLUME_SMOOTH:g} voxels)")
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, spacing=voxel_size_m)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    # taubin smoothing removes the voxel staircase with far less shrinkage
    # than plain laplacian — residual shrink is corrected in finalize() anyway
    trimesh.smoothing.filter_taubin(mesh, iterations=SMOOTH_ITERATIONS)
    print(f"    raw mesh: {len(mesh.faces):,} faces")
    if TARGET_FACES and len(mesh.faces) > TARGET_FACES:
        # fast-simplification mishandles float64 vertices: it stalls short of
        # face_count and needs several quality-destroying passes (IoU 0.99 ->
        # 0.92 on the snowglobe). float32 converges in one clean pass, and
        # .glb stores float32 anyway. The loop stays as a safety net.
        mesh.vertices = mesh.vertices.astype(np.float32).astype(np.float64)
        while len(mesh.faces) > TARGET_FACES:
            before = len(mesh.faces)
            mesh = mesh.simplify_quadric_decimation(face_count=TARGET_FACES)
            if len(mesh.faces) >= before:
                break
        print(f"    decimated to {len(mesh.faces):,} faces")
    # decimation sheds a few degenerate slivers; mask speckles can also leave
    # floaters — either way, the product is the largest connected component
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda p: len(p.faces))
    print(f"    final: {len(mesh.faces):,} faces   watertight: {mesh.is_watertight}")
    return mesh


def finalize(mesh, target_extents_m):
    """Exact-size rescale (undoes smoothing shrink), then move the origin to
    the bottom-center so the asset sits on the floor at y=0."""
    current = mesh.bounding_box.extents
    mesh.apply_scale(np.asarray(target_extents_m) / current)
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    mesh.fix_normals()
    mesh.visual.face_colors = [190, 190, 195, 255]  # flat gray until M2 (color pass)
    return mesh


def embed_provenance(glb_path, dims_cm, stage2_meta, views_used, grid_shape):
    g = GLTF2().load(glb_path)
    assert g is not None, f"pygltflib could not re-open {glb_path}"
    g.scenes[g.scene or 0].extras = {
        'subject_id':         SUBJECT_ID,
        'route':              'silhouette_hull',
        'generator':          'asset_generation_step/build_silhouette_mesh.py',
        'generated_at':       datetime.now().isoformat(),
        'measurements_cm':    dims_cm,
        'error_estimates_cm': stage2_meta.get('error_estimates_cm'),
        'views_used':         views_used,
        'side_captured_from': SIDE_FROM if 'side' in views_used else None,
        'holes_filled_views': sorted(FILL_HOLES_VIEWS & set(views_used)),
        'cross_section':      CROSS_SECTION,
        'smooth_iterations':  SMOOTH_ITERATIONS,
        'profile_smooth':     PROFILE_SMOOTH,
        'sil_smooth':         SIL_SMOOTH,
        'volume_smooth':      VOLUME_SMOOTH,
        'voxel_grid':         list(grid_shape),
        'stage2_model_versions': stage2_meta.get('model_versions'),
    }
    g.save(glb_path)


# ---------------------------------------------------------------- main

def main():
    print("=" * 60)
    print("STAGE 3 — SILHOUETTE RECONSTRUCTION (Route C)")
    print("=" * 60)
    print(f"Subject: {SUBJECT_ID}   Resolution: {RESOLUTION}\n")

    dims_cm, stage2_meta = load_measurements()
    w, h, d = (dims_cm[k] / 100.0 for k in ('width', 'height', 'depth'))
    print(f"[*] Target size (W x H x D): {dims_cm['width']} x {dims_cm['height']}"
          f" x {dims_cm['depth']} cm")

    # grid proportional to the real dimensions -> cubic voxels
    longest = max(w, h, d)
    nx, ny, nz = (max(8, round(RESOLUTION * v / longest)) for v in (w, h, d))
    voxel = (w / nx, h / ny, d / nz)
    print(f"[*] Voxel grid: {nx} x {ny} x {nz}")

    print("[*] Loading silhouettes...")
    face_res = {'front': (ny, nx), 'back': (ny, nx), 'side': (ny, nz),
                'top': (nz, nx), 'bottom': (nz, nx)}
    faces = {}
    for view in VIEWS:
        sil = load_view_silhouette(view, *face_res[view])
        if sil is not None:
            faces[view] = orient_to_grid(sil, view)
    if 'front' not in faces:
        print(f"[!] No front masks found in {os.path.join(MASKS_DIR, 'front')}")
        print("    Copy the segmentation step's *_product_mask.png files there.")
        sys.exit(1)
    if list(faces) == ['front']:
        print("    [!] front view only — depth profile will be a straight extrusion.")
        print("        Add side (and top) masks to carve the real profile.")

    print("[*] Carving voxel grid...")
    occ = carve((nx, ny, nz), faces)
    fill = occ.mean()
    print(f"    {occ.sum():,} voxels survived ({fill:.0%} of grid)")
    if fill < 0.01:
        print("[!] Almost nothing survived the carve — a view's mask is likely")
        print("    misassigned (e.g. side masks in the front folder). Aborting.")
        sys.exit(1)

    print("[*] Marching cubes + smoothing...")
    mesh = voxels_to_mesh(occ, voxel)
    mesh = finalize(mesh, (w, h, d))

    os.makedirs(WORK_DIR, exist_ok=True)
    out_path = os.path.join(WORK_DIR, f'{SUBJECT_ID}_hull.glb')
    mesh.export(out_path)
    embed_provenance(out_path, dims_cm, stage2_meta, sorted(faces), (nx, ny, nz))

    # reload-and-check: the exported file, not the in-memory mesh, is the truth
    check = trimesh.load(out_path, force='mesh')
    ext_cm = check.bounding_box.extents * 100
    print(f"\n[*] Wrote {out_path}")
    print(f"    triangles: {len(check.faces):,}   watertight: {check.is_watertight}")
    print(f"    bbox (cm): W {ext_cm[0]:.2f}  H {ext_cm[1]:.2f}  D {ext_cm[2]:.2f}"
          f"   (target {dims_cm['width']} x {dims_cm['height']} x {dims_cm['depth']})")
    print("\nDone. Inspect it at https://gltf-viewer.donmccurdy.com — then run")
    print("validate_glb.py (once built) before shipping.")


if __name__ == '__main__':
    main()
