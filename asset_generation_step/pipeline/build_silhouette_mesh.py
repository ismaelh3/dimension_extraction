"""
Stage 3, Route C — silhouette reconstruction (visual hull).

Builds a real-world-scaled 3D mesh (.glb) for a product by carving a voxel
grid — sized exactly to the Stage 2 measurements — with the product's
segmentation masks. Supports any subset of four views:

    front  (required)   side | back | top | bottom   (optional, each tightens the hull)

Note: top and bottom produce the SAME carving constraint (an object's silhouette
along the vertical axis is identical from above and below, mirrored) — one of the
two is enough for geometry. When both are provided they are UNIONED into a single
footprint before carving: camera tilt skews each one slightly, and intersecting
two skewed footprints carves off whatever they disagree about (it cost the
sneaker 14 mm of toe). Union keeps anything either view saw.

Masks are consumed from  masks/<SUBJECT>/<view>/*.png  (the 0/255 PNGs the
segmentation step writes). Copy each capture set's masks there after running
the segmentation step on it — extra views (back/top) only need segmentation,
not depth/measurement, since scale comes entirely from the measurements JSON.

Usage:  SUBJECT=snowglobe CROSS_SECTION=round make build-asset
        SUBJECT=snowglobe RESOLUTION=768 SIDE_FROM=left FILL_HOLES=all \
            venv/bin/python asset_generation_step/pipeline/build_silhouette_mesh.py

Output: work/<SUBJECT>_hull.glb  — metres, +Y up, front facing +Z,
        origin at bottom-center, provenance embedded in glTF extras.
"""

import glob
import json
import os
import pickle
import sys
from datetime import datetime

import cv2
import numpy as np
import trimesh
from pygltflib import GLTF2
from skimage import measure

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # asset_generation_step/
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
# 'footprint' is the middle ground for products with a rounded but
# non-elliptical outline (shoes): it sweeps the top-view footprint along the
# height axis, rescaled per row to the front/side spans, so the ends taper
# with the height profile instead of being extruded as boxy walls — without
# the lathe forcing an ellipse onto them. Requires top and/or bottom masks.
CROSS_SECTION = os.environ.get('CROSS_SECTION', 'silhouette')
# Silhouettes provably cannot carve a concavity — the rim of an opening (a
# shoe's mouth, a cap's bowl) hides the interior from every outline, so the
# hull shrink-wraps a flat lid across it. DEPTH_CARVE re-opens it using
# monocular depth from the one view that looks INTO the cavity: per frame,
# disparity is converted to real height above the table via two anchors (the
# table itself, and the shell's own closest points, both solved exactly, see
# _height_m), a sink-fill test finds pixels enclosed deep enough to be a true
# pocket rather than an honest slope, and voxels above that floor get
# cleared. Needs per-frame disparity .npy files from tools/probe_opening_depth
# .py in work/depth_probe/<SUBJECT>/. Opt-in per subject ('top', 'bottom', or
# 'top,bottom'): matte interiors (fabric, foam) read well; glass/gloss breaks
# monocular depth — leave those subjects off.
DEPTH_CARVE = os.environ.get('DEPTH_CARVE', 'none')
# Monocular relative depth is an ORDINAL signal, not a metric one: it gets
# the SHAPE of the cavity right (where it's deeper vs shallower) but
# compresses the absolute magnitude in a dim, low-texture interior — a real
# limitation of the model, not a calibration bug (verified: even with both
# anchors solved exactly, the sneaker's mouth measured 5.2cm max against a
# 7.9cm tape measurement). DEPTH_CARVE_GAIN scales the carved depth (about
# each column's own lid height, so shape is preserved) to match a real
# measurement: gain = true_depth_cm / reported_depth_cm from the previous
# run's "cavity ... depth ... max" line. 1.0 = no correction.
DEPTH_CARVE_GAIN = float(os.environ.get('DEPTH_CARVE_GAIN', '1.0'))
# Top/bottom photos are easy to shoot 90° off convention (product's long axis
# along image rows instead of columns) — load_view_silhouette detects this by
# aspect ratio and auto-rotates the mask. Aspect alone can't distinguish the
# two possible 90° rotations, so this says which side of the as-shot image
# faces the product's FRONT (the face the front view shows): 'left' or
# 'right', either one value for both views or per-view as
# 'top:left,bottom:right'. Only consulted when a rotation is actually
# applied. The default matches the nike-shoe captures (verified against the
# collar position in the top view and the forefoot bulge in the sole view):
# both sets were shot toe-at-image-top, and orient_to_grid's roll-mirror for
# the bottom view means the two then need OPPOSITE rotations.
TOPDOWN_FRONT = os.environ.get('TOPDOWN_FRONT', 'top:left,bottom:right')

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
if CROSS_SECTION not in ('silhouette', 'round', 'footprint'):
    print(f"[!] CROSS_SECTION must be 'silhouette', 'round', or 'footprint', "
          f"not '{CROSS_SECTION}'.")
    sys.exit(1)
_dc = DEPTH_CARVE.strip().lower()
DEPTH_CARVE_VIEWS = set() if _dc in ('', 'none') else {v.strip() for v in _dc.split(',')}
if DEPTH_CARVE_VIEWS - {'top', 'bottom'}:
    print(f"[!] DEPTH_CARVE only supports top/bottom (the views that look into "
          f"an opening) — got '{DEPTH_CARVE}'.")
    sys.exit(1)
def _parse_topdown_front(spec):
    spec = spec.strip().lower()
    if spec in ('left', 'right'):
        return {'top': spec, 'bottom': spec}
    sides = {}
    for part in spec.split(','):
        view, _, side = part.strip().partition(':')
        if view not in ('top', 'bottom') or side not in ('left', 'right'):
            print(f"[!] TOPDOWN_FRONT must be 'left', 'right', or per-view like "
                  f"'top:left,bottom:right' — got '{spec}'.")
            sys.exit(1)
        sides[view] = side
    return {'top': sides.get('top', 'left'), 'bottom': sides.get('bottom', 'left')}

TOPDOWN_FRONT_SIDES = _parse_topdown_front(TOPDOWN_FRONT)


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


def _topdown_rotation_k(shape_hw, view, rows, cols, name):
    """Rotation (in np.rot90 quarter-turns) that fixes a top/bottom capture
    shot 90° off convention — split out so the depth loader can apply the
    SAME decision to a frame's disparity map as to its mask."""
    if min(rows, cols) / max(rows, cols) > 0.8:
        return 0                      # near-square face: aspect is ambiguous
    h, w = shape_hw
    err_asis    = abs(np.log((w / h) / (cols / rows)))
    err_rotated = abs(np.log((h / w) / (cols / rows)))
    if err_rotated >= err_asis:
        return 0
    side = TOPDOWN_FRONT_SIDES[view]
    k = 1 if side == 'left' else 3
    print(f"    [!] {view} mask {name}: aspect {w / h:.2f} matches the "
          f"transposed face (expected {cols / rows:.2f}) — rotated 90° "
          f"{'ccw' if k == 1 else 'cw'} (TOPDOWN_FRONT {view}:{side}). "
          f"Future captures: long axis along image columns, front at bottom.")
    return k


def fix_topdown_rotation(cropped, view, rows, cols, name):
    """Catch top/bottom masks shot 90° off convention. The resize in
    load_view_silhouette stretches whatever it gets onto the face, so a
    transposed mask doesn't fail — it silently garbles the footprint (this
    cost the sneaker its toe and heel). If the tight bbox's aspect matches
    the transposed face better than the expected one, rotate the mask 90°:
    ccw puts the image's left side at the bottom (= product front, per the
    top-view convention), cw puts the right side there — TOPDOWN_FRONT picks
    which, since aspect alone can't."""
    k = _topdown_rotation_k(cropped.shape, view, rows, cols, name)
    return np.rot90(cropped, k) if k else cropped


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
        if view in ('top', 'bottom'):
            cropped = fix_topdown_rotation(cropped, view, rows, cols,
                                           os.path.basename(p))
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


def _height_profile(grid_shape, faces):
    """Per-height span centers and half-extents: cx/rx from the front view's
    x-span, cz/rz from the side view's z-span (NaN where there's no side
    view, or the row is outside the object). Row centers come from each
    row's own extent, so an off-axis profile (spout, lean) follows the
    silhouette rather than snapping to the grid center. Shared by the round
    and footprint cross-section modes.

    With PROFILE_SMOOTH > 0 the per-row centers/extents are gaussian-smoothed
    along the height axis — mask-edge noise otherwise jitters each row
    independently and leaves horizontal rings on the surface."""
    nx, ny, nz = grid_shape
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
    if PROFILE_SMOOTH > 0:
        prof = {k: _smooth_rows(v, PROFILE_SMOOTH) for k, v in prof.items()}
        print(f"    height profile smoothed (sigma {PROFILE_SMOOTH:g} rows)")
    return prof


def lathe_constraint(grid_shape, faces):
    """CROSS_SECTION=round: per height, an elliptical disc whose semi-axes are
    that height's front half-width and side half-depth. Equivalent to carving
    with a continuum of views rotated about the vertical axis — removes the
    diagonal bulges the four perpendicular views can't see."""
    nx, ny, nz = grid_shape
    occ = np.zeros(grid_shape, dtype=bool)
    xs, zs = np.arange(nx), np.arange(nz)
    prof = _height_profile(grid_shape, faces)
    for iy in range(ny):
        cx, rx, cz, rz = (prof[k][iy] for k in ('cx', 'rx', 'cz', 'rz'))
        if np.isnan(cx):
            continue
        if np.isnan(cz):
            # no side view: assume a circular section centered in depth
            cz, rz = (nz - 1) / 2, rx
        occ[:, iy, :] = (((xs - cx) / rx) ** 2)[:, None] \
                      + (((zs - cz) / rz) ** 2)[None, :] <= 1.0
    return occ


def footprint_sweep(grid_shape, faces, footprint):
    """CROSS_SECTION=footprint: sweep the top-view footprint along the height
    axis — at each height, the footprint is rescaled and re-centered to fit
    that row's front x-span and side z-span, and everything outside it is
    carved.

    The middle ground between the other two modes: 'silhouette' extrudes the
    footprint at full size through every height, so any tilt-skew
    disagreement between it and the front view carves flat vertical faces at
    the ends; 'round' fixes that by replacing the section with an ellipse,
    but shaves the ends off anything that isn't lathe-symmetric (a shoe toe).
    Sweeping keeps the footprint's true outline while letting it taper with
    the height profile. The sweep is pinned to the front/side spans per row,
    which also makes the raw footprint AND redundant — carve() skips it in
    this mode, so footprint-vs-front disagreements can't cut anything."""
    nx, ny, nz = grid_shape
    fx = np.flatnonzero(footprint.any(axis=1))
    fz = np.flatnonzero(footprint.any(axis=0))
    fcx, frx = (fx[0] + fx[-1]) / 2, max((fx[-1] - fx[0]) / 2, 0.5)
    fcz, frz = (fz[0] + fz[-1]) / 2, max((fz[-1] - fz[0]) / 2, 0.5)
    occ = np.zeros(grid_shape, dtype=bool)
    xs, zs = np.arange(nx), np.arange(nz)
    prof = _height_profile(grid_shape, faces)
    for iy in range(ny):
        cx, rx, cz, rz = (prof[k][iy] for k in ('cx', 'rx', 'cz', 'rz'))
        if np.isnan(cx):
            continue
        if np.isnan(cz):
            # no side view: keep the footprint's own depth-to-width aspect
            cz, rz = (nz - 1) / 2, rx * frz / frx
        # map this row's box onto the footprint's box and sample it
        u = np.round((xs - cx) / rx * frx + fcx).astype(int)
        v = np.round((zs - cz) / rz * frz + fcz).astype(int)
        ok_u = (u >= 0) & (u < nx)
        ok_v = (v >= 0) & (v < nz)
        sect = np.zeros((nx, nz), dtype=bool)
        sect[np.ix_(ok_u, ok_v)] = footprint[np.ix_(u[ok_u], v[ok_v])]
        occ[:, iy, :] = sect
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
    # top and bottom describe the SAME footprint, but each is skewed a little
    # by camera tilt (and by genuinely different outlines: collar/laces from
    # above vs sole from below). AND-ing them as independent constraints lets
    # any disagreement carve real geometry — on the sneaker the two footprints
    # shared zero depth-rows at the toe/heel columns and chopped 14/11 mm off
    # the ends as flat vertical faces. Union them into ONE footprint first, so
    # a region survives if either view saw it; redundancy then adds coverage
    # instead of subtracting it.
    footprint = None
    for v in ('top', 'bottom'):
        if v in faces:
            footprint = faces[v] if footprint is None else (footprint | faces[v])
    if 'top' in faces and 'bottom' in faces:
        # low agreement means one set is oriented differently from the other
        # (e.g. TOPDOWN_FRONT wrong for one of them) or badly tilt-skewed
        iou = (faces['top'] & faces['bottom']).sum() / footprint.sum()
        print(f"    top/bottom footprint agreement (IoU): {iou:.2f}")
    if CROSS_SECTION == 'footprint':
        if footprint is None:
            print("[!] CROSS_SECTION=footprint needs top and/or bottom masks —")
            print("    none found. Use 'silhouette' or 'round' instead.")
            sys.exit(1)
        occ &= footprint_sweep(grid_shape, faces, footprint)
        print("    swept-footprint cross-section applied")
    elif footprint is not None:
        occ &= footprint[:, None, :]                # [ix, iz] -> broadcast over Y
    if CROSS_SECTION == 'round':
        occ &= lathe_constraint(grid_shape, faces)
        print("    round cross-section (lathe) applied")
    return occ


# ---------------------------------------------------------------- depth carve

CALIB_FILE = os.path.join(BASE_DIR, '..', 'camera_calibration_step',
                          'output', 'calibration_data.pkl')


def _load_calibration():
    """(fx, fy, calib_size_wh) from Step 1's pickle, or None if missing."""
    if not os.path.exists(CALIB_FILE):
        return None
    with open(CALIB_FILE, 'rb') as f:
        d = pickle.load(f)
    cm = np.asarray(d['camera_matrix'])
    return float(cm[0, 0]), float(cm[1, 1]), d.get('image_size_wh')


def load_view_depthfields(view, rows, cols, dims_cm, calib):
    """Per-frame (disparity, mask, added, d_table, H_cam) tuples on the view's face
    grid, put through the SAME transform chain as the silhouettes (fill ->
    crop -> rotation -> resize -> orient) so pixel (ix, iz) means the same
    place in both.

    Frames stay separate on purpose: monocular disparity has an unknown
    scale AND shift per image, so frames only become comparable after each
    one is anchored to real units (depth_carve does that) — averaging raw
    disparities across frames would mix incompatible scales. Masks are
    always hole-filled here: SAM speckles on cavity interiors (dark insoles)
    would drop exactly the pixels the carve exists to use. The filled-in
    pixels are tracked separately ('added') because that same fill also seals
    genuine see-through gaps, where the depth behind the gap is a lie."""
    fx, fy, calib_wh = calib if calib else (None, None, None)
    fields = []
    for mp in sorted(glob.glob(os.path.join(MASKS_DIR, view, '*_product_mask.png'))):
        name = os.path.basename(mp).replace('_product_mask.png', '')
        dp = os.path.join(WORK_DIR, 'depth_probe', SUBJECT_ID, f'{name}_disparity.npy')
        if not os.path.exists(dp):
            print(f"    [!] {view} {name}: no disparity map — run "
                  f"tools/probe_opening_depth.py (SUBJECT={SUBJECT_ID} "
                  f"VIEW={view}) first; frame skipped")
            continue
        img = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        raw  = img > 127
        mask = fill_holes(raw)
        # Which pixels fill_holes INVENTED. They are not observed product, so
        # their disparity is whatever happened to be behind the gap — see
        # depth_carve, which drops the ones that turn out to be see-through.
        added = mask & ~raw
        pts = cv2.findNonZero(mask.astype(np.uint8))
        if pts is None or len(pts) < MIN_MASK_AREA_PX:
            continue
        x, y, w, h = cv2.boundingRect(pts)
        disp = np.load(dp).astype(np.float32)
        if disp.shape != mask.shape:
            disp = cv2.resize(disp, (mask.shape[1], mask.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        m_c, d_c = mask[y:y + h, x:x + w], disp[y:y + h, x:x + w]
        a_c = added[y:y + h, x:x + w]

        # Table anchor: median disparity of a ring around the product, well
        # clear of its shadowed edge — the height-0 reference the camera
        # distance below is measured against.
        ring_in  = cv2.dilate(mask.astype(np.uint8),
                              np.ones((2 * (max(w, h) // 40) + 1,) * 2,
                                      np.uint8)).astype(bool)
        ring_out = cv2.dilate(ring_in.astype(np.uint8),
                              np.ones((2 * (max(w, h) // 8) + 1,) * 2,
                                      np.uint8)).astype(bool)
        ring = ring_out & ~ring_in
        d_table = float(np.median(disp[ring])) if ring.any() else None

        # Camera height above the table, from the ONE ruler already in every
        # frame: the product itself, whose real width/depth Stage 2 already
        # measured. pixel_size = fx * real_size / distance (pinhole), solved
        # both ways (long axis vs short axis) and averaged. This replaces
        # trying to discover the disparity->height curve's shape from the
        # image alone, which was never well-posed — the shell's own height
        # range is a few cm, and the cavity floor is pure extrapolation
        # below it, degenerate with only the table as a second point (v1
        # here compressed the true 7.9cm mouth to 4-6cm). With H_cam known,
        # height = H_cam * (1 - d_table / disparity) is Step 5's own
        # k = distance * disparity relation, just anchored on the product's
        # measured size instead of the A4 sheet.
        H_cam = None
        if fx and d_table:
            long_m, short_m = (max(dims_cm['width'], dims_cm['depth']) / 100,
                               min(dims_cm['width'], dims_cm['depth']) / 100)
            axis_m = (long_m, short_m) if w >= h else (short_m, long_m)
            H_cam = 0.5 * (fx * axis_m[0] / w + fy * axis_m[1] / h)

        k = _topdown_rotation_k(m_c.shape, view, rows, cols, name)
        if k:
            m_c, d_c = np.rot90(m_c, k), np.rot90(d_c, k)
            a_c = np.rot90(a_c, k)
        m_f = cv2.resize(m_c.astype(np.float32), (cols, rows),
                         interpolation=cv2.INTER_AREA) >= 0.5
        d_f = cv2.resize(d_c, (cols, rows), interpolation=cv2.INTER_AREA)
        # any coverage counts as 'invented' — a half-covered pixel is still
        # part-background, so its depth is still not honest product
        a_f = cv2.resize(a_c.astype(np.float32), (cols, rows),
                         interpolation=cv2.INTER_AREA) > 0
        fields.append((orient_to_grid(d_f, view), orient_to_grid(m_f, view),
                       orient_to_grid(a_f, view), d_table, H_cam))
    return fields


def _height_m(disp, d_table, H_cam, valid, surf_y, voxel_y_m):
    """height above the table (metres), from the model's own affine-invariant
    disparity relation  disp = k/distance + shift  (Depth Anything's stated
    form — see depth_estimation.py's docstring), i.e.
        distance = k / (disp - shift)   =>   height = H_cam - k/(disp - shift)

    k and shift are solved EXACTLY from two real anchors, not fitted:
      1. the table (height 0, disparity d_table — already known)
      2. the shell's closest points (disparity's own 95th percentile within
         the mask — by definition nearer the camera than anything else
         visible, so guaranteed to be raised material — laces/tongue —
         never the recessed interior), whose height is read off the HULL,
         which is trustworthy there (silhouette carving is only wrong
         inside cavities).
    Two anchors exactly fix the two unknowns in a 2-parameter family — no
    search, no ambiguity. Assuming shift=0 (v1 here) instead systematically
    undershot the cavity by 2-4x: with only the table point actually
    constraining the curve, shift was implicitly forced to 0 regardless of
    the model's real (nonzero) affine offset."""
    d_ok = disp[valid]
    d_rim = float(np.percentile(d_ok, 95))
    rim_px = valid & (disp >= d_rim)
    h_rim = float(np.median(surf_y[rim_px])) * voxel_y_m   # metres, above table
    denom = h_rim if abs(h_rim) > 1e-6 else 1e-6
    shift = (H_cam * (d_table - d_rim) + h_rim * d_rim) / denom
    k = H_cam * (d_table - shift)
    return H_cam - k / np.maximum(disp - shift, 1e-3)


def depth_carve(occ, view, fields, voxel_y_m):
    """Carve the concavity a top/bottom view saw into the hull (Report 21).

    Grid work happens 'as seen from above': for the bottom view the volume
    is flipped in Y first so one code path serves both. Per frame the
    disparity is fitted to the hull surface (_fit_floor); the per-frame
    floors are median-combined; pixels whose floor sits well below the hull
    surface are the cavity (small speckle components dropped); the floor is
    clamped so at least a few voxels of wall/sole survive, smoothed so
    per-pixel depth noise doesn't texture the cavity walls, and everything
    above it is cleared."""
    work = occ[:, ::-1, :] if view == 'bottom' else occ
    nx, ny, nz = work.shape
    ys = np.arange(ny)[None, :, None]
    surf_y = np.max(np.where(work, ys, -1), axis=1)          # hull lid height
    base_y = np.min(np.where(work, ys, ny), axis=1)          # opposite face
    inside = surf_y >= 0

    # mask edges blend product depth into background depth — same lesson the
    # probe tool learned; erode each frame's valid region before fitting.
    # The bleed is wide: the depth model works at ~518px internally, so its
    # edges smear across tens of face pixels, not a few
    ek = max(3, RESOLUTION // 64) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))

    floors, n_leak = [], 0
    for disp, m, added, d_table, H_cam in fields:
        if not d_table or not H_cam or H_cam <= 0:
            print(f"    [!] {view}: frame missing a table/camera anchor — skipped")
            continue
        valid = cv2.erode((m & inside).astype(np.uint8), kernel).astype(bool)
        if valid.sum() < 256:
            print(f"    [!] {view}: too little valid area after erosion — frame skipped")
            continue
        floor = _height_m(disp, d_table, H_cam, valid, surf_y, voxel_y_m) / voxel_y_m

        # fill_holes patches SAM speckle over cavity interiors (necessary —
        # dark linings drop out), but it also seals genuine SEE-THROUGH gaps.
        # A snapback cap's strap opening became 'product' whose depth is the
        # TABLE showing through it, so the fit drove the floor to table level
        # there and the carve punched a tunnel out the back of the crown
        # (the mesh went from genus 0 to genus 1). Observed product always
        # sits raised above the table, so an INVENTED pixel reading at table
        # height was never product — drop it rather than carve to it. Only
        # invented pixels are eligible, so honest interior surface that
        # happens to sit low (a shoe's insole) is never rejected.
        leak = added & (floor <= max(2.0, ny / 64))
        if leak.any():
            valid &= ~leak
            n_leak += int(leak.sum())
        # the floor field must live on the ERODED mask, not the full one:
        # at the silhouette edge the depth map blends product into table,
        # and that ring of near-zero floor heights cuts a drain channel
        # through the thin collar rim — the sink-fill test then thinks the
        # mouth can't hold water (spill level 1.8 cm on the sneaker) and
        # refuses to call it a cavity
        floors.append(np.where(valid, floor, np.nan))
    if not floors:
        print(f"    [!] {view}: no frame survived anchoring — depth carve skipped")
        return occ
    if n_leak:
        print(f"    {view}: {n_leak:,} see-through px rejected "
              f"(hole-filled gaps reading at table height)")

    with np.errstate(invalid='ignore'):
        floor = np.nanmedian(np.stack(floors), axis=0)
    known = inside & ~np.isnan(floor)

    # A cavity is a RIM-ENCLOSED depression — a pit the depth surface could
    # hold water in — not just "deeper than the fit expected". Sink-fill the
    # height field from its borders (morphological reconstruction by
    # erosion, the DEM fill-sinks operation): pixels that hold water are
    # cavity; anything that drains off the silhouette edge — the toe box's
    # honest downhill slope, any smooth perspective/affine ramp the linear
    # fit left behind — is provably not. Outside the footprint the field is
    # set very low so the border drains.
    from skimage.morphology import reconstruction
    low = float(np.nanmin(np.where(known, floor, np.nan))) - float(ny)
    field_h = np.where(known, floor, low).astype(np.float32)
    # narrow false drain channels — shadowed slots between laces read as
    # deep as the cavity and chain it to the outside, so the raw field's
    # mouth "leaks" and never registers as a pit (measured spill level:
    # 1.8 cm). Grayscale-close with a kernel wider than a lace gap but far
    # narrower than the opening: slots get bridged, the mouth cannot be.
    # Enclosure is tested on the closed field; carving still uses the true
    # floor, and at the bridged slots the closed field sits at lid level so
    # the lace bridge itself is never carved.
    kc = max(9, RESOLUTION // 24) | 1
    closed = cv2.morphologyEx(
        field_h, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    seed = np.full_like(closed, closed.max())
    seed[0, :], seed[-1, :], seed[:, 0], seed[:, -1] = \
        closed[0, :], closed[-1, :], closed[:, 0], closed[:, -1]
    filled = reconstruction(seed, closed, method='erosion')
    pit = filled - closed

    # margin keeps honest fit scatter from nibbling the surface; requiring
    # the lid to also sit well above the floor skips no-op carves
    margin = max(3.0, ny / 48)
    cavity = known & (pit > margin) & (surf_y - floor > margin)
    # round off the cavity boundary: a jagged contour forces a steep,
    # high-curvature rim wall into the mesh (fine at full res, but exactly
    # the kind of thin feature quadric decimation later collapses into a
    # gap — texture_hull.py's 20k-face pass on an earlier build did this on
    # one side of this same mouth)
    ok = max(3, RESOLUTION // 128) | 1
    cavity = cv2.morphologyEx(cavity.astype(np.uint8), cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
                              ).astype(bool)
    n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(
        cavity.astype(np.uint8), connectivity=8)
    min_area = max(16, int(inside.sum()) // 200)
    cavity &= np.isin(lbl, [i for i in range(1, n_lbl)
                            if stats[i, cv2.CC_STAT_AREA] >= min_area])
    if not cavity.any():
        print(f"    {view}: no cavity found beyond the {margin:.0f}-voxel "
              f"margin — nothing carved")
        return occ

    if DEPTH_CARVE_GAIN != 1.0:
        # deepen about each column's own lid (preserves the vision-derived
        # SHAPE — where the pocket is deeper/shallower — and only rescales
        # the magnitude to match a real measurement)
        floor = np.where(cavity, surf_y - (surf_y - floor) * DEPTH_CARVE_GAIN, floor)

    # never carve through the far side: keep a few voxels of sole/shell.
    # The reserve must also survive VOLUME_SMOOTH, which blurs the occupancy
    # AFTER this clamp: a wall thinner than a few sigma washes below the 0.5
    # iso-level and marching cubes never emits it, so the shell silently
    # perforates (the cap's crown tore into 3 extra holes at gain 1.53 —
    # genus 1 -> 4 — while this guard still "reserved" its 3 voxels).
    shell = max(2, ny // 64, int(np.ceil(2.5 * VOLUME_SMOOTH)))
    # ...but `shell` is a VERTICAL reserve, while wall thickness is measured
    # PERPENDICULAR to the surface. Where the far surface is steep, a vertical
    # reserve of N voxels is only N*cos(slope) of actual wall, and that goes
    # to zero as the surface turns vertical — so this guard protected the flat
    # apex and did nothing on the flanks. That is exactly where the cap tore:
    # one 2.8 x 1.7 cm hole at 2-4 cm height on the crown's side, never near
    # the top. Scale by 1/cos(slope) = sqrt(1 + |grad base_y|^2) so the
    # perpendicular thickness is what stays constant. base_y is smoothed
    # first (its raw per-column gradient is voxel-staircase noise) and the
    # factor is capped, since the gradient blows up at the silhouette edge.
    # The grid is isotropic here, so gradient in voxel units needs no rescale.
    base_s = cv2.GaussianBlur(np.where(inside, base_y, 0).astype(np.float32),
                              (0, 0), 2.0)
    g0, g1 = np.gradient(base_s)
    shell_v = shell * np.minimum(np.sqrt(1.0 + g0 ** 2 + g1 ** 2), 8.0)
    floor = np.maximum(floor, base_y + shell_v)
    # smooth the floor field so pixel noise doesn't ripple the cavity walls;
    # fill non-cavity pixels with the lid height first so the blur doesn't
    # drag rim heights down
    field = np.where(cavity, floor, surf_y).astype(np.float32)
    field = cv2.GaussianBlur(field, (0, 0), 1.5)
    # Re-assert the shell reserve AFTER the blur. Clamping only before it is
    # not enough: the blur mixes each column's floor with its neighbours',
    # and where base_y falls away steeply (the crown's flanks) a neighbour's
    # legitimately-lower floor drags this column below its OWN reserve and
    # the carve punches through. Clamping pre-blur alone shattered the cap's
    # crown into ~20 small handles instead of one clean tear.
    field = np.maximum(field, (base_y + shell_v).astype(np.float32))

    clear = cavity[:, None, :] & (ys > field[:, None, :])
    carved = int((work & clear).sum())
    work &= ~clear
    depth_vox = (surf_y - field)[cavity]
    print(f"    {view}: cavity {cavity.sum():,} columns, depth "
          f"{np.median(depth_vox) * voxel_y_m * 100:.1f} cm median / "
          f"{depth_vox.max() * voxel_y_m * 100:.1f} cm max "
          f"({carved:,} voxels cleared)")
    return work[:, ::-1, :] if view == 'bottom' else work


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
        'generator':          'asset_generation_step/pipeline/build_silhouette_mesh.py',
        'generated_at':       datetime.now().isoformat(),
        'measurements_cm':    dims_cm,
        'error_estimates_cm': stage2_meta.get('error_estimates_cm'),
        'views_used':         views_used,
        'side_captured_from': SIDE_FROM if 'side' in views_used else None,
        'holes_filled_views': sorted(FILL_HOLES_VIEWS & set(views_used)),
        'cross_section':      CROSS_SECTION,
        'depth_carve':        sorted(DEPTH_CARVE_VIEWS) or None,
        'depth_carve_gain':   DEPTH_CARVE_GAIN if DEPTH_CARVE_VIEWS else None,
        'topdown_front':      TOPDOWN_FRONT,
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

    if DEPTH_CARVE_VIEWS:
        print("[*] Depth-carving openings...")
        calib = _load_calibration()
        if calib is None:
            print(f"    [!] No calibration at {CALIB_FILE} — depth carve "
                  f"needs it to anchor real units. Skipping.")
        for view in sorted(DEPTH_CARVE_VIEWS) if calib else []:
            fields = load_view_depthfields(view, *face_res[view], dims_cm, calib)
            if not fields:
                print(f"    [!] {view}: no usable depth frames — skipped")
                continue
            occ = depth_carve(occ, view, fields, voxel[1])

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
