"""
glassiness.py — DETECT reflectivity/transparency from the captures, so the
pipeline decides HOW glass should be portrayed instead of asking the user or
guessing by geometry (material_pass's neck pinch).

Two optical cues, both from data the pipeline already produces:
  * SEE-THROUGH — the monocular depth estimator reads BEHIND a transparent
    surface (it sees the background through the glass). Per pixel:
    (depth_m - a4_surface_depth) / a4_surface_depth, positive over glass.
    (Bottle: A4 said 0.42 m, depth-map said ~0.48 m -> ~13% overshoot.)
  * SPECULAR — glass/gloss throws bright, low-saturation, view-dependent
    highlights. Per pixel: high HSV Value + low Saturation.

frame_glassiness()      -> per-pixel [0,1] map (specular always; + see-through
                           when a depth map is present).
file_view_glass_maps()  -> file <stem>_glass.png beside each view's masks
                           (called from measure_all.py while depth is fresh).
object_score()          -> one scalar per object -> suggested class (Part A).
bake_face_glassiness()  -> per-mesh-face glassiness -> which faces are glass
                           (Part B; added below).
"""

import glob
import json
import os

import cv2
import numpy as np

# --- tunables (calibrated: bottle glass body vs opaque control) ---
SPEC_V = 0.75          # specular highlight: HSV Value above this ...
SPEC_S = 0.40          # ... and Saturation below this
SEETHROUGH_TAU = 0.25  # depth overshoot fraction that saturates see-through to 1
SPEC_W = 0.8           # specular weight when combined (max) with see-through
OBJECT_THRESHOLD = float(os.environ.get("GLASS_OBJECT_THRESHOLD", "0.12"))


def frame_glassiness(image_bgr, mask, depth_m=None, a4_depth=None):
    """Grayscale [0,1] glassiness over the product mask for one frame."""
    h, w = mask.shape[:2]
    sel = mask > 127
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    s, v = hsv[..., 1], hsv[..., 2]
    spec = (np.clip((v - SPEC_V) / (1 - SPEC_V), 0, 1)
            * np.clip((SPEC_S - s) / SPEC_S, 0, 1))
    g = SPEC_W * spec
    if depth_m is not None and a4_depth and a4_depth > 0:
        if depth_m.shape[:2] != (h, w):
            depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
        seethrough = np.clip((depth_m - a4_depth) / (a4_depth * SEETHROUGH_TAU),
                             0, 1)
        g = np.maximum(g, seethrough)
    g = np.clip(g, 0, 1)
    g[~sel] = 0.0
    return g


def _depth_index(depth_results_path):
    """stem -> depth_results entry (depth_map_path, estimated_distance_m)."""
    if not depth_results_path or not os.path.exists(depth_results_path):
        return {}
    idx = {}
    for e in json.load(open(depth_results_path)):
        stem = os.path.splitext(os.path.basename(e["frame"]))[0]
        idx[stem] = e
    return idx


def file_view_glass_maps(view_dir, mask_dir, depth_results_path=None):
    """For each product mask filed in mask_dir, compute + save <stem>_glass.png
    there. Uses depth (see-through) when depth_results has the frame; else
    specular-only. Returns the count."""
    didx = _depth_index(depth_results_path)
    n = 0
    for mp in sorted(glob.glob(os.path.join(mask_dir, "*_product_mask.png"))):
        stem = os.path.basename(mp).replace("_product_mask.png", "")
        frames = (glob.glob(os.path.join(view_dir, stem + ".*"))
                  or glob.glob(os.path.join(view_dir, "**", stem + ".*"),
                               recursive=True))
        if not frames:
            continue
        img = cv2.imread(frames[0])
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        depth_m = a4 = None
        e = didx.get(stem)
        if e and os.path.exists(e.get("depth_map_path", "")):
            depth_m = np.load(e["depth_map_path"])
            a4 = e.get("estimated_distance_m")
        g = frame_glassiness(img, mask, depth_m, a4)
        cv2.imwrite(os.path.join(mask_dir, f"{stem}_glass.png"),
                    (g * 255).astype(np.uint8))
        n += 1
    kind = "depth+specular" if didx else "specular-only"
    print(f"[glassiness] filed {n} glass map(s) -> {mask_dir} ({kind})")
    return n


def object_score(subject, masks_root, views=("front", "side", "back", "top", "bottom")):
    """Mean glassiness over the product masks across all filed glass maps.
    Returns (score, suggested_class in {'transparent','opaque'} or None)."""
    base = os.path.join(masks_root, subject)
    vals = []
    for v in views:
        d = os.path.join(base, v)
        for gp in glob.glob(os.path.join(d, "*_glass.png")):
            stem = os.path.basename(gp).replace("_glass.png", "")
            g = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
            m = cv2.imread(os.path.join(d, stem + "_product_mask.png"),
                           cv2.IMREAD_GRAYSCALE)
            if g is None or m is None:
                continue
            pix = g[m > 127]
            if pix.size:
                vals.append(float((pix / 255.0).mean()))
    if not vals:
        return 0.0, None
    score = float(np.mean(vals))
    return score, ("transparent" if score > OBJECT_THRESHOLD else "opaque")


# --------------------------------------------------------------------------- #
# Part B — project the per-frame glass maps onto the mesh -> per-face glassiness
# --------------------------------------------------------------------------- #
def _view_uv(view, X, Y, Z, side_from):
    """Normalized [0,1] photo (u,v) for a normalized-box point — the same
    convention as color_hull.view_uv (front faces +Z, u along width)."""
    if view == "front":
        return X, 1 - Y
    if view == "back":
        return 1 - X, 1 - Y
    if view == "side":
        return (1 - Z, 1 - Y) if side_from == "right" else (Z, 1 - Y)
    if view == "top":
        return X, Z
    return 1 - X, Z          # bottom


def _front_most(u, v, depth, res, tol):
    """Cheap depth buffer: points at/near the nearest depth for their (u,v) cell
    are visible; ones behind them are occluded. Mirrors texture_hull._front_most."""
    iu = np.clip((u * (res - 1)).astype(np.int32), 0, res - 1)
    iv = np.clip((v * (res - 1)).astype(np.int32), 0, res - 1)
    cell = iv * res + iu
    nearest = np.full(res * res, np.inf, np.float32)
    np.minimum.at(nearest, cell, depth.astype(np.float32))
    return depth <= nearest[cell] + tol


def bake_face_glassiness(mesh, subject, masks_root, blend_power=2.0):
    """Per-face glassiness [0,1] by projecting each face centroid onto the filed
    per-view glass maps (normal-weighted, occlusion-tested), same view geometry
    as the texture bake. Returns None if no glass maps exist for the subject."""
    side_from = os.environ.get("SIDE_FROM", "right")
    base = os.path.join(masks_root, subject)
    lo, hi = mesh.bounds
    C = mesh.triangles_center
    X, Y, Z = ((C - lo) / np.maximum(hi - lo, 1e-9)).T
    nx, ny, nz = mesh.face_normals.T
    has_back = bool(glob.glob(os.path.join(base, "back", "*_glass.png")))

    total = np.zeros(len(mesh.faces))
    wsum = np.zeros(len(mesh.faces))
    used = []
    for view in ("front", "side", "back", "top", "bottom"):
        # pair each glass map with its product-mask BOUNDING BOX, so normalized
        # object coords map into the box the object occupies in the frame (the
        # same convention as color_hull.sample_view) — NOT the whole frame.
        pairs = []
        for gp in sorted(glob.glob(os.path.join(base, view, "*_glass.png"))):
            stem = os.path.basename(gp).replace("_glass.png", "")
            gm = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
            mk = cv2.imread(os.path.join(base, view, stem + "_product_mask.png"),
                            cv2.IMREAD_GRAYSCALE)
            if gm is None or mk is None:
                continue
            pts = cv2.findNonZero((mk > 127).astype(np.uint8))
            if pts is None:
                continue
            bx, by, bw, bh = cv2.boundingRect(pts)
            pairs.append((gm, bx, by, bw, bh))
        if not pairs:
            continue
        used.append(view)
        u, v = _view_uv(view, X, Y, Z, side_from)
        if view == "front":
            comp, two_sided = nz, not has_back
            w = (np.maximum(nz, 0) if has_back else np.abs(nz)) ** blend_power
        elif view == "back":
            comp, two_sided, w = -nz, False, np.maximum(-nz, 0) ** blend_power
        elif view == "side":
            comp, two_sided, w = nx, True, np.abs(nx) ** blend_power
        elif view == "top":
            comp, two_sided, w = ny, False, np.maximum(ny, 0) ** blend_power
        else:
            comp, two_sided, w = -ny, False, np.maximum(-ny, 0) ** blend_power
        # occlusion: which end of the view axis the camera sits at
        axis = {"front": Z, "back": Z, "top": Y, "bottom": Y, "side": X}[view]
        if two_sided:
            vis = np.where(comp >= 0, _front_most(u, v, 1 - axis, 512, 0.01),
                           _front_most(u, v, axis, 512, 0.01))
        else:
            near_high = view in ("front", "top") or (view == "side" and side_from == "right")
            vis = _front_most(u, v, (1 - axis) if near_high else axis, 512, 0.01)
        w = w * vis
        vals = np.zeros(len(mesh.faces))
        for gm, bx, by, bw, bh in pairs:
            iu = np.clip((bx + u * (bw - 1)).astype(np.int32), 0, gm.shape[1] - 1)
            iv = np.clip((by + v * (bh - 1)).astype(np.int32), 0, gm.shape[0] - 1)
            vals += gm[iv, iu] / 255.0
        vals /= len(pairs)
        total += vals * w
        wsum += w
    if not used:
        return None
    glass = np.where(wsum > 1e-6, total / np.maximum(wsum, 1e-9), 0.0)
    print(f"[glassiness] baked face glassiness from views {used}: "
          f"mean {glass.mean():.2f}, faces>0.5 = {(glass > 0.5).mean():.0%}")
    return glass
