"""
Stage 3 (edge-case) — GENERATE an interior that extraction can't recover.

For a sealed refractive object (snowglobe), the contents are seen only through
the glass, so shape-from-silhouette merges them into a blob. The fix is
generative: reconstruct the interior from ONE clean photo with a learned prior
(TripoSR), then hand the geometry to texture_hull.py for photo-accurate colour.

This stage folds the proven recipe into one SUBJECT-driven step:
  1. cutout      — a clean foreground cutout from the chosen view's frame×mask,
                   composited on neutral gray (TripoSR expects a padded square).
  2. reconstruct — TripoSR single-image->3D (venv, MPS/CPU); auto-vendored+patched
                   via tools/setup_triposr.py.
  3. orient      — permute axes into the canonical carve frame (faces->+Z, up->+Y)
                   so texture_hull's front/side/back projection lands correctly.
  4. smooth      — optional volume-preserving Taubin to kill single-view surface
                   noise (reads as 'dirty' bodies once lit).
Output: work/<SUBJECT>_interior_hull.glb  (drop-in for texture_hull.py)
        + work/<SUBJECT>_interior_debug.png (front/side/back — VERIFY orientation)

Usage: SUBJECT=snowglobe venv/bin/python asset_generation_step/pipeline/generate_interior.py
Knobs: INTERIOR_VIEW (front) INTERIOR_FRAME (auto=middle of the view's masks)
       FRAMES_DIR (../instance_segmentation_step/frames) MC_RESOLUTION (256)
       ORIENT (yzx; axis permutation, 'none' to skip) SMOOTH_ITERS (16)
       FG_RATIO (0.85) CUTOUT_SIZE (768) DEVICE (mps)
"""

import glob
import os
import subprocess
import sys

import numpy as np
import trimesh
from PIL import Image, ImageOps, ImageDraw, ImageFilter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # asset_generation_step/
REPO_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "tools"))
import setup_triposr                              # noqa: E402
from preview_render import raster_preview         # noqa: E402

SUBJECT     = os.environ.get("SUBJECT", "product_000")
VIEW        = os.environ.get("INTERIOR_VIEW", "front")
FRAMES_DIR  = os.environ.get("FRAMES_DIR",
                             os.path.join(REPO_ROOT, "instance_segmentation_step", "frames"))
MC_RES      = int(os.environ.get("MC_RESOLUTION", "256"))
ORIENT      = os.environ.get("ORIENT", "yzx")
SMOOTH_ITERS = int(os.environ.get("SMOOTH_ITERS", "16"))
FG_RATIO    = float(os.environ.get("FG_RATIO", "0.85"))
CUTOUT_SIZE = int(os.environ.get("CUTOUT_SIZE", "768"))
GRAY        = 128
DEVICE      = os.environ.get("DEVICE", "mps")

WORK        = os.path.join(BASE_DIR, "work")
MASKS_DIR   = os.path.join(BASE_DIR, "masks", SUBJECT, VIEW)
CUTOUT_PNG  = os.path.join(WORK, f"{SUBJECT}_triposr_input.png")
TRIPOSR_OUT = os.path.join(WORK, f"{SUBJECT}_triposr_out")
# same slot the carve (build_silhouette_mesh.py) writes, so texture_hull.py picks
# it up: work/<SUBJECT>_hull.glb (the "interior" is encoded in the SUBJECT name,
# e.g. SUBJECT=snowglobe_interior, not doubled into the filename).
OUT_GLB     = os.path.join(WORK, f"{SUBJECT}_hull.glb")
DEBUG_PNG   = os.path.join(WORK, f"{SUBJECT}_hull_debug.png")


def pick_frame():
    masks = sorted(glob.glob(os.path.join(MASKS_DIR, "*_product_mask.png")))
    if not masks:
        sys.exit(f"[generate_interior] no masks in {MASKS_DIR} — run segmentation")
    want = os.environ.get("INTERIOR_FRAME")
    if want:
        hit = [m for m in masks if os.path.basename(m).startswith(want)]
        if not hit:
            sys.exit(f"[generate_interior] INTERIOR_FRAME={want} not in {MASKS_DIR}")
        mask = hit[0]
    else:
        mask = masks[len(masks) // 2]           # middle frame of the view
    stem = os.path.basename(mask).replace("_product_mask.png", "")
    # recursive so FRAMES_DIR may be a per-view capture root or a flat folder
    frames = (glob.glob(os.path.join(FRAMES_DIR, stem + ".*"))
              or glob.glob(os.path.join(FRAMES_DIR, "**", stem + ".*"),
                           recursive=True))
    if not frames:
        sys.exit(f"[generate_interior] no frame for {stem} in {FRAMES_DIR}")
    return frames[0], mask, stem


def make_cutout(frame_path, mask_path):
    """frame×mask on gray, centred + scaled to FG_RATIO of a square. The frame
    is EXIF-transposed to match the mask (which was made on the applied image)."""
    frame = ImageOps.exif_transpose(Image.open(frame_path)).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if frame.size != mask.size:
        sys.exit(f"[generate_interior] frame{frame.size} != mask{mask.size} "
                 f"(EXIF mismatch?) — cannot align cutout")
    fa = np.asarray(frame)
    ma = np.asarray(mask) > 127
    ys, xs = np.where(ma)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = fa[y0:y1, x0:x1].copy()
    crop[~ma[y0:y1, x0:x1]] = GRAY
    h, w = crop.shape[:2]
    scale = (FG_RATIO * CUTOUT_SIZE) / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    crop_img = Image.fromarray(crop).resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (CUTOUT_SIZE, CUTOUT_SIZE), (GRAY, GRAY, GRAY))
    canvas.paste(crop_img, ((CUTOUT_SIZE - nw) // 2, (CUTOUT_SIZE - nh) // 2))
    canvas.save(CUTOUT_PNG)
    print(f"[generate_interior] cutout {w}x{h} -> {nw}x{nh} on "
          f"{CUTOUT_SIZE}px gray -> {os.path.basename(CUTOUT_PNG)}")


def reconstruct(triposr_dir):
    env = dict(os.environ, PYTHONPATH=triposr_dir)
    cmd = [sys.executable, os.path.join(triposr_dir, "run.py"), CUTOUT_PNG,
           "--no-remove-bg", "--device", DEVICE,
           "--mc-resolution", str(MC_RES), "--model-save-format", "glb",
           "--output-dir", TRIPOSR_OUT]
    print(f"[generate_interior] TripoSR reconstruct (mc={MC_RES}, {DEVICE}) ...")
    r = subprocess.run(cmd, cwd=triposr_dir, env=env)
    if r.returncode != 0:
        sys.exit("[generate_interior] TripoSR run.py failed")
    mesh_path = os.path.join(TRIPOSR_OUT, "0", "mesh.glb")
    if not os.path.exists(mesh_path):
        sys.exit(f"[generate_interior] TripoSR produced no mesh at {mesh_path}")
    return mesh_path


def apply_orient(mesh):
    """Permute vertex axes into the canonical carve frame. ORIENT is a 3-char
    axis spec (e.g. 'yzx' = new axes take old y,z,x). 'none'/'xyz' = identity.
    Object-dependent — inspect the debug PNG and adjust if the family lies down
    or faces backward."""
    spec = ORIENT.lower()
    if spec in ("none", "xyz", ""):
        return mesh
    axes = {"x": 0, "y": 1, "z": 2}
    perm = [axes[c] for c in spec if c in axes]
    if len(perm) != 3:
        print(f"[generate_interior] bad ORIENT '{ORIENT}' — skipping reorient")
        return mesh
    mesh.vertices = mesh.vertices[:, perm].copy()
    mesh.fix_normals()
    print(f"[generate_interior] oriented (ORIENT={spec}) extents={mesh.extents}")
    return mesh


def save_debug(mesh):
    vc = mesh.visual.vertex_colors if mesh.visual.kind == "vertex" else None
    tiles = []
    for az, lab in [(0, "FRONT(+Z)"), (90, "side"), (180, "back")]:
        im = raster_preview(mesh, size=340, azim=az, elev=-6,
                            vertex_colors=vc, shade_mix=0.5)
        ImageDraw.Draw(im).text((6, 6), lab, fill=(255, 0, 0))
        tiles.append(im)
    mont = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height),
                     (255, 255, 255))
    x = 0
    for t in tiles:
        mont.paste(t, (x, 0)); x += t.width
    mont.save(DEBUG_PNG)
    print(f"[generate_interior] debug views -> {os.path.basename(DEBUG_PNG)} "
          f"(VERIFY: front should show faces/front, up correct)")


def main():
    os.makedirs(WORK, exist_ok=True)
    frame, mask, stem = pick_frame()
    print(f"[generate_interior] {SUBJECT}: view={VIEW} frame={stem}")
    make_cutout(frame, mask)

    triposr_dir = setup_triposr.ensure()
    mesh_path = reconstruct(triposr_dir)

    mesh = trimesh.load(mesh_path, force="mesh")
    print(f"[generate_interior] reconstructed {len(mesh.vertices):,} verts, "
          f"{len(mesh.faces):,} faces")
    mesh = apply_orient(mesh)
    if SMOOTH_ITERS > 0:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.53, nu=0.53,
                                        iterations=SMOOTH_ITERS)
        print(f"[generate_interior] Taubin-smoothed x{SMOOTH_ITERS}")

    mesh.export(OUT_GLB)
    save_debug(mesh)
    print(f"[generate_interior] wrote {OUT_GLB} "
          f"({os.path.getsize(OUT_GLB)/1e6:.1f} MB) — next: texture_hull.py")


if __name__ == "__main__":
    main()
