"""
measure_all.py — dimension extraction for ALL views from a SINGLE upload.

OLD flow (the pain): upload front frames -> run segment/depth/measure VIEW=front
-> DELETE, upload side frames -> run again VIEW=side -> merge. The swap is forced
because segmentation/depth share non-view-namespaced intermediate JSONs
(segmentation_results.json / depth_results.json), so each view must finish before
the next overwrites them.

NEW flow: drop every capture ONCE into per-view subfolders under one root:

    <root>/front/*.jpg    <root>/side/*.jpg    [<root>/back/  <root>/top/ ...]

then

    SUBJECT=x FRAMES_ROOT=<root> venv/bin/python measurement_extraction_step/measure_all.py
    (or: SUBJECT=x FRAMES_ROOT=<root> make measure-all)

loops each view (segment -> depth -> measure), files that view's product masks into
asset_generation_step/masks/<subject>/<view>/ (so the CARVE path has them), and
merges front+side into the final measurements_<subject>.json. One upload, no swap.

Only 'front' and 'side' feed the measurement math (W×H from front, front-to-back D
from side). Any other view subfolder is still segmented + filed so shape-from-
silhouette carving has masks for it — it just isn't measured.
"""

import glob
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "asset_generation_step", "pipeline"))
import glassiness  # noqa: E402
PY = sys.executable
SEG   = os.path.join(REPO, "instance_segmentation_step", "segmentation.py")
DEPTH = os.path.join(REPO, "depth_estimation_step", "depth_estimation.py")
MEAS  = os.path.join(REPO, "measurement_extraction_step", "measurement_extraction.py")
MERGE = os.path.join(REPO, "measurement_extraction_step", "merge_views.py")
# segmentation MUST write here (its default) — depth reads segmentation_results.json
# from this hardcoded location.
SEG_OUT = os.path.join(REPO, "instance_segmentation_step", "output")
MASKS   = os.path.join(REPO, "asset_generation_step", "masks")
FINAL   = os.path.join(REPO, "measurement_extraction_step", "output")

SUBJECT = os.environ.get("SUBJECT", "product_000")
ROOT    = (os.environ.get("FRAMES_ROOT") or os.environ.get("FRAMES_DIR")
           or os.path.join(REPO, "instance_segmentation_step", "frames"))
PROMPT  = os.environ.get("PRODUCT_PROMPT", "product.")
MULTI   = os.environ.get("MULTI_INSTANCE", "0")
MEASURE_VIEWS = ("front", "side")          # only these feed the measurement math
IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def run(script, env_extra, label):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_extra.items()})
    print(f"\n----- [measure-all] {label} -----")
    if subprocess.run([PY, script], env=env, cwd=REPO).returncode != 0:
        sys.exit(f"[measure-all] FAILED at: {label}")


def frames_in(d):
    fs = []
    for e in IMG_EXT:
        fs += glob.glob(os.path.join(d, e))
    return fs


def discover_views(root):
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and frames_in(os.path.join(root, d)))


def file_masks(view_dir, view):
    """Copy THIS view's product masks (matched by frame stem) out of the shared
    segmentation output into masks/<subject>/<view>/ before the next view
    overwrites the segmentation output. Copy (not move) so measurement can still
    read them from the segmentation output for this view."""
    dst = os.path.join(MASKS, SUBJECT, view)
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in frames_in(view_dir):
        stem = os.path.splitext(os.path.basename(f))[0]
        m = os.path.join(SEG_OUT, f"{stem}_product_mask.png")
        if os.path.exists(m):
            shutil.copy(m, os.path.join(dst, f"{stem}_product_mask.png"))
            n += 1
    print(f"[measure-all] filed {n} '{view}' mask(s) -> masks/{SUBJECT}/{view}/")


def main():
    if not os.path.isdir(ROOT):
        sys.exit(f"[measure-all] capture root not found: {ROOT}")
    views = discover_views(ROOT)
    if not views:
        sys.exit(f"[measure-all] no per-view subfolders with images under {ROOT}\n"
                 f"    expected e.g. {ROOT}/front/  {ROOT}/side/")
    print(f"[measure-all] {SUBJECT}: {len(views)} view(s) {views} under {ROOT}")

    depth_json = os.path.join(REPO, "depth_estimation_step", "output",
                              "depth_results.json")
    for v in views:
        vdir = os.path.join(ROOT, v)
        run(SEG, {"SUBJECT": SUBJECT, "PRODUCT_PROMPT": PROMPT,
                  "MULTI_INSTANCE": MULTI, "FRAMES_DIR": vdir,
                  "OUTPUT_DIR": SEG_OUT}, f"segment [{v}]")
        file_masks(vdir, v)
        used_depth = v in MEASURE_VIEWS
        if used_depth:
            run(DEPTH, {}, f"depth [{v}]")
            run(MEAS, {"SUBJECT": SUBJECT, "VIEW": v}, f"measure [{v}]")
        else:
            print(f"[measure-all] '{v}' filed for carving (not a measurement view)")
        # glassiness map per frame (depth see-through when this view ran depth,
        # else specular-only) — feeds region-level glass detection at bake time.
        glassiness.file_view_glass_maps(
            vdir, os.path.join(MASKS, SUBJECT, v),
            depth_json if used_depth else None)

    run(MERGE, {"SUBJECT": SUBJECT}, "merge front+side")
    final = os.path.join(FINAL, f"measurements_{SUBJECT}.json")
    if os.path.exists(final):
        print(f"\n[measure-all] DONE — {os.path.relpath(final, REPO)} "
              f"(masks filed under masks/{SUBJECT}/) — ready to carve")
    else:
        print("\n[measure-all] merge produced no final JSON — need both a 'front' "
              "and a 'side' subfolder for full W×H×D")


if __name__ == "__main__":
    main()
