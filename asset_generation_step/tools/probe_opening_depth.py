"""
Feasibility probe for depth-carving object openings (Report 21 follow-up).

Before wiring monocular depth into the Stage 3 carve, verify the premise on
our own captures: does Depth Anything V2 actually see INTO the opening — the
shoe mouth from the top view, the cap bowl from the bottom view — as deeper
than the surrounding rim? If this signal isn't in the depth maps, no carving
logic downstream can recover it.

For every mask in  masks/<SUBJECT>/<VIEW>/  this runs the same depth model
Step 5 uses on the matching frame from  instance_segmentation_step/frames/,
then writes per frame:

    work/depth_probe/<SUBJECT>/<name>_disparity.npy   raw model output (full frame)
    work/depth_probe/<SUBJECT>/<name>_probe.jpg       photo | product-masked disparity

and prints a cavity statistic: how far below the product's rim surface the
deepest in-mask pocket sits, as a fraction of the product's own disparity
range in that frame. Raw disparity is affine-ambiguous, so the number is a
CONTRAST measure for go/no-go judgement, not metres — metric alignment
against the hull rim is the carve's job, not the probe's.

Usage:  SUBJECT=nike-shoe VIEW=top venv/bin/python asset_generation_step/tools/probe_opening_depth.py
"""

import glob
import os
import sys

import cv2
import numpy as np

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # asset_generation_step/
ROOT_DIR   = os.path.dirname(BASE_DIR)
SUBJECT_ID = os.environ.get('SUBJECT', 'nike-shoe')
VIEW       = os.environ.get('VIEW', 'top')
FRAMES_DIR = os.path.join(ROOT_DIR, 'instance_segmentation_step', 'frames')
MASKS_DIR  = os.path.join(BASE_DIR, 'masks', SUBJECT_ID, VIEW)
OUT_DIR    = os.path.join(BASE_DIR, 'work', 'depth_probe', SUBJECT_ID)
# Small is Step 5's default; Base has sharper depth edges (less bleed at the
# rim/mask boundary), worth it for carving at the cost of a slower pass
DEPTH_MODEL = os.environ.get('DEPTH_MODEL',
                             'depth-anything/Depth-Anything-V2-Small-hf')


def frame_for_mask(mask_path):
    """masks are named <frame>_product_mask.png; frames are jpeg/jpg/png."""
    stem = os.path.basename(mask_path).replace('_product_mask.png', '')
    for ext in ('.jpeg', '.jpg', '.png', '.JPEG', '.JPG'):
        p = os.path.join(FRAMES_DIR, stem + ext)
        if os.path.exists(p):
            return p
    return None


def cavity_stats(disparity, mask):
    """Contrast of the deepest in-product pocket vs the rim surface.

    'Rim' = the high-disparity (near-camera) end of the product's own
    distribution (90th percentile); 'pocket floor' = the low end (5th).
    Returned contrast is their gap relative to the product's disparity
    spread — ~0 means the product reads as one flat surface (no cavity
    signal), larger means the model sees real interior depth. Also returns
    the pocket mask (pixels deeper than halfway down that gap) so the
    visualisation can outline where the model thinks the opening is.

    The mask is hole-filled and eroded first: at the silhouette edge the
    depth map blends product into the (deeper) table, and un-filled SAM
    dropouts on the insole would drop interior pixels — both skew the
    percentiles and, worse, make the 'pocket' hug the outline instead of
    the opening (first probe run did exactly that).
    """
    cnts, _ = cv2.findContours(mask.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, cnts, -1, 1, cv2.FILLED)
    k = max(9, mask.shape[1] // 100) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.erode(filled, kernel).astype(bool)
    vals = disparity[mask]
    rim, floor = np.percentile(vals, 90), np.percentile(vals, 5)
    spread = max(np.percentile(vals, 98) - np.percentile(vals, 2), 1e-6)
    contrast = (rim - floor) / spread
    pocket = mask & (disparity < (rim + floor) / 2)
    return contrast, rim, floor, pocket


def main():
    mask_paths = sorted(glob.glob(os.path.join(MASKS_DIR, '*_product_mask.png')))
    if not mask_paths:
        print(f"[!] no masks in {MASKS_DIR}")
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)

    from PIL import Image
    from transformers import pipeline as hf_pipeline
    print(f"Loading depth model '{DEPTH_MODEL}'...")
    depth_pipe = hf_pipeline(task='depth-estimation', model=DEPTH_MODEL)
    print("ready.\n")

    for mp in mask_paths:
        fp = frame_for_mask(mp)
        name = os.path.basename(mp).replace('_product_mask.png', '')
        if fp is None:
            print(f"[!] {name}: no frame found in {FRAMES_DIR} — skipped")
            continue

        out = depth_pipe(Image.open(fp).convert('RGB'))
        disparity = out['predicted_depth'].squeeze().cpu().numpy().astype(np.float32)

        frame = cv2.imread(fp)
        fh, fw = frame.shape[:2]
        if disparity.shape != (fh, fw):
            disparity = cv2.resize(disparity, (fw, fh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127

        np.save(os.path.join(OUT_DIR, f'{name}_disparity.npy'), disparity)

        contrast, rim, floor, pocket = cavity_stats(disparity, mask)
        print(f"{name}: cavity contrast {contrast:.2f}  "
              f"(rim p90 {rim:.2f} -> floor p5 {floor:.2f}, "
              f"pocket = {pocket.sum() / mask.sum():.0%} of product)")

        # side-by-side: photo | disparity inside the mask (bright = close),
        # with the detected pocket outlined in red on both panels
        vals = disparity[mask]
        lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
        norm = np.clip((disparity - lo) / max(hi - lo, 1e-6), 0, 1)
        vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        vis[~mask] = (40, 40, 40)
        cnts, _ = cv2.findContours(pocket.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for panel in (frame, vis):
            cv2.drawContours(panel, cnts, -1, (0, 0, 255), max(2, fw // 500))
        panel = np.hstack([frame, vis])
        scale = min(1.0, 1600 / panel.shape[1])
        panel = cv2.resize(panel, None, fx=scale, fy=scale)
        cv2.imwrite(os.path.join(OUT_DIR, f'{name}_probe.jpg'), panel)

    print(f"\nProbe images + raw disparity saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
