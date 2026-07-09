import cv2
import numpy as np
import os
import json
from datetime import datetime
from ultralytics import SAM
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Reuse the REAL pipeline functions so this experiment tests exactly what
# production runs: same undistortion, same detection call, same SAM prompting.
import segmentation as seg

# -------------------------------------- Configuration --------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR    = os.path.join(SCRIPT_DIR, 'prompt_experiments')

# One entry per product. 'frames_dir' holds that product's FRONT-view capture set
# (the view that produces width + height — 2 of the 3 output dimensions).
# Prompt variants deliberately span the phrasing spectrum, plus two controls:
#   wrong_attribute — a description that contradicts the product (does wording
#                     actually constrain the detector, or does it just find the
#                     most salient object anyway?)
#   placeholder     — meaningless text. A real run was once made with the literal
#                     'INSERT_PROMPT.' placeholder and still boxed the product,
#                     which is what motivated this control.
PRODUCTS = {
    'converse': {
        'frames_dir': os.path.join(SCRIPT_DIR, 'frames'),
        'prompts': {
            'short_label':      'shoe',
            'category':         'sneaker',
            'color_label':      'black shoe',
            'full_description': 'black canvas high-top sneaker',
            'over_specified':   'worn black converse all-star high-top sneaker with white laces',
            'wrong_attribute':  'red shoe',
            'placeholder':      'insert prompt',
        },
    },
    'handbag': {
        'frames_dir': os.path.join(EXP_DIR, 'frames', 'handbag'),
        'prompts': {
            'short_label':      'bag',
            'category':         'handbag',
            'color_label':      'red handbag',
            'full_description': 'red glossy leather handbag',
            'over_specified':   'shiny dark red crocodile-embossed leather handbag with two shoulder straps',
            'wrong_attribute':  'blue handbag',
            'placeholder':      'insert prompt',
        },
    },
    'nike': {
        'frames_dir': os.path.join(EXP_DIR, 'frames', 'nike'),
        'prompts': {
            'short_label':      'shoe',
            'category':         'sneaker',
            # NOTE: the A4 reference sheet is also white — this tests whether a
            # color attribute can pull the detector toward the wrong white object.
            'color_label':      'white shoe',
            'full_description': 'white mesh running sneaker',
            'over_specified':   'brand new white nike running sneaker with white laces and gray mesh panels',
            'wrong_attribute':  'black shoe',
            'placeholder':      'insert prompt',
        },
    },
}

# Prompt variants EXCLUDED from the per-frame consensus box (they are expected
# to misbehave; measuring them against the sensible prompts' consensus is the point).
CONTROL_VARIANTS = {'wrong_attribute', 'placeholder'}

# -------------------------------------- Helpers --------------------------------------

def mask_tight_box(mask):
    """Tight bounding box [x1,y1,x2,y2] of a binary mask — what measurement reads."""
    xs = cv2.findNonZero(mask)
    if xs is None:
        return None
    x, y, w, h = cv2.boundingRect(xs)
    return [x, y, x + w, y + h]


def box_iou(a, b):
    """Intersection-over-union of two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# -------------------------------------- Per-product experiment --------------------------------------

def run_product(product_id, config, dino_processor, dino_model, sam_model, camera_matrix, dist_coeffs):
    frames_dir = config['frames_dir']
    prompts    = config['prompts']

    frame_paths = sorted(
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if not frame_paths:
        print(f"[!] {product_id}: no frames in '{frames_dir}' — skipped.")
        return None

    print(f"\n=== {product_id}: {len(frame_paths)} frames × {len(prompts)} prompts ===\n")

    # records[variant] = list of per-frame dicts
    records = {v: [] for v in prompts}

    for i, fp in enumerate(frame_paths):
        name = os.path.basename(fp)
        frame = cv2.imread(fp)
        if frame is None:
            print(f"  [!] unreadable frame skipped: {name}")
            continue
        frame, _ = seg.undistort_frame(frame, camera_matrix, dist_coeffs)

        per_frame = {}
        for variant, prompt in prompts.items():
            box, label, conf = seg.detect_product(frame, dino_processor, dino_model, prompt)
            entry = {'frame': name, 'detected': box is not None,
                     'confidence': round(conf, 4), 'matched_label': label,
                     'dino_box': box, 'mask_box': None}
            if box is not None:
                # End-to-end: measurement reads the SAM mask's tight box, not
                # the DINO box, so mask-box geometry is the metric that matters.
                mask = seg.segment_with_sam2(frame, box, sam_model)
                if mask is not None:
                    entry['mask_box'] = mask_tight_box(mask)
            per_frame[variant] = entry
            records[variant].append(entry)

        # Per-frame consensus: element-wise median of the sensible prompts'
        # mask boxes. Deviation from it is reported in % of box size, which
        # reads directly as % measurement error downstream.
        sensible = [per_frame[v]['mask_box'] for v in prompts
                    if v not in CONTROL_VARIANTS and per_frame[v]['mask_box']]
        consensus = np.median(np.array(sensible), axis=0).tolist() if len(sensible) >= 2 else None

        for variant, entry in per_frame.items():
            mb = entry['mask_box']
            if consensus and mb:
                cw, ch = consensus[2] - consensus[0], consensus[3] - consensus[1]
                entry['iou_vs_consensus'] = round(box_iou(mb, consensus), 3)
                entry['width_dev_pct']  = round(((mb[2] - mb[0]) - cw) / cw * 100, 2)
                entry['height_dev_pct'] = round(((mb[3] - mb[1]) - ch) / ch * 100, 2)

        found = sum(1 for e in per_frame.values() if e['detected'])
        print(f"  [{i+1}/{len(frame_paths)}] {name}: {found}/{len(prompts)} prompts detected")

    # ---- Aggregate per prompt variant ----
    summary = {}
    for variant, entries in records.items():
        n = len(entries)
        det   = [e for e in entries if e['detected']]
        confs = [e['confidence'] for e in det]
        ious  = [e['iou_vs_consensus'] for e in det if 'iou_vs_consensus' in e]
        wdev  = [abs(e['width_dev_pct'])  for e in det if 'width_dev_pct'  in e]
        hdev  = [abs(e['height_dev_pct']) for e in det if 'height_dev_pct' in e]
        summary[variant] = {
            'prompt':            prompts[variant],
            'detection_rate':    f"{len(det)}/{n}",
            'mean_confidence':   round(float(np.mean(confs)), 3) if confs else None,
            'min_confidence':    round(float(np.min(confs)), 3) if confs else None,
            # How far the WEAKEST frame sits above BOX_THRESHOLD — the robustness
            # margin. Near zero means one shadow away from a missed detection.
            'min_margin_above_threshold': round(float(np.min(confs)) - seg.BOX_THRESHOLD, 3) if confs else None,
            'mean_iou_vs_consensus': round(float(np.mean(ious)), 3) if ious else None,
            'min_iou_vs_consensus':  round(float(np.min(ious)), 3) if ious else None,
            'mean_abs_width_dev_pct':  round(float(np.mean(wdev)), 2) if wdev else None,
            'max_abs_width_dev_pct':   round(float(np.max(wdev)), 2) if wdev else None,
            'mean_abs_height_dev_pct': round(float(np.mean(hdev)), 2) if hdev else None,
            'max_abs_height_dev_pct':  round(float(np.max(hdev)), 2) if hdev else None,
        }

    # ---- Print summary table ----
    print(f"\n{'variant':<17} {'det':>6} {'conf':>6} {'min':>6} {'margin':>7} "
          f"{'IoU':>6} {'|Δw|%':>7} {'|Δh|%':>7}  prompt")
    print("─" * 100)
    for variant, s in summary.items():
        print(f"{variant:<17} {s['detection_rate']:>6} "
              f"{s['mean_confidence'] if s['mean_confidence'] is not None else '—':>6} "
              f"{s['min_confidence'] if s['min_confidence'] is not None else '—':>6} "
              f"{s['min_margin_above_threshold'] if s['min_margin_above_threshold'] is not None else '—':>7} "
              f"{s['mean_iou_vs_consensus'] if s['mean_iou_vs_consensus'] is not None else '—':>6} "
              f"{s['mean_abs_width_dev_pct'] if s['mean_abs_width_dev_pct'] is not None else '—':>7} "
              f"{s['mean_abs_height_dev_pct'] if s['mean_abs_height_dev_pct'] is not None else '—':>7}  "
              f"'{s['prompt']}'")

    return {'product_id': product_id, 'run_at': datetime.now().isoformat(),
            'frames': len(frame_paths), 'box_threshold': seg.BOX_THRESHOLD,
            'text_threshold': seg.TEXT_THRESHOLD, 'summary': summary,
            'per_frame_records': records}


# -------------------------------------- Main --------------------------------------

def main():
    print("=== Prompt-Robustness Pass — Grounding DINO ===")

    camera_matrix, dist_coeffs = seg.load_calibration(seg.CALIB_FILE)

    print(f"Loading Grounding DINO '{seg.DINO_MODEL}' (device: {seg.DEVICE})...")
    dino_processor = AutoProcessor.from_pretrained(seg.DINO_MODEL)
    dino_model     = AutoModelForZeroShotObjectDetection.from_pretrained(seg.DINO_MODEL).to(seg.DEVICE).eval()
    print(f"Loading SAM 2 '{seg.SAM2_MODEL}'...")
    sam_model = SAM(seg.SAM2_MODEL)

    os.makedirs(EXP_DIR, exist_ok=True)
    for product_id, config in PRODUCTS.items():
        out = os.path.join(EXP_DIR, f'results_{product_id}.json')
        if os.path.exists(out):
            print(f"\n[i] {product_id}: results file already exists — skipped (delete '{out}' to re-run).")
            continue
        result = run_product(product_id, config, dino_processor, dino_model,
                             sam_model, camera_matrix, dist_coeffs)
        if result:
            out = os.path.join(EXP_DIR, f'results_{product_id}.json')
            with open(out, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
