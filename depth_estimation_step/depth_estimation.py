import cv2
import numpy as np
import os
import json
from PIL import Image
from transformers import pipeline as hf_pipeline

# ─── Configuration ────────────────────────────────────────────────────────────

SEG_RESULTS  = '../instance_segmentation_step/output/segmentation_results.json'
OUTPUT_DIR   = 'output'
DEPTH_MODEL  = 'depth-anything/Depth-Anything-V2-Small-hf'  # small = runs on M1 comfortably

A4_REAL_WIDTH_M = 0.210   # A4 sheet width in metres (210mm)

# ─── Load depth model ─────────────────────────────────────────────────────────

def load_depth_model(model_name):
    """
    Load the Depth Anything V2 model via the HuggingFace pipeline.
    Downloads the model weights on first run — this may take a moment.
    """
    print(f"Loading depth model '{model_name}'...")
    depth_pipe = hf_pipeline(
        task="depth-estimation",
        model=model_name
    )
    print("Depth model ready.\n")
    return depth_pipe


# ─── Run depth estimation ──────────────────────────────────────────────────────

def estimate_depth(frame_path, depth_pipe):
    """
    Run Depth Anything V2 on a single frame.

    Returns a float32 numpy array of the model's RAW output. Two things
    matter about this output and both were originally gotten wrong here:

      1. It is DISPARITY-like: HIGHER values mean CLOSER to the camera
         (verified empirically on our frames: near table edge ≈ high,
         far wall ≈ low). Distance is proportional to 1/value, so the
         conversion to metres happens in compute_metric_depth(), not by
         a simple multiply.
      2. We read output['predicted_depth'] (the raw float tensor), NOT
         output['depth'] — that one is a uint8 visualisation image,
         quantised to 256 levels, which destroys measurement precision.

    No per-frame min-max normalisation either: subtracting the frame
    minimum shifts every value and breaks the 1/value relationship that
    the metric conversion depends on. Scale anchoring is done per-frame
    anyway, so normalisation bought nothing.
    """
    image     = Image.open(frame_path).convert('RGB')
    output    = depth_pipe(image)
    disparity = output['predicted_depth'].squeeze().cpu().numpy().astype(np.float32)
    return disparity


# ─── Scale anchoring using A4 sheet ───────────────────────────────────────────

def compute_metric_depth(disparity_map, a4_box, camera_matrix):
    """
    Convert the raw disparity map to metric depth in metres, anchored on the A4 sheet.

    How it works:
      - We know the A4 sheet's real width: 210mm.
      - We measure its pixel width in the image (from the bounding box).
      - Using the pinhole camera model:
            pixel_width = (real_width_m * focal_length_px) / real_distance_m
        Rearranged:
            real_distance_m = (real_width_m * focal_length_px) / pixel_width
      - This gives us the real-world distance from the camera to the A4 sheet.
      - The model outputs disparity: value ≈ k / distance (higher = closer),
        with k unknown. The A4 anchor pins it down:
            k = real_distance_m * median_disparity(A4 region)
      - Every pixel's metric depth is then:  depth_m = k / disparity
        (NOT depth = disparity * scale — that linear form is only correct at
        the anchor itself and increasingly wrong everywhere else.)

    Caveat: the model's output is affine-invariant disparity (k/z + shift).
    With a single reference object we can only solve for k, so we assume the
    shift is ~0. Any residual systematic bias from that assumption is exactly
    what Step 7 (accuracy validation vs. tape measure) exists to catch.

    Returns (metric_depth_map, real_distance_m, k) or (None, None, None).
    """
    x1, y1, x2, y2 = a4_box
    a4_pixel_width  = x2 - x1

    if a4_pixel_width <= 0:
        return None, None, None

    # Focal length in pixels (from camera calibration)
    focal_length_px = camera_matrix[0, 0]

    # Estimated real-world distance to A4 sheet
    real_distance_m = (A4_REAL_WIDTH_M * focal_length_px) / a4_pixel_width

    # Sample the disparity map inside the A4 bounding box
    # Clip coordinates to stay within the map bounds
    h, w     = disparity_map.shape
    y1c, y2c = max(0, y1), min(h, y2)
    x1c, x2c = max(0, x1), min(w, x2)
    a4_region = disparity_map[y1c:y2c, x1c:x2c]

    if a4_region.size == 0:
        return None, None, None

    median_disparity = float(np.median(a4_region))

    if median_disparity <= 0:
        return None, None, None

    # k converts disparity → metres via depth = k / disparity
    k = real_distance_m * median_disparity

    # Clamp near-zero disparities (far background / sky) so the division
    # doesn't produce inf — those pixels are never measured anyway.
    metric_depth = k / np.maximum(disparity_map, 1e-6)

    return metric_depth, real_distance_m, k


# ─── Resize depth map to match frame ──────────────────────────────────────────

def resize_depth_to_frame(depth_map, frame_path):
    """
    The depth model may output at a different resolution than the input frame.
    Resize the depth map to match the frame so pixel coordinates line up exactly.
    """
    frame = cv2.imread(frame_path)
    fh, fw = frame.shape[:2]
    if depth_map.shape != (fh, fw):
        depth_map = cv2.resize(depth_map, (fw, fh), interpolation=cv2.INTER_LINEAR)
    return depth_map


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 5 — Depth Estimation + Scale Anchoring ===\n")

    # Load segmentation results from Step 3
    if not os.path.exists(SEG_RESULTS):
        print(f"[!] Segmentation results not found: '{SEG_RESULTS}'")
        print("    Run segmentation.py first.")
        return

    with open(SEG_RESULTS) as f:
        seg_results = json.load(f)

    print(f"Loaded segmentation results for {len(seg_results)} frame(s).\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the depth model once and reuse it across all frames
    depth_pipe = load_depth_model(DEPTH_MODEL)

    all_depth_results = []

    for i, seg in enumerate(seg_results):
        frame_path = seg['frame']
        base_name  = os.path.splitext(os.path.basename(frame_path))[0]
        print(f"Frame {i+1}/{len(seg_results)}: {os.path.basename(frame_path)}")

        # Skip frames where the A4 sheet was not detected — we can't anchor scale without it
        if not seg['a4_sheet']['detected']:
            print("  [!] A4 sheet missing — skipping this frame (cannot anchor scale).")
            print()
            continue

        # Skip frames where the product was not detected
        if not seg['product']['detected']:
            print("  [!] Product not detected — skipping this frame.")
            print()
            continue

        # Run depth estimation on this frame (raw disparity — higher = closer)
        print("  Running depth estimation...")
        disparity_map = estimate_depth(frame_path, depth_pipe)

        # Resize disparity map to match frame dimensions so coordinates align
        disparity_map = resize_depth_to_frame(disparity_map, frame_path)

        # Convert disparity → metric depth (metres), anchored on the A4 sheet
        camera_matrix = np.array(seg['camera_matrix'])
        a4_box        = seg['a4_sheet']['box_xyxy']
        depth_metric, estimated_distance, k = compute_metric_depth(disparity_map, a4_box, camera_matrix)

        if depth_metric is None or estimated_distance is None or k is None:
            print("  [!] Scale anchoring failed — A4 region had no valid disparity values.")
            print()
            continue

        print(f"  [✓] Estimated distance to A4 sheet: {estimated_distance:.3f}m")
        print(f"  [✓] Anchor constant k: {k:.4f} (depth_m = k / disparity)")

        # Save the METRIC depth map (metres) — measurement_extraction.py loads it as-is
        depth_path = os.path.join(OUTPUT_DIR, f'{base_name}_depth.npy')
        np.save(depth_path, depth_metric)

        # Save a visual version for inspection (colourised disparity: bright = close)
        d_min, d_max = disparity_map.min(), disparity_map.max()
        depth_vis    = ((disparity_map - d_min) / max(d_max - d_min, 1e-6) * 255).astype(np.uint8)
        depth_colour = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f'{base_name}_depth_vis.jpg'), depth_colour)

        all_depth_results.append({
            'frame':              frame_path,
            'depth_map_path':     os.path.abspath(depth_path),
            'depth_map_units':    'metres',
            'anchor_constant_k':  k,
            'estimated_distance_m': round(estimated_distance, 4),
            'camera_matrix':      seg['camera_matrix'],
            'product':            seg['product'],
            'a4_sheet':           seg['a4_sheet'],
        })

        print()

    # Write depth results JSON — consumed by measurement_extraction.py
    results_path = os.path.join(OUTPUT_DIR, 'depth_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_depth_results, f, indent=2)

    print("─" * 45)
    print(f"Results saved to : {results_path}")
    print(f"Frames processed : {len(all_depth_results)}/{len(seg_results)}")
    print("\nStep 5 complete. Check output/depth/ for colourised depth maps.")


if __name__ == '__main__':
    main()
