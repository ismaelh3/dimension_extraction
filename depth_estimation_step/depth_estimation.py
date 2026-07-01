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

    Returns a float32 numpy array (same height/width as the frame) where
    each value is the model's relative depth estimate for that pixel.
    Higher values mean further from the camera in this model's convention.

    Note: these values are unitless — we scale them to metres in the next step.
    """
    image     = Image.open(frame_path).convert('RGB')
    output    = depth_pipe(image)
    depth_pil = output['depth']                             # PIL image
    depth_arr = np.array(depth_pil, dtype=np.float32)      # convert to numpy

    # Normalise to 0–1 range so the scale factor is consistent across frames
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max - d_min > 0:
        depth_arr = (depth_arr - d_min) / (d_max - d_min)

    return depth_arr


# ─── Scale anchoring using A4 sheet ───────────────────────────────────────────

def compute_scale_factor(depth_map, a4_box, camera_matrix):
    """
    Convert the relative depth map to metric (metres) using the A4 sheet as reference.

    How it works:
      - We know the A4 sheet's real width: 210mm.
      - We measure its pixel width in the image (from the bounding box).
      - Using the pinhole camera model:
            pixel_width = (real_width_m * focal_length_px) / real_distance_m
        Rearranged:
            real_distance_m = (real_width_m * focal_length_px) / pixel_width
      - This gives us the real-world distance from the camera to the A4 sheet.
      - We then sample the relative depth values inside the A4 bounding box and
        compute their median. That median relative depth corresponds to the real distance.
      - Scale factor = real_distance_m / median_relative_depth
      - Multiply any pixel's relative depth by this factor to get its distance in metres.
    """
    x1, y1, x2, y2 = a4_box
    a4_pixel_width  = x2 - x1

    if a4_pixel_width <= 0:
        return None, None

    # Focal length in pixels (from camera calibration)
    focal_length_px = camera_matrix[0, 0]

    # Estimated real-world distance to A4 sheet
    real_distance_m = (A4_REAL_WIDTH_M * focal_length_px) / a4_pixel_width

    # Sample the relative depth map inside the A4 bounding box
    # Clip coordinates to stay within the depth map bounds
    h, w     = depth_map.shape
    y1c, y2c = max(0, y1), min(h, y2)
    x1c, x2c = max(0, x1), min(w, x2)
    a4_region = depth_map[y1c:y2c, x1c:x2c]

    if a4_region.size == 0:
        return None, None

    median_relative_depth = float(np.median(a4_region))

    if median_relative_depth <= 0:
        return None, None

    # Scale factor converts relative depth values → metres
    scale_factor = real_distance_m / median_relative_depth

    return scale_factor, real_distance_m


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

        # Run depth estimation on this frame
        print("  Running depth estimation...")
        depth_map = estimate_depth(frame_path, depth_pipe)

        # Resize depth map to match frame dimensions so coordinates align
        depth_map = resize_depth_to_frame(depth_map, frame_path)

        # Compute scale factor using the A4 sheet bounding box
        camera_matrix = np.array(seg['camera_matrix'])
        a4_box        = seg['a4_sheet']['box_xyxy']
        scale_factor, estimated_distance = compute_scale_factor(depth_map, a4_box, camera_matrix)

        if scale_factor is None:
            print("  [!] Scale anchoring failed — A4 region had no valid depth values.")
            print()
            continue

        print(f"  [✓] Estimated distance to A4 sheet: {estimated_distance:.3f}m")
        print(f"  [✓] Scale factor: {scale_factor:.4f} (relative depth → metres)")

        # Save the depth map as a .npy file so measurement_extraction.py can load it directly
        depth_path = os.path.join(OUTPUT_DIR, f'{base_name}_depth.npy')
        np.save(depth_path, depth_map)

        # Save a visual version of the depth map for inspection (colourised)
        depth_vis    = (depth_map * 255).astype(np.uint8)
        depth_colour = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f'{base_name}_depth_vis.jpg'), depth_colour)

        all_depth_results.append({
            'frame':              frame_path,
            'depth_map_path':     os.path.abspath(depth_path),
            'scale_factor':       scale_factor,
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
