import cv2
import numpy as np
import os
import json
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────

DEPTH_RESULTS = '../depth_estimation_step/output/depth_results.json'
OUTPUT_DIR    = 'output'
SUBJECT_ID    = 'product_001'   # change this per product you measure

# ─── Pixel → world coordinate conversion ──────────────────────────────────────

def pixel_to_world(px, py, depth_m, camera_matrix):
    """
    Convert a single pixel (px, py) at a known real-world depth into a 3D world point.

    Uses the standard pinhole camera model:
        X = (px - cx) * depth / fx
        Y = (py - cy) * depth / fy
        Z = depth

    Where fx, fy are focal lengths and cx, cy is the principal point —
    all from the camera calibration matrix.
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    X  = (px - cx) * depth_m / fx
    Y  = (py - cy) * depth_m / fy
    return np.array([X, Y, depth_m])


def measure_3d_distance(p1_px, p2_px, depth_map_metric, camera_matrix):
    """
    Compute the real-world distance (in metres) between two pixel points,
    using the metric depth map to get each point's depth.

    Clamps pixel indices to valid array bounds to avoid out-of-range errors.
    """
    h, w = depth_map_metric.shape

    # Clamp coordinates to valid array indices
    x1 = int(np.clip(p1_px[0], 0, w - 1))
    y1 = int(np.clip(p1_px[1], 0, h - 1))
    x2 = int(np.clip(p2_px[0], 0, w - 1))
    y2 = int(np.clip(p2_px[1], 0, h - 1))

    d1 = float(depth_map_metric[y1, x1])
    d2 = float(depth_map_metric[y2, x2])

    p1_world = pixel_to_world(x1, y1, d1, camera_matrix)
    p2_world = pixel_to_world(x2, y2, d2, camera_matrix)

    return float(np.linalg.norm(p1_world - p2_world))


# ─── Extract product bounding box from mask ────────────────────────────────────

def get_mask_tight_bbox(mask_path):
    """
    Load the product segmentation mask and find the tightest bounding box
    around the actual masked pixels (not the YOLO predicted box).

    Using the mask rather than the YOLO box gives more accurate edges —
    it excludes any padding or background that the box might include.

    Returns (x1, y1, x2, y2) or None if mask is empty.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # Find the coordinates of all non-zero (masked) pixels
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    return (x, y, x + w, y + h)


# ─── Measure a single frame ────────────────────────────────────────────────────

def measure_frame(depth_result):
    """
    Extract product width and height from one frame.

    Approach:
      1. Load the metric depth map (relative depth × scale factor).
      2. Get the tight bounding box of the product mask.
      3. Identify the four corner pixels of that bounding box.
      4. Convert corners to 3D world points using the depth at each pixel.
      5. Compute real-world width (left edge → right edge) and
         height (top edge → bottom edge).

    Returns a dict of measurements in metres, or None if anything is missing.
    """
    # Load depth map and convert from relative to metric using the scale factor
    depth_map_relative = np.load(depth_result['depth_map_path'])
    scale_factor       = depth_result['scale_factor']
    depth_map_metric   = depth_map_relative * scale_factor   # now in metres

    camera_matrix = np.array(depth_result['camera_matrix'])

    # Get tight bounding box from the saved product mask
    mask_path = depth_result['product']['mask_path']
    if not mask_path or not os.path.exists(mask_path):
        print("  [!] Product mask not found — skipping.")
        return None

    bbox = get_mask_tight_bbox(mask_path)
    if bbox is None:
        print("  [!] Product mask is empty — skipping.")
        return None

    x1, y1, x2, y2 = bbox

    # Define corner pixels of the bounding box
    # We measure width along the middle row and height along the middle column
    # to avoid edge pixels that might have noisy depth values
    mid_y = (y1 + y2) // 2
    mid_x = (x1 + x2) // 2

    left_px  = (x1, mid_y)
    right_px = (x2, mid_y)
    top_px   = (mid_x, y1)
    bottom_px= (mid_x, y2)

    # Measure real-world width (horizontal span of the product)
    width_m  = measure_3d_distance(left_px,   right_px,  depth_map_metric, camera_matrix)

    # Measure real-world height (vertical span of the product)
    height_m = measure_3d_distance(top_px,    bottom_px, depth_map_metric, camera_matrix)

    # Estimated depth of the product itself — sampled at mask centre
    product_depth_m = float(depth_map_metric[mid_y, mid_x])

    print(f"  Width  : {width_m  * 100:.2f} cm")
    print(f"  Height : {height_m * 100:.2f} cm")

    return {
        'width_m':        width_m,
        'height_m':       height_m,
        'product_depth_m': product_depth_m,   # camera-to-product distance, not object depth
    }


# ─── Multi-frame averaging with outlier rejection ──────────────────────────────

def robust_average(values):
    """
    Average a list of measurements while rejecting outliers.

    Keeps only values within 1 standard deviation of the mean.
    This discards frames with bad depth estimates or partial occlusions
    without needing to manually flag them.

    Returns (mean, std) of the filtered values, or (mean, 0) if only one value.
    """
    arr  = np.array(values)
    mean = np.mean(arr)
    std  = np.std(arr)

    # If all values are identical (std = 0), skip filtering
    if std == 0:
        return float(mean), 0.0

    filtered = arr[np.abs(arr - mean) < std]

    # Fall back to full array if filtering removes everything
    if len(filtered) == 0:
        filtered = arr

    return float(np.mean(filtered)), float(np.std(filtered))


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 6 — Product Measurement Extraction ===\n")

    # Load depth results from Step 5
    if not os.path.exists(DEPTH_RESULTS):
        print(f"[!] Depth results not found: '{DEPTH_RESULTS}'")
        print("    Run depth_estimation.py first.")
        return

    with open(DEPTH_RESULTS) as f:
        depth_results = json.load(f)

    print(f"Loaded depth data for {len(depth_results)} frame(s).\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect measurements from every valid frame
    width_measurements  = []
    height_measurements = []

    for i, dr in enumerate(depth_results):
        frame_name = os.path.basename(dr['frame'])
        print(f"Frame {i+1}/{len(depth_results)}: {frame_name}")

        measurements = measure_frame(dr)
        if measurements is None:
            print()
            continue

        width_measurements.append(measurements['width_m'])
        height_measurements.append(measurements['height_m'])
        print()

    if not width_measurements:
        print("[!] No valid measurements collected. Check that segmentation and depth steps ran correctly.")
        return

    # Average across frames, discarding outliers
    final_width_m,  width_err_m  = robust_average(width_measurements)
    final_height_m, height_err_m = robust_average(height_measurements)

    # Convert to centimetres for the output
    final_width_cm  = round(final_width_m  * 100, 1)
    final_height_cm = round(final_height_m * 100, 1)
    width_err_cm    = round(width_err_m    * 100, 2)
    height_err_cm   = round(height_err_m   * 100, 2)

    print("─" * 45)
    print(f"Final width  : {final_width_cm} cm  ± {width_err_cm} cm  ({len(width_measurements)} frames)")
    print(f"Final height : {final_height_cm} cm  ± {height_err_cm} cm  ({len(height_measurements)} frames)")

    # Build output JSON for Stage 3 (asset generation)
    output = {
        "subject_id":     SUBJECT_ID,
        "captured_at":    datetime.now().isoformat(),
        "frame_count":    len(width_measurements),
        "measurements_cm": {
            "width":  final_width_cm,
            "height": final_height_cm,
            # Depth (front-to-back) requires a side-view capture — add here when available
            "depth":  None,
        },
        "error_estimates_cm": {
            "width":  width_err_cm,
            "height": height_err_cm,
        },
        "reference_object": "A4_sheet_210x297mm",
        "model_versions": {
            "segmentation":     "yolov8n-seg",
            "depth_estimation": "Depth-Anything-V2-Small",
        },
        "notes": (
            "Depth (front-to-back) dimension not measured — requires a separate side-view capture. "
            "Run both a front and side capture set and merge the JSONs for a full 3D profile."
        )
    }

    out_path = os.path.join(OUTPUT_DIR, f'measurements_{SUBJECT_ID}.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nOutput saved to: {out_path}")
    print("\nStep 6 complete. Pass this JSON to Stage 3 (asset generation).")


if __name__ == '__main__':
    main()
