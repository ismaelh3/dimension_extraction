import cv2
import numpy as np
import os
import json
import pickle
from ultralytics import YOLOE, SAM

# -------------------------------------- Configuration --------------------------------------

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))  # folder this script lives in, so paths work from any cwd
FRAMES_DIR  = os.path.join(SCRIPT_DIR, 'frames')          # folder containing your captured product frames
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, 'output')          # all results written here (local to this step)
CALIB_FILE  = os.path.join(SCRIPT_DIR, '..', 'camera_calibration_step', 'output', 'calibration_data.pkl')
YOLO_MODEL  = os.path.join(SCRIPT_DIR, 'yoloe-26s-seg.pt')  # YOLOE-26 small — open-vocabulary segmentation

# What to segment. YOLOE is open-vocabulary: describe the product in a word or two
# ('shoe', 'water bottle', 'cardboard box') — change this per product you measure.
# The COCO-class limitation of plain YOLO does not apply here.
PRODUCT_PROMPT = 'INSERT_PROMPT'

# Inference resolution. The measurements are taken from the mask's edges, so run the
# network at a higher resolution than the default 640 for cleaner boundaries.
IMG_SIZE = 1280

# SAM 2 mask refinement. YOLOE stays responsible for FINDING the product (its
# open-vocabulary box), but its prototype-based masks are soft at object
# boundaries — and the measurement step reads dimensions off exactly that
# boundary. When enabled, YOLOE's box is passed to SAM 2 as a prompt and the
# SAM 2 mask replaces YOLOE's. The raw YOLOE mask is still saved alongside
# (*_product_mask_yoloe.png) so the two can be compared visually.
SAM2_REFINE = True
SAM2_MODEL  = os.path.join(SCRIPT_DIR, 'sam2.1_b.pt')  # downloads automatically on first run

# A4 sheet is 210mm wide × 297mm tall (portrait).
# These constants control how the detector decides what counts as an A4 sheet.
A4_ASPECT_RATIO     = 297 / 210   # ideal portrait aspect ratio ≈ 1.414
A4_ASPECT_TOLERANCE = 0.30        # allow ±30% deviation from ideal (handles slight angle)
A4_MIN_AREA_FRAC    = 0.003       # sheet must cover at least 0.3% of frame area
A4_MAX_AREA_FRAC    = 0.30        # and no more than 30% (rejects accidental full-frame white)

# The sheet is detected as a region that is both bright AND smooth (low local texture
# variance) — this distinguishes a plain paper sheet from a background that happens to
# be similarly bright but textured (e.g. patterned wallpaper), which a plain color/
# brightness threshold cannot tell apart.
A4_TEXTURE_KSIZE      = 15   # window size (px) used to estimate local texture variance
A4_TEXTURE_STD_MAX    = 8    # local grayscale std-dev must be below this to count as "smooth"
A4_BRIGHTNESS_MIN     = 150  # local mean brightness (0-255) must be above this to count as "bright"

# -------------------------------------- Load calibration data --------------------------------------

def load_calibration(path):
    """Read the camera matrix and distortion coefficients saved by camera_calibration.py."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['camera_matrix'], data['distortion_coefficients']


# -------------------------------------- Undistort frame --------------------------------------

def undistort_frame(frame, camera_matrix, dist_coeffs):
    """
    Remove lens distortion from a raw captured frame using the calibration data.
    This must happen before any pixel measurements are taken.

    NOTE: we deliberately do NOT crop to the "valid pixel" ROI that
    cv2.getOptimalNewCameraMatrix reports. Our current calibration checkerboard
    photos never covered the edges/corners of the frame (see camera_calibration_step),
    so the distortion model is poorly constrained there and produces an unstable,
    overly aggressive ROI that cuts real scene content (e.g. the top of the A4 sheet)
    out of the frame. Keeping the full canvas avoids losing real content; any thin
    black border introduced by undistortion does not interfere with detection.
    """
    h, w = frame.shape[:2]
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1, newImgSize=(w, h)
    )
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_matrix)
    return undistorted, new_matrix


# -------------------------------------- Product segmentation (YOLOE) --------------------------------------

def detect_product(frame, model):
    """
    Run YOLOE on the frame and return the mask + bounding box for the target product.

    The model is text-prompted with PRODUCT_PROMPT (set up in main), so everything it
    returns is already "the product" — no class filtering needed here. If several
    instances match the prompt, the most confident one is kept.

    retina_masks=True makes ultralytics build the mask at the original frame
    resolution instead of the network's internal size — sharper edges, which matter
    because the measurement step reads dimensions off this mask's boundary.

    Returns the binary mask (white = product area), bounding box [x1,y1,x2,y2],
    detected class name, and confidence score.
    """
    results = model(frame, imgsz=IMG_SIZE, retina_masks=True, verbose=False)

    best_conf      = 0.0
    best_mask      = None
    best_box       = None
    best_class     = None

    for result in results:
        if result.masks is None:
            continue

        for i, cls_id in enumerate(result.boxes.cls):
            conf = float(result.boxes.conf[i])
            if conf <= best_conf:
                continue  # keep only the most confident detection

            best_conf  = conf
            best_class = result.names[int(cls_id)]

            raw_mask = result.masks.data[i].cpu().numpy()
            h, w = frame.shape[:2]
            if raw_mask.shape != (h, w):
                raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            best_mask = (raw_mask > 0.5).astype(np.uint8) * 255

            box = result.boxes.xyxy[i].cpu().numpy().astype(int)
            best_box = box.tolist()

    return best_mask, best_box, best_class, best_conf


# -------------------------------------- SAM 2 mask refinement --------------------------------------

def refine_mask_with_sam2(frame, box, sam_model):
    """
    Re-segment the product with SAM 2, prompted by YOLOE's bounding box.

    SAM 2 is class-agnostic — it doesn't know what a 'bottle' is, it just
    segments whatever object the box points at, with boundary quality that
    YOLO-family mask heads can't match. The box prompt is what ties it to the
    product YOLOE identified.

    Returns a binary mask at frame resolution, or None if SAM 2 returned nothing
    (caller keeps the YOLOE mask in that case).
    """
    results = sam_model(frame, bboxes=[box], verbose=False)
    if not results or results[0].masks is None or len(results[0].masks.data) == 0:
        return None

    raw_mask = results[0].masks.data[0].cpu().numpy()
    h, w = frame.shape[:2]
    if raw_mask.shape != (h, w):
        raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (raw_mask > 0.5).astype(np.uint8) * 255


# -------------------------------------- A4 sheet detection (OpenCV contours) --------------------------------------

def detect_a4_sheet(frame):
    """
    Detect the A4 reference sheet in the frame using a smoothness + brightness mask,
    followed by contour/shape analysis.

    A plain absolute-brightness threshold (e.g. plain HSV "is this pixel white")
    is not reliable against non-plain backgrounds: a beige/textured wall can fall
    in the same brightness range as the paper, so the wall and the paper merge into
    one giant contour that gets rejected for being too large, and the sheet is never
    detected. Paper is not just bright — it is also visually SMOOTH (very low local
    variance), whereas textured backgrounds (wallpaper, wood grain, fabric) are bright
    in places but noisy at the pixel level. Combining "bright" AND "smooth" isolates
    the sheet even against similarly-bright, textured surroundings.

    Strategy:
      1. Compute local mean and local standard deviation of grayscale intensity.
      2. Keep only regions that are both bright and low-variance ("paper-like").
      3. Find contours and filter by area and shape.
      4. Approximate each contour to a polygon and keep 4-sided ones.
      5. Choose the candidate whose aspect ratio is closest to A4 (1.414).

    Returns the 4-corner polygon (numpy array shape [4,2]) and bounding rect,
    or (None, None) if nothing plausible is found.
    """
    h, w = frame.shape[:2]
    frame_area = h * w

    # Step 1 — local mean/std of grayscale intensity via a sliding box filter
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ksize  = A4_TEXTURE_KSIZE
    mean   = cv2.boxFilter(gray, -1, (ksize, ksize))
    sqmean = cv2.boxFilter(gray * gray, -1, (ksize, ksize))
    std    = np.sqrt(np.maximum(sqmean - mean * mean, 0))

    # Step 2 — keep only bright AND smooth ("paper-like") regions
    low_texture = (std  < A4_TEXTURE_STD_MAX).astype(np.uint8) * 255
    bright      = (mean > A4_BRIGHTNESS_MIN).astype(np.uint8) * 255
    white_mask  = cv2.bitwise_and(low_texture, bright)

    # Step 3 — clean up noise with morphological operations
    kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN,  kernel)  # remove tiny blobs
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)  # fill small holes

    # Step 4 — find all contours in the mask
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_candidate  = None
    best_box        = None
    best_ratio_diff = float('inf')

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Reject contours that are too small or suspiciously large
        if not (A4_MIN_AREA_FRAC * frame_area < area < A4_MAX_AREA_FRAC * frame_area):
            continue

        # Step 5 — approximate the contour to a polygon
        perimeter = cv2.arcLength(cnt, closed=True)
        epsilon   = 0.04 * perimeter   # tolerance: 4% of perimeter
        approx    = cv2.approxPolyDP(cnt, epsilon, closed=True)

        # We want a 4-sided shape (rectangle / quadrilateral)
        if len(approx) != 4:
            continue

        # Step 6 — check aspect ratio against known A4 dimensions
        x, y, bw, bh = cv2.boundingRect(approx)
        if bw == 0 or bh == 0:
            continue

        # Account for both portrait and landscape orientations
        ratio      = max(bw, bh) / min(bw, bh)
        ratio_diff = abs(ratio - A4_ASPECT_RATIO)

        if ratio_diff > A4_ASPECT_TOLERANCE:
            continue  # shape is not A4-like enough

        # Keep the candidate with ratio closest to ideal A4
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_candidate  = approx.reshape(4, 2)
            best_box        = [x, y, x + bw, y + bh]

    return best_candidate, best_box


# -------------------------------------- Debug visualisation --------------------------------------

def draw_detections(frame, product_mask, product_box, product_class, a4_corners, a4_box):
    """
    Overlay detection results on a copy of the frame for visual inspection.
    - Product mask shown as a red tint
    - Product bounding box in red
    - A4 corners drawn as a blue quadrilateral
    - A4 bounding box in blue
    """
    vis = frame.copy()

    # Red overlay for product mask
    if product_mask is not None:
        red_layer = np.zeros_like(frame)
        red_layer[:, :, 2] = 180  # red channel
        product_region = cv2.bitwise_and(red_layer, red_layer, mask=product_mask)
        vis = cv2.addWeighted(vis, 1.0, product_region, 0.4, 0)

    # Red bounding box around product
    if product_box is not None:
        x1, y1, x2, y2 = product_box
        label = product_class if product_class else 'product'
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 60, 220), 2)
        cv2.putText(vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 220), 2)

    # Blue quadrilateral for A4 sheet corners
    if a4_corners is not None:
        pts = a4_corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 100, 0), thickness=2)

    # Blue bounding box around A4
    if a4_box is not None:
        x1, y1, x2, y2 = a4_box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(vis, 'A4 ref', (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    return vis


# -------------------------------------- Process a single frame --------------------------------------

def process_frame(frame_path, model, camera_matrix, dist_coeffs, output_dir, sam_model=None):
    """
    Full Step 3 pipeline for one frame:
      1. Load and undistort the image.
      2. Detect the product with YOLO.
      3. Optionally refine the product mask with SAM 2 (box-prompted).
      4. Detect the A4 sheet with OpenCV.
      5. Save a labelled debug image.
      6. Return a result dict for this frame.
    """
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"  [!] Could not read: {frame_path}")
        return None

    # 1 — Remove lens distortion using calibration data
    frame, camera_matrix = undistort_frame(frame, camera_matrix, dist_coeffs)

    # 2 — YOLO product segmentation
    product_mask, product_box, product_class, product_conf = detect_product(frame, model)
    mask_source = 'yoloe-26s-seg'
    yoloe_mask  = None
    if product_mask is None:
        print(f"  [!] No product detected in {os.path.basename(frame_path)}")
    else:
        print(f"  [✓] Product detected  — class '{product_class}', confidence {product_conf:.2f}, box {product_box}")

        # 3 — SAM 2 refinement: same product, sharper boundary
        if sam_model is not None:
            refined = refine_mask_with_sam2(frame, product_box, sam_model)
            if refined is not None:
                yoloe_mask   = product_mask   # kept for visual comparison
                product_mask = refined
                mask_source  = 'yoloe-26s-seg + sam2.1_b refine'
                print(f"  [✓] Mask refined with SAM 2")
            else:
                print(f"  [!] SAM 2 returned no mask — keeping YOLOE mask")

    # 3 — OpenCV A4 sheet detection
    a4_corners, a4_box = detect_a4_sheet(frame)
    if a4_corners is None:
        print(f"  [!] A4 sheet not detected in {os.path.basename(frame_path)}")
    else:
        print(f"  [✓] A4 sheet detected — box {a4_box}")

    # 4 — Save debug visualisation
    vis = draw_detections(frame, product_mask, product_box, product_class, a4_corners, a4_box)
    base_name  = os.path.splitext(os.path.basename(frame_path))[0]
    debug_path = os.path.join(output_dir, f'{base_name}_detections.jpg')
    cv2.imwrite(debug_path, vis)

    # Save product mask as a separate image so later steps can load it directly.
    # When SAM 2 refinement ran, the refined mask takes the canonical filename
    # (downstream steps are unchanged) and the raw YOLOE mask is saved alongside.
    mask_path = None
    if product_mask is not None:
        mask_path = os.path.join(output_dir, f'{base_name}_product_mask.png')
        cv2.imwrite(mask_path, product_mask)
    if yoloe_mask is not None:
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_product_mask_yoloe.png'), yoloe_mask)

    # 5 — Package results for this frame
    # NOTE: paths are stored as absolute so that later pipeline steps (which live in
    # their own sibling folders, e.g. depth_estimation_step/, measurement_extraction_step/)
    # can load these files directly regardless of their own working directory.
    return {
        'frame':        os.path.abspath(frame_path),
        'debug_image':  os.path.abspath(debug_path),
        'mask_source':  mask_source,
        'product': {
            'detected':     product_mask is not None,
            'class':        product_class,
            'confidence':   round(product_conf, 4),
            'box_xyxy':     product_box,
            'mask_path':    os.path.abspath(mask_path) if mask_path else None,
        },
        'a4_sheet': {
            'detected':     a4_corners is not None,
            'box_xyxy':     a4_box,
            # Corner coordinates used downstream for scale anchoring
            'corners_px':   a4_corners.tolist() if a4_corners is not None else None,
        },
        # Pass the refined camera matrix (post-undistortion) to later steps
        'camera_matrix': camera_matrix.tolist(),
    }


# -------------------------------------- Main --------------------------------------

def main():
    
    print("\n=== Step 3 — Instance Segmentation (Product) ===\n")

    if not os.path.isdir(FRAMES_DIR):
        print(f"[!] Frames directory not found: '{FRAMES_DIR}'")
        print("    Create a 'frames/' folder and place your captured product images inside it.")
        return

    # Collect all JPEG/PNG frames, sorted for consistent ordering
    frame_paths = sorted([
        os.path.join(FRAMES_DIR, f)
        for f in os.listdir(FRAMES_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not frame_paths:
        print(f"[!] No images found in '{FRAMES_DIR}'. Add your captured frames and re-run.")
        return

    print(f"Found {len(frame_paths)} frame(s) to process.")
    print(f"Product prompt: '{PRODUCT_PROMPT}'")
    print()

    # Load calibration data produced by camera_calibration.py
    print(f"Loading calibration data from '{CALIB_FILE}'...")
    camera_matrix, dist_coeffs = load_calibration(CALIB_FILE)
    print("Calibration loaded.\n")

    # Load the YOLOE segmentation model (downloads automatically on first run) and
    # prompt it with the product description — after set_classes() the model only
    # looks for that one thing, whatever it is.
    print(f"Loading YOLOE model '{YOLO_MODEL}'...")
    model = YOLOE(YOLO_MODEL)
    model.set_classes([PRODUCT_PROMPT], model.get_text_pe([PRODUCT_PROMPT]))
    print("Model ready.\n")

    # Load SAM 2 for boundary refinement (optional — see SAM2_REFINE above)
    sam_model = None
    if SAM2_REFINE:
        print(f"Loading SAM 2 model '{SAM2_MODEL}' for mask refinement...")
        sam_model = SAM(SAM2_MODEL)
        print("SAM 2 ready.\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process every frame and collect results
    all_results = []
    for i, fp in enumerate(frame_paths):
        print(f"Frame {i+1}/{len(frame_paths)}: {os.path.basename(fp)}")
        result = process_frame(fp, model, camera_matrix, dist_coeffs, OUTPUT_DIR, sam_model)
        if result:
            all_results.append(result)
        print()

    # Write all frame results to a single JSON file — consumed by Steps 5 and 6
    results_path = os.path.join(OUTPUT_DIR, 'segmentation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    product_ok = sum(1 for r in all_results if r['product']['detected'])
    a4_ok      = sum(1 for r in all_results if r['a4_sheet']['detected'])
    total      = len(all_results)

    print("─" * 45)
    print(f"Results saved to : {results_path}")
    print(f"Product detected : {product_ok}/{total} frames")
    print(f"A4 detected      : {a4_ok}/{total} frames")

    if product_ok < total:
        print("\n[!] Product missed in some frames.")
        print("    Try rewording PRODUCT_PROMPT at the top of this file (e.g. 'sneaker'")
        print("    instead of 'shoe'), or put more light on the product.")

    if a4_ok < total:
        print("\n[!] A4 sheet missed in some frames.")
        print("    Check debug images in output/segmentation/ to see what was found.")
        print("    If the sheet is not plain white, adjust the HSV thresholds in detect_a4_sheet().")

    print("\nStep 3 complete. Check output/segmentation/ for labelled debug images.")


if __name__ == '__main__':
    main()
