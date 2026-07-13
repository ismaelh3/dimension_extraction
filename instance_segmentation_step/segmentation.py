import cv2
import numpy as np
import os
import json
import pickle
import torch
from PIL import Image
from ultralytics import SAM
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# -------------------------------------- Configuration --------------------------------------

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))  # folder this script lives in, so paths work from any cwd
FRAMES_DIR  = os.path.join(SCRIPT_DIR, 'frames')          # folder containing your captured product frames
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, 'output')          # all results written here (local to this step)
CALIB_FILE  = os.path.join(SCRIPT_DIR, '..', 'camera_calibration_step', 'output', 'calibration_data.pkl')
# Grounded-SAM: Grounding DINO FINDS the product (text-prompted, boxes only),
# then SAM 2 segments whatever the box points at. Grounding DINO fuses a language
# model into the detector, so it recognises far more than YOLO-family detectors —
# full descriptions work as prompts, not just category labels. SAM 2 provides the
# pixel-precise boundary that the measurement step reads dimensions from.
DINO_MODEL = 'IDEA-Research/grounding-dino-base'       # downloads from HuggingFace on first run (~700 MB)
SAM2_MODEL = os.path.join(SCRIPT_DIR, 'sam2.1_b.pt')   # downloads automatically on first run

# What to detect. Grounding DINO understands full descriptions, not just category
# labels — 'black cylindrical thermos' or 'cardboard shipping box' both work.
# Change this per product you measure. (The model expects lowercase text ending
# in a period; normalise_prompt() applies that automatically.)
# Defaul Placeholder: 'INSERT_PRODUCT_NAME.'
PRODUCT_PROMPT = 'glass snowglobe.'

# Detection thresholds. BOX: minimum confidence for a detection to count at all.
# TEXT: how strongly the detection must match the words of the prompt.
# Lower them if the product isn't being found; raise them if the wrong thing is.
BOX_THRESHOLD  = 0.35
TEXT_THRESHOLD = 0.25

# Where to run Grounding DINO: Apple-Silicon GPU ('mps') when available, CPU
# otherwise. If MPS ever throws an unsupported-operation error, hardcode 'cpu'.
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Reference sheet: US Letter, 8.5 × 11 in (215.9 × 279.4 mm, portrait) — tape-measured
# 2026-07-09. Every "A4" name in this pipeline refers to this sheet; the original A4
# assumption (210 × 297 mm) silently biased every measurement by −2.7%. If you switch
# to a real A4 sheet, also update A4_REAL_WIDTH_M in depth_estimation.py and the
# aspect check in measurement_extraction.py.
A4_ASPECT_RATIO     = 279.4 / 215.9   # ideal portrait aspect ratio ≈ 1.294
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
    """
    Read the camera matrix, distortion coefficients, and the image resolution the
    matrix is valid at, saved by camera_calibration.py.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    calib_size_wh = data.get('image_size_wh')
    if calib_size_wh is None:
        print("[!] Calibration file has no 'image_size_wh' — re-run camera_calibration.py.")
        print("    Without it, frames at a different resolution than the calibration shots")
        print("    get a mis-scaled camera matrix (silently wrong distances). Assuming every")
        print("    frame matches the calibration resolution.")
    return data['camera_matrix'], data['distortion_coefficients'], calib_size_wh


def scale_matrix_to_frame(camera_matrix, calib_size_wh, frame_size_wh):
    """
    Rescale the calibration matrix to a frame captured at a different resolution.

    fx/fy/cx/cy are in PIXELS of the calibration images, so a frame the camera
    saved at another resolution (iPhones silently switch 48MP/12MP) needs them
    multiplied by the resolution ratio. Without this, the pinhole distance
    estimate is off by exactly that ratio and undistortion warps the image
    (masks/quads land hundreds of px from where they belong at 12MP).
    Distortion coefficients are normalized and need no scaling.
    """
    if calib_size_wh is None or tuple(frame_size_wh) == tuple(calib_size_wh):
        return camera_matrix
    sx = frame_size_wh[0] / calib_size_wh[0]
    sy = frame_size_wh[1] / calib_size_wh[1]
    if abs(sx - sy) > 0.01 * sx:
        print(f"  [!] Frame aspect ratio differs from calibration "
              f"({frame_size_wh} vs {calib_size_wh}) — matrix scaling is approximate.")
    scaled = camera_matrix.copy()
    scaled[0, 0] *= sx  # fx
    scaled[0, 2] *= sx  # cx
    scaled[1, 1] *= sy  # fy
    scaled[1, 2] *= sy  # cy
    print(f"  [i] Frame is {frame_size_wh[0]}x{frame_size_wh[1]} but calibration is "
          f"{calib_size_wh[0]}x{calib_size_wh[1]} — camera matrix rescaled x{sx:.3f}.")
    return scaled


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


# -------------------------------------- Product detection (Grounding DINO) --------------------------------------

def normalise_prompt(prompt):
    """
    Grounding DINO's training data used prompts that are lowercase and end with a
    period — matching that format measurably improves detection quality.
    """
    prompt = prompt.strip().lower()
    return prompt if prompt.endswith('.') else prompt + '.'


def detect_product(frame, processor, model, prompt):
    """
    Run Grounding DINO on the frame and return the bounding box for the target product.

    Grounding DINO is a text-prompted open-vocabulary detector: everything it returns
    already matches the given prompt, so no class filtering is needed. If several
    instances match, the highest-scoring one is kept. It produces boxes only, no
    masks — SAM 2 turns the winning box into a pixel-precise mask right after
    (see segment_with_sam2).

    Returns bounding box [x1,y1,x2,y2], the matched text label, and confidence
    score — or (None, None, 0.0) when nothing matched the prompt.
    """
    # cv2 loads images as BGR; the HuggingFace processor expects RGB
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(
        images=Image.fromarray(rgb),
        text=normalise_prompt(prompt),
        return_tensors='pt',
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    h, w = frame.shape[:2]
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(h, w)],
    )[0]

    if len(results['scores']) == 0:
        return None, None, 0.0

    best   = int(results['scores'].argmax())
    box    = results['boxes'][best].cpu().numpy().astype(int).tolist()
    labels = results.get('text_labels', results.get('labels'))
    return box, str(labels[best]), float(results['scores'][best])


# -------------------------------------- SAM 2 segmentation --------------------------------------

def segment_with_sam2(frame, box, sam_model):
    """
    Segment the product with SAM 2, prompted by Grounding DINO's bounding box.

    SAM 2 is class-agnostic — it doesn't know what a 'bottle' is, it just
    segments whatever object the box points at, with excellent boundary quality.
    The box prompt is what ties it to the product Grounding DINO identified.
    This is the classic Grounded-SAM pairing: DINO knows WHAT, SAM knows WHERE
    the edges are.

    Returns a binary mask at frame resolution, or None if SAM 2 returned nothing
    (the frame is then treated as having no product and skipped downstream).
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
      5. Gate candidates by aspect ratio, then choose the LARGEST one.

    Aspect ratio is a gate, not a ranking: the detected bbox of the real sheet
    reads 1.36-1.47 depending on tilt and how far the bright-smooth region bleeds
    into the wall/stand, while small background patches (bare wall between
    furniture) can land closer to the ideal ratio by pure luck. Ranking by
    closest-aspect made the winner flip frame to frame; the sheet is reliably
    the most PROMINENT paper-like quad in a capture that follows the protocol,
    so area decides among survivors.

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

    best_candidate = None
    best_box       = None
    best_area      = 0.0

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

        # Keep the largest candidate that survived every gate
        if area > best_area:
            best_area      = area
            best_candidate = approx.reshape(4, 2)
            best_box       = [x, y, x + bw, y + bh]

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

def process_frame(frame_path, dino_processor, dino_model, sam_model, camera_matrix, dist_coeffs, calib_size_wh, output_dir):
    """
    Full Step 3 pipeline for one frame:
      1. Load and undistort the image.
      2. Detect the product with Grounding DINO (text-prompted box).
      3. Segment it with SAM 2 (box-prompted mask).
      4. Detect the A4 sheet with OpenCV.
      5. Save a labelled debug image.
      6. Return a result dict for this frame.
    """
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"  [!] Could not read: {frame_path}")
        return None

    # 1 — Remove lens distortion using calibration data (matrix rescaled first if
    #     this frame's resolution differs from the calibration shots)
    camera_matrix = scale_matrix_to_frame(camera_matrix, calib_size_wh, frame.shape[1::-1])
    frame, camera_matrix = undistort_frame(frame, camera_matrix, dist_coeffs)

    # 2 — Grounding DINO product detection (box only)
    product_box, product_class, product_conf = detect_product(frame, dino_processor, dino_model, PRODUCT_PROMPT)
    product_mask = None
    mask_source  = 'grounding-dino-base + sam2.1_b'
    if product_box is None:
        print(f"  [!] No product matched '{PRODUCT_PROMPT}' in {os.path.basename(frame_path)}")
    else:
        print(f"  [✓] Product detected  — matched '{product_class}', confidence {product_conf:.2f}, box {product_box}")

        # 3 — SAM 2 turns the box into a pixel-precise mask
        product_mask = segment_with_sam2(frame, product_box, sam_model)
        if product_mask is None:
            print(f"  [!] SAM 2 returned no mask — frame will be skipped downstream")
        else:
            print(f"  [✓] Mask extracted with SAM 2")

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
    mask_path = None
    if product_mask is not None:
        mask_path = os.path.join(output_dir, f'{base_name}_product_mask.png')
        cv2.imwrite(mask_path, product_mask)

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
    
    print("\n======== Step 3 — Instance Segmentation (Product) =========\n")

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
    camera_matrix, dist_coeffs, calib_size_wh = load_calibration(CALIB_FILE)
    print("Calibration loaded.\n")

    # Load Grounding DINO (detector). The prompt itself is passed per-frame in
    # detect_product() — no per-class setup needed, that's the point of a
    # language-grounded detector.
    print(f"Loading Grounding DINO '{DINO_MODEL}' (device: {DEVICE})...")
    dino_processor = AutoProcessor.from_pretrained(DINO_MODEL)
    dino_model     = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL).to(DEVICE).eval()
    print("Grounding DINO ready.\n")

    # Load SAM 2 (mask producer — mandatory, since Grounding DINO gives boxes only)
    print(f"Loading SAM 2 model '{SAM2_MODEL}'...")
    sam_model = SAM(SAM2_MODEL)
    print("SAM 2 ready.\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process every frame and collect results
    all_results = []
    for i, fp in enumerate(frame_paths):
        print(f"Frame {i+1}/{len(frame_paths)}: {os.path.basename(fp)}")
        result = process_frame(fp, dino_processor, dino_model, sam_model, camera_matrix, dist_coeffs, calib_size_wh, OUTPUT_DIR)
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
        print("    Try a more descriptive PRODUCT_PROMPT at the top of this file (Grounding")
        print("    DINO handles full descriptions, e.g. 'black cylindrical thermos'), or")
        print("    lower BOX_THRESHOLD / TEXT_THRESHOLD slightly (e.g. 0.30 / 0.20).")

    if a4_ok < total:
        print("\n[!] A4 sheet missed in some frames.")
        print("    Check debug images in output/segmentation/ to see what was found.")
        print("    If the sheet is not plain white, adjust the HSV thresholds in detect_a4_sheet().")

    print("\nStep 3 complete. Check output/segmentation/ for labelled debug images.")


if __name__ == '__main__':
    main()
