# Dimension Extraction Pipeline — Technical Brief

> **Role:** Stage 2 — Dimension Extraction
> **Input:** Captured product photos
> **Output:** Structured JSON dimensional profile per product → handed to Stage 3 (asset generation)
> **Hardware:** iPhone 17 (no LiDAR) + MacBook Pro M1 8GB RAM

---

## Goal

Extract accurate real-world **product** dimensions from standard RGB camera input, without a depth sensor, precise enough for a colleague to generate fitting 3D assets. Target error tolerance is **under 2cm** on major product dimensions.

This pipeline measures inanimate products (boxes, bottles, furniture, etc.) — not people. There is no pose/skeleton step; a product's own bounding box (from its segmentation mask) is what gets measured.

---

## Software Stack (all free)

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language (MacBook) |
| YOLOv8 (Ultralytics) | Product + reference-object instance segmentation |
| Depth Anything V2 | Monocular depth estimation |
| OpenCV | Camera calibration, image processing, contour-based reference detection |
| NumPy | Measurement calculations, averaging, and accuracy statistics |
| Open3D | Optional — 3D point cloud visualisation during dev |

### Install commands
```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install open3d
pip install transformers torch torchvision  # for Depth Anything V2
```

---

## Pipeline Overview

```
Capture (iPhone 17)
       ↓
Camera Calibration (one-time, OpenCV)                    [camera_calibration_step/]
       ↓
Instance Segmentation — Product + A4 Reference (YOLOv8 + OpenCV)   [instance_segmentation_step/]
       ↓
Monocular Depth Estimation + Scale Anchoring (Depth Anything V2)    [depth_estimation_step/]
       ↓
Measurement Extraction + Multi-frame Averaging                     [measurement_extraction_step/]
       ↓
JSON Output → Stage 3 (Asset Generation)
       ↓
Accuracy Validation — vs. tape-measure ground truth                [accuracy_validation_step/]
```

---

## Step-by-Step Implementation

### Step 1 — Camera Calibration (one-time setup)

Calibrate the iPhone 17 camera to extract intrinsic parameters (focal length, principal point). These convert pixel distances to real-world distances and must be done before any measurement work.

**How:**
- Print the checkerboard pattern (`camera_calibration_step/checkerboard_9x7_2.5cm.pdf`, 2.5cm squares)
- Capture 20–30 photos of it at varied angles — either with `capture-images.py` (webcam) or directly on the iPhone, saved into `calibration_images/`
- Run `camera_calibration.py` to extract the camera matrix and distortion coefficients
- Results are saved to `output/calibration_data.pkl` (used by later steps) and human-readable `.txt` copies

**Target reprojection error:** under 1.0 (lower is better)

**Known limitation:** the current calibration photos don't cover the frame edges/corners well, so `instance_segmentation_step/segmentation.py` deliberately skips ROI-cropping after undistortion (see comment in `undistort_frame`) to avoid losing real scene content. Re-shooting calibration photos that cover the full frame, including corners, would tighten this up.

---

### Step 2 — Controlled Capture Protocol

Accuracy is won or lost at capture time. Follow this protocol strictly:

- Product placed **1.5–2 metres** from camera
- Camera on tripod or stable surface — **no handheld**
- Camera **perpendicular** to the product — not angled
- **Plain, high-contrast background**
- **A4 sheet** (210mm × 297mm) standing **in the same plane as the product's front face** — right next to it, NOT behind it — visible in every frame. The product's camera distance is taken directly from the sheet, so any gap between their planes becomes a proportional error on every dimension (ground-truth tested: a ~7cm plane gap produced a ~10% / +0.9cm overestimate on both width and height)
- Even, diffuse lighting — no harsh shadows
- Capture **minimum 5 frames** per product — measurements are averaged across frames, with outliers rejected
- For a full 3D profile, capture a **front set and a separate side set** — the current pipeline measures width/height from a front view only; depth (front-to-back) requires the side view (see Step 5)

---

### Step 3 — Instance Segmentation (Product + Reference Object)

`instance_segmentation_step/segmentation.py` runs two independent detectors per frame:

**Product detection (YOLOv8):**
- If you know the product's YOLO class name (e.g. `'bottle'`, `'chair'`, `'laptop'`), set `PRODUCT_CLASS` at the top of the file to filter to it directly.
- Otherwise, leave `PRODUCT_CLASS = None` — auto mode picks the highest-confidence detection that isn't `'person'`, so unlabelled or generic products still get picked up.
- The segmentation mask (not just the bounding box) is saved per frame, since Step 5 uses the mask's tight bounding box for more accurate edges than YOLO's raw box.

**A4 reference sheet detection (OpenCV, no ML):**
- A plain brightness threshold isn't reliable — a textured, similarly-bright background (wallpaper, wood) can merge into one giant contour with the sheet.
- `detect_a4_sheet()` instead looks for regions that are both **bright** and **smooth** (low local pixel variance), since paper is smooth but most backgrounds that are equally bright are also textured.
- Candidate contours are then filtered to 4-sided polygons whose aspect ratio is close to A4's 1.414.

Frames are undistorted using the Step 1 calibration data before either detector runs. Results (masks, boxes, confidence, A4 corners) are written to `output/segmentation_results.json` for the next step.

---

### Step 4 — Monocular Depth Estimation + Scale Anchoring

`depth_estimation_step/depth_estimation.py` runs Depth Anything V2 to get a *relative* depth map per frame, then anchors it to real-world scale using the A4 sheet detected in Step 3.

**How the anchoring works:**
1. A4's real width is known: 210mm.
2. Its pixel width in the frame comes from Step 3's bounding box.
3. Pinhole camera model: `pixel_width = (real_width_m × focal_length_px) / distance_m` → solve for `distance_m`.
4. Sample the raw model output inside the A4 box and take the median. The model outputs **disparity** (higher = closer), so distance is proportional to `1/value` — the raw `predicted_depth` tensor is used, unnormalised, because any shift breaks that inverse mapping.
5. `k = real_distance_m × median_disparity` — every pixel's metric depth is then `depth_m = k / disparity`.

**Trust limits (ground-truth tested):** the model's output is affine-invariant, meaning it has an unknown offset that a single reference object cannot solve for. The anchored map is reliable for depth *ordering* and relative structure, but its absolute distances can be badly biased (it placed a bottle at 0.45m that a tape measure put at 0.673m). Step 5 therefore takes the product's distance from the A4 pinhole geometry, not from this map.

Frames where the A4 sheet or product wasn't detected in Step 3 are skipped (scale can't be anchored without the reference object). Metric depth maps and colourised visualisations are saved to `output/`, consumed by Step 5.

---

### Step 5 — Measurement Extraction + Multi-Frame Averaging

`measurement_extraction_step/measurement_extraction.py`:

1. Loads the metric depth map and the product's segmentation mask for each frame.
2. Takes the **tight bounding box of the mask** (`get_mask_tight_bbox`) rather than YOLO's raw box — this excludes padding/background YOLO's box might include.
3. Takes the product's camera distance **from the A4 sheet's pinhole-derived distance** (hence the coplanar requirement in Step 2), and projects the box's edge midpoints to 3D at that single shared depth via `pixel_to_world`. Two things are deliberately *not* used: per-edge-pixel depth-map values (the upscaled depth map smears across the silhouette boundary — an edge pixel reading the background's depth once turned a 9cm bottle into an "81cm" one), and the depth map's absolute distances in general (see Step 4's trust limits). The mask-interior median depth is still printed as a diagnostic — a large gap vs. the A4 distance signals a non-coplanar A4 or a confused depth model.
4. Repeats across all captured frames, then applies `robust_average()`: discards any per-frame measurement more than 1 standard deviation from the mean, and reports the mean ± std of what remains.

**Current limitation:** only width and height are measured (a single front-facing view can't recover depth/front-to-back thickness). The output JSON's `depth` field is `null` with a note — run a second, side-view capture set and merge the two JSONs for a full 3D profile.

---

### Step 6 — JSON Output

`measurement_extraction.py` writes one file per product to `measurement_extraction_step/output/measurements_<subject_id>.json`:

```json
{
  "subject_id": "product_001",
  "captured_at": "2026-07-01T12:00:00",
  "frame_count": 5,
  "measurements_cm": { "width": 12.5, "height": 30.1, "depth": null },
  "error_estimates_cm": { "width": 0.14, "height": 0.22 },
  "reference_object": "A4_sheet_210x297mm",
  "model_versions": { "segmentation": "yolov8n-seg", "depth_estimation": "Depth-Anything-V2-Small" },
  "notes": "Depth (front-to-back) dimension not measured — requires a separate side-view capture."
}
```

`error_estimates_cm` is **precision** (frame-to-frame agreement), not accuracy — see Step 7 for the distinction and how it's checked.

---

### Step 7 — Accuracy Validation

Precision and confidence scores tell you the pipeline is *consistent* with itself; they don't tell you it's *correct*. Accuracy can only be established by comparing pipeline output against real, physically-measured ground truth — that's what this step does.

`accuracy_validation_step/accuracy_validation.py`:

1. **Ground truth** — tape-measure 3–5 real products by hand and record their true dimensions in `ground_truth.json`, keyed by the same `subject_id` used in Step 5/6. (An example entry and a `_README` key ship in the file — delete both once you've added real ones.)
2. **Compare** — for every product with both a ground-truth entry and a pipeline JSON, computes per dimension: signed error, absolute error, % error, and whether it's within the 2cm tolerance.
3. **Flag overconfidence** — separately checks whether the pipeline's *reported* `error_estimates_cm` (precision) was small while the *actual* deviation from truth (accuracy) was large. That combination is the signature of a **systematic bias** — e.g. a calibration or scale-anchoring problem — as opposed to random per-frame noise, and it would be invisible if you only looked at `error_estimates_cm`.
4. **Aggregate** — rolls all validated products up into per-dimension mean absolute error, max error, mean *signed* bias (systematic over/under-measurement — random error would average back towards zero, bias won't), and pass rate.
5. **Track over time** — every run appends a summary line to `output/accuracy_history.jsonl`, so you can see whether a calibration or protocol change actually improved accuracy, rather than re-validating from scratch each time.
6. Warns if fewer than 3 products have been validated — a single match could be a fluke either way.

Run this any time you change calibration, the capture protocol, or the detection thresholds — not just once before "production."

---

## Key Accuracy Levers (in order of impact)

1. **Camera calibration quality** — low reprojection error = everything downstream is more accurate
2. **Reference object reliability** — if A4 detection is flaky, scale anchoring breaks
3. **Frame averaging** — more frames = lower random error (but won't fix systematic bias — see Step 7)
4. **Capture protocol discipline** — fixed distance, perpendicular angle, good lighting
5. **Product mask tightness** — a loose/noisy segmentation mask shifts the bounding-box corners used for measurement

---

## What to Read / Look Into Next

- `cv2.calibrateCamera()` — OpenCV docs
- Depth Anything V2 GitHub — `https://github.com/DepthAnything/Depth-Anything-V2`
- `cv2.findHomography()` — for more robust reference object scale extraction
- Perspective-n-Point (PnP) — next-level technique for recovering 3D positions from 2D points: `cv2.solvePnP()`
- `cv2.findNonZero()` / `cv2.boundingRect()` — used for tight mask-based bounding boxes in Step 5
- Apple Vision framework — if processing moves to iPhone directly later

---

## Handoff to Stage 3

Deliver per-product JSON files (Step 6) to your colleague for asset generation, including the `error_estimates_cm` block so they know per-frame measurement consistency. Also share the latest `accuracy_validation_step/output/accuracy_report.json` (or at least its `overall_pass` / `summary_by_dimension`) so they know the pipeline's *actual*, ground-truth-checked accuracy — not just its internal precision — when deciding fitting tolerances for generated assets.
