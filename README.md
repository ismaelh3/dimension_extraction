# Dimension Extraction Pipeline

Extract real-world **product dimensions (cm) from ordinary RGB photos** — no LiDAR, no depth sensor. An iPhone photo of a product next to an A4 sheet of paper goes in; a structured JSON with validated width/height measurements comes out, ready for 3D asset generation.

**Target accuracy: under 2 cm** on major dimensions.
**Current validated accuracy: ±0.9 cm** against tape-measure ground truth. ✅

| | Pipeline | Tape measure | Error |
|---|---|---|---|
| Width | 10.1 ± 0.39 cm | 9.19 cm | +0.91 cm |
| Height | 26.6 ± 0.35 cm | 25.73 cm | +0.87 cm |

*(7-frame capture of a test bottle; the remaining +0.9 cm is a known systematic bias from the A4 sheet standing ~7 cm behind the product — see [Capture Protocol](#2--capture-protocol).)*

---

## How it works

```
iPhone photos (product + A4 sheet in frame)
        │
        ▼
① Camera Calibration ─────────── one-time: checkerboard photos → focal length,
        │                        distortion coefficients (OpenCV)
        ▼
② Instance Segmentation ──────── Grounded-SAM: Grounding DINO finds the product
        │                        from a text prompt, SAM 2 masks it; OpenCV
        │                        contour analysis finds the A4 sheet's corners
        ▼
③ Depth Estimation ───────────── Depth Anything V2 gives relative depth;
        │                        the A4 sheet anchors it to metres
        ▼
④ Measurement Extraction ─────── pinhole projection at the A4-derived
        │                        distance → width/height in cm, averaged
        │                        across frames with outlier rejection
        ▼
⑤ Accuracy Validation ────────── compares output vs. tape-measure ground
                                 truth; tracks accuracy across runs
```

The core idea: the A4 sheet (210 × 297 mm, a free, universally available object of exactly known size) appears in every photo. Its pixel width plus the calibrated focal length give the exact camera-to-sheet distance via the pinhole camera model. Everything else is measured relative to that anchor.

### Repository layout

```
Dimension Extraction/
├── camera_calibration_step/      ① capture-images.py, camera_calibration.py,
│                                    checkerboard_9x7_2.5cm.pdf
├── instance_segmentation_step/   ② segmentation.py  (+ frames/ — your photos go here)
├── depth_estimation_step/        ③ depth_estimation.py
├── measurement_extraction_step/  ④ measurement_extraction.py
├── accuracy_validation_step/     ⑤ accuracy_validation.py, ground_truth.json
├── dimension_extraction_brief.md    full technical brief
└── requirements.txt
```

Each step writes its results to its own `output/` folder; the next step reads them from there. **Run every script from inside its own folder** — the cross-step paths are relative.

---

## Setup

Python 3.10+ (developed on 3.13, macOS/Apple Silicon — an M1 with 8 GB RAM is enough).

```bash
cd "Dimension Extraction"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Model weights (Grounding DINO base ~700 MB, SAM 2.1 base ~160 MB, Depth Anything V2 Small ~100 MB) download automatically on first run.

---

## Running the pipeline

### 1 — Camera calibration (one-time per camera)

Print `camera_calibration_step/checkerboard_9x7_2.5cm.pdf` **at 100% scale — never "fit to page"** — and verify with a ruler that the squares are exactly 2.5 cm. (PDF units are points, 1/72 inch; silent printer rescaling here becomes a silent scale error on every measurement downstream.)

Photograph the checkerboard 20–30 times at varied angles and distances, **making sure some shots push it to the frame edges and corners** — the distortion model is only constrained where it has seen data. Put the photos in `camera_calibration_step/calibration_images/`, then:

```bash
cd camera_calibration_step
python camera_calibration.py
```

Outputs `output/calibration_data.pkl` (camera matrix + distortion coefficients) plus a worst-first per-image error report (`output/reprojection_errors.txt`) so you can spot and retake the images dragging the RMS up. Target: RMS reprojection error **< 1.0 px** (lower = better; everything downstream inherits this quality).

### 2 — Capture protocol

Accuracy is won or lost here. For each product:

- Camera on a **tripod or stable surface**, **perpendicular** to the product
- **A4 sheet standing in the same plane as the product's front face** — right next to it, **not behind it**. The product's distance is taken from the sheet, so every cm of gap between their planes becomes a proportional error on every dimension (measured: a 7 cm gap → +10% / +0.9 cm on both width and height)
- Plain, high-contrast background; even diffuse lighting, no harsh shadows
- **Minimum 5 frames** — measurements are averaged with outlier rejection
- Front view gives width + height; front-to-back depth needs a separate side-view capture set

Put the frames in `instance_segmentation_step/frames/`.

### 3 — Instance segmentation

```bash
cd instance_segmentation_step
python segmentation.py
```

Detects the product with Grounded-SAM — Grounding DINO finds it from a text prompt, SAM 2 turns the box into a pixel-precise mask — and the A4 sheet with OpenCV (regions that are both *bright and smooth* — plain brightness thresholds fail against bright textured backgrounds), after undistorting each frame with the calibration data.

- Set `PRODUCT_PROMPT` at the top of the script to describe the product. Grounding DINO understands full descriptions, not just category labels — `'black cylindrical thermos'` works as well as `'bottle'`. If the product isn't found, try a more specific description or lower `BOX_THRESHOLD`/`TEXT_THRESHOLD` slightly.
- Check `output/*_detections.jpg` — product mask tinted red, A4 quadrilateral in blue. Both should be found in every frame.

### 4 — Depth estimation + scale anchoring

```bash
cd depth_estimation_step
python depth_estimation.py
```

Runs Depth Anything V2 per frame and anchors its output to metres using the A4 sheet. Sanity checks:

- **Estimated distance to A4** should match your real setup distance and be nearly identical across frames (ours: 0.717–0.749 m).
- `output/*_depth_vis.jpg`: bright = close. Look for uniform colour across each flat surface, and a sharp, clean silhouette where the product meets the background — no speckle, no bleeding.

### 5 — Measurement extraction

```bash
cd measurement_extraction_step
python measurement_extraction.py    # set SUBJECT_ID at the top per product
```

Projects the product mask's tight bounding box through the pinhole model at the A4-derived distance; averages across frames rejecting outliers beyond 1σ. Output: `output/measurements_<subject_id>.json` with dimensions, per-frame precision (±), capture metadata (resolution, mm-per-pixel), and model versions.

The per-frame log prints the depth-map median next to the A4 distance as a diagnostic — a large gap between them means the A4 wasn't coplanar with the product (or the depth model misread the scene).

### 6 — Accuracy validation

Tape-measure the real product and record it in `accuracy_validation_step/ground_truth.json`, keyed by the same `subject_id`:

```json
{ "product_001": { "width": 9.19, "height": 25.73, "depth": null } }
```

```bash
cd accuracy_validation_step
python accuracy_validation.py
```

Reports signed/absolute/% error per dimension against the 2 cm tolerance, and separately flags **overconfidence**: a small reported ± (frames agreeing with each other) combined with a large deviation from truth is the fingerprint of a *systematic* bias — a calibration or scale-anchoring problem that frame averaging can never fix. Every run appends to `output/accuracy_history.jsonl`, so you can verify a calibration or protocol change actually improved accuracy. Validate 3–5 products before trusting the numbers; re-run after any calibration/protocol/threshold change.

---

## Hard-won lessons (read before modifying)

These cost real debugging time; the code comments reference them.

1. **Depth Anything V2 outputs disparity, not depth** — higher value = *closer*, and distance ∝ 1/value. The original code assumed a linear relationship, which is only correct at the anchor point itself. Conversion must be `depth = k / disparity`.
2. **Use `output['predicted_depth']`, not `output['depth']`** from the HuggingFace pipeline. The latter is a visualisation image quantised to 256 grey levels — it silently destroys measurement precision.
3. **Never min-max normalise the disparity map.** Subtracting the frame minimum shifts every value and breaks the 1/value mapping.
4. **The depth map's absolute distances cannot be trusted, even anchored.** The model's output is affine-invariant (unknown offset that one reference object can't solve for): it placed a bottle at 0.45 m that a tape measure put at 0.673 m — a 33% error that scaled every dimension down proportionally. It's reliable for depth *ordering* only. The product's distance therefore comes from A4 pinhole geometry, which the tape measure validated.
5. **Never read depth at a mask's edge pixels.** The depth map is predicted at ~518 px and upscaled ~8× to the full frame, smearing object boundaries — an edge pixel can read the *background's* depth. Trusting per-edge-pixel depth once turned a 9 cm bottle into an "81 cm" one, with deceptively tight ±1.3 cm frame agreement. (That combination — precise but wrong — is exactly what Step 6's overconfidence flag catches.)
6. **EXIF DPI (72 px/inch) means nothing for measurement.** Real-world scale comes from calibration + the A4 anchor, never from image DPI metadata. The 1/72-inch point unit only matters when *printing* the calibration checkerboard at true scale.
7. **Precision ≠ accuracy.** Frames agreeing to ±0.4 cm tells you the error is repeatable, not that it's small. Only tape-measure ground truth (Step 6) establishes accuracy.

## Known limitations / next steps

- **Recalibration pending**: current calibration is RMS 1.82 px from 14 images, with a suspicious fx (4513) vs fy (5005) mismatch (~10%, abnormal for a phone camera) that height measurements inherit via fy. Retake 20–30 checkerboard shots covering the full frame including corners. Because the calibration photos missed the frame edges, `segmentation.py` also deliberately skips ROI-cropping after undistortion.
- **Only 1 ground-truth subject validated** — provisional; the brief calls for 3–5.
- **Front-to-back depth not measured** — the output JSON's `depth` is `null`; requires a side-view capture set merged with the front-view JSON.
- **Model scale**: Grounding DINO base + SAM 2.1 base. If mask-edge tightness ever becomes the limiting error, `sam2.1_l.pt` (large) is the first lever; if detection recall is the problem, richer prompt wording usually beats a bigger model.
