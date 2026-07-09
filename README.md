# Dimension Extraction Pipeline

Extract real-world **product dimensions (cm) from ordinary RGB photos** — no LiDAR, no depth sensor. An iPhone photo of a product next to an A4 sheet of paper goes in; a structured JSON with validated width/height measurements comes out, ready for 3D asset generation.

**Target accuracy: under 2 cm** on major dimensions.
**Current validated accuracy: 2 of 3 products fully within tolerance** against tape-measure ground truth.

| Product (validated 2026-07-07) | Width err | Height err | Depth err | Verdict |
|---|---|---|---|---|
| Handbag | 0.25 cm | 0.47 cm | 0.76 cm | PASS (97.8%) |
| Nike sneaker | 1.78 cm | 0.33 cm | 0.83 cm | PASS (94.8%) |
| Converse sneaker | 3.28 cm | 1.11 cm | 1.19 cm | FAIL (90.1%) |

*(Mean absolute error vs. tape measure, one capture set per product. The one failure is the known coplanarity trap — the A4 sheet was taped to the wall behind the product instead of standing in its plane; see [Capture Protocol](#2--capture-protocol). The camera was recalibrated on 2026-07-08 — all numbers above predate it and should improve on re-validation.)*

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
│                                    prompt_robustness.py, prompt_robustness_results.md
├── depth_estimation_step/        ③ depth_estimation.py
├── measurement_extraction_step/  ④ measurement_extraction.py, merge_views.py
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

Outputs `output/calibration_data.pkl` (camera matrix + distortion coefficients) plus a worst-first per-image error report (`output/reprojection_errors.txt`) so you can spot and retake the images dragging the RMS up.

Target: RMS reprojection error **< 1.0 px** (lower = better; everything downstream inherits this quality). Our result: **1.59 px RMS from 79 images**. Part of that residual is inherent to phone cameras — the lens geometry changes slightly between shots (optical image stabilisation floats the lens, autofocus shifts the effective focal length, and in-camera processing warps each frame a little differently), so one distortion model can never fit every image perfectly — and part may be our capture conditions (lighting, motion blur, print flatness). We accepted it because at a 0.75 m working distance 1.59 px corresponds to ~0.3 mm on the object, far below the pipeline's other error sources. Also check the fx/fy agreement in the camera matrix (ours: 4370 vs 4369, 0.04%) — a mismatch there is a *systematic* scale error on every height measurement, which no amount of frame averaging can fix.

### 2 — Capture protocol

Accuracy is won or lost here. For each product:

- Camera on a **tripod or stable surface**, **perpendicular** to the product
- **A4 sheet standing in the same plane as the product's front face** — right next to it, **not behind it**. The product's distance is taken from the sheet, so every cm of gap between their planes becomes a proportional error on every dimension (measured: a 7 cm gap → +10% / +0.9 cm on both width and height)
- Plain, high-contrast background; even diffuse lighting, no harsh shadows
- **Minimum 5 frames** — measurements are averaged with outlier rejection
- Front view gives width + height; front-to-back depth needs a separate side-view capture set (merged in step 5 by `merge_views.py`)

Put the frames in `instance_segmentation_step/frames/`.

### 3 — Instance segmentation

```bash
cd instance_segmentation_step
python segmentation.py
```

Detects the product with Grounded-SAM — Grounding DINO finds it from a text prompt, SAM 2 turns the box into a pixel-precise mask — and the A4 sheet with OpenCV (regions that are both *bright and smooth* — plain brightness thresholds fail against bright textured backgrounds), after undistorting each frame with the calibration data.

- Set `PRODUCT_PROMPT` at the top of the script. **Use a simple noun plus at most one attribute** — `'black shoe'`, `'red handbag'`. Measured across the three validated products ([full data](instance_segmentation_step/prompt_robustness_results.md)), this form consistently scored ~0.9 confidence, while stacked adjectives only drained confidence (a 10-word description scored ~0.5) without changing the box at all — prompt wording affects how *reliably* frames detect, not what gets measured.
- If frames are missed, **simplify the prompt before lowering `BOX_THRESHOLD`/`TEXT_THRESHOLD`** — plain wording recovers margin; lowering thresholds moves the cliff toward the noise floor.
- Don't read high confidence as proof the description matched: wrong-color prompts scored 0.86–0.90 on all three test products. Check `output/*_detections.jpg` — product mask tinted red (the tint recolors the product, so don't judge its true color there), A4 quadrilateral in blue. Both should be found in every frame.
- To vet prompts for a new product empirically, use `prompt_robustness.py` (see the results doc for how).

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

For full 3-D dimensions, run steps 3–5 once on a front capture set and once on a side set, then:

```bash
python merge_views.py    # optionally: SUBJECT=<subject_id> python merge_views.py
```

It maps the side view's silhouette width to the product's front-to-back **depth**, and cross-checks the two views' independent height measurements against each other (they must agree within 1.5 cm — a bigger gap means the two capture sets are inconsistent, usually different A4 placement, and the depth number shouldn't be trusted). The merged `output/measurements_<subject_id>.json` is what step 6 validates.

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

- **Re-validate under the new calibration.** The camera was recalibrated on 2026-07-08: RMS 1.59 px from 79 full-frame-coverage images (previously 1.82 px from 14), and the old fx 4513 vs fy 5005 mismatch — a ~10% systematic bias that every height measurement inherited — is resolved (now 4370 vs 4369). All entries in `accuracy_history.jsonl` predate this calibration, so the pipeline should be re-run on the capture sets and re-validated to measure the improvement. The remaining 1.59 px RMS is accepted for now (~0.3 mm at working distance — see step 1); better capture conditions might still push it under the 1.0 px target.
- **One capture set still violates the protocol** — the Converse set has the A4 sheet taped to the wall behind the product, which is why it fails validation (width −3.28 cm). Re-capture with the sheet standing in the product's plane.
- **Validation is one capture set per product** — the brief calls for 3–5 subjects validated together; run them through step 6 as a batch once re-captured.
- **Model scale**: Grounding DINO base + SAM 2.1 base. If mask-edge tightness ever becomes the limiting error, `sam2.1_l.pt` (large) is the first lever; if detection recall is the problem, *simplify* the prompt first — the prompt-robustness pass measured plain noun-plus-one-attribute prompts as the strongest form on every product tested.
- **Prompt attributes are not verified against pixels** — a wrong-color prompt still boxes the salient object with high confidence. Untested: scenes with two similar objects; keep one product per frame per the capture protocol.
