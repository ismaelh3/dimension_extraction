# Prompt-Robustness Results — Grounding DINO

**Run 2026-07-08** · `prompt_robustness.py` · 3 products, front-view capture sets (12 + 14 + 11 frames), 7 prompt variants each, full production chain (undistort → Grounding DINO → SAM 2), thresholds `BOX_THRESHOLD = 0.35`, `TEXT_THRESHOLD = 0.25`. Raw per-frame data: `prompt_experiments/results_<product>.json` (git-ignored, regenerate with the script).

## TL;DR

**Prompt wording never changed the measurement — it only changed how reliably the product gets detected.** Across all 259 prompt×frame runs, every successful detection produced a pixel-identical final box (IoU 1.0, size deviation ≤ 0.05%). But confidence varied by more than 2× between phrasings, and the worst prompt actually lost 2 of 11 frames. **Best form on all three products: a simple noun plus one attribute** — `black shoe`, `red handbag`, `white shoe` — each scoring ~0.9.

---

## How to read the numbers

Each metric below is computed per prompt, across every frame in the product's capture set.

**det — detection rate.**
In how many frames the prompt found the product at all. Every missed frame is silently dropped from the pipeline: with a 12-frame set, a prompt that detects 10/12 means the final measurement averages 10 samples instead of 12 — and nothing warns you unless you read the step-3 summary line.

**conf / min — mean and worst-frame confidence.**
Grounding DINO's own 0–1 score for "this box matches your prompt". The *mean* tells you how well the prompt matches in general; the *min* is the frame where it struggled most. The min is the number that predicts future failures: your next capture set will have frames at least this weak.

**margin — worst-frame confidence minus the 0.35 detection threshold.**
The most operationally important number. It answers: *how much worse can conditions get (dimmer light, more shadow, slightly different angle) before the pipeline starts silently losing frames?* A margin of 0.5 means the prompt could lose half its confidence and still work. A margin of 0.05 means the very next capture set is a coin flip. Rule of thumb: **treat margin < 0.15 as a prompt that needs rewording.**

**IoU — overlap between this prompt's final box and the consensus box.**
Consensus = the median box that the five sensible prompts agree on, per frame. IoU (intersection-over-union) is 1.0 when the boxes are identical and drops toward 0 if the prompt boxed something else entirely (e.g. the A4 sheet instead of the shoe). This is the metric that would catch a prompt *measuring the wrong object*.

**|Δw|% / |Δh|% — width/height difference vs. the consensus box.**
Because the pipeline converts box pixels straight into centimeters, these percentages read directly as measurement error: a prompt whose box is 1% wider would report a 30 cm shoe as 30.3 cm. Values here were ≤ 0.05% — under 0.2 mm on any of these products, i.e. nothing.

---

## Results

### Converse high-top (black, white laces) — 12 frames

| variant | prompt | det | conf | min | margin | IoU | \|Δw\|% | \|Δh\|% |
|---|---|---|---|---|---|---|---|---|
| color + label | `black shoe` | 12/12 | **0.916** | 0.910 | **0.56** | 1.0 | 0.0 | 0.0 |
| short label | `shoe` | 12/12 | 0.902 | 0.894 | 0.54 | 1.0 | 0.0 | 0.0 |
| category | `sneaker` | 12/12 | 0.876 | 0.869 | 0.52 | 1.0 | 0.0 | 0.0 |
| wrong color ⚠ | `red shoe` | 12/12 | 0.860 | 0.850 | 0.50 | 1.0 | 0.0 | 0.0 |
| full description | `black canvas high-top sneaker` | 12/12 | 0.853 | 0.846 | 0.50 | 1.0 | 0.01 | 0.0 |
| over-specified | `worn black converse all-star high-top sneaker with white laces` | 12/12 | 0.531 | 0.519 | 0.17 | 1.0 | 0.01 | 0.0 |
| placeholder ⚠ | `insert prompt` | 12/12 | 0.413 | 0.399 | **0.05** | 1.0 | 0.0 | 0.0 |

**Implications.** Every prompt — including the two controls — boxed the shoe identically, so for this product the prompt only sets the safety margin. `black shoe` gives the most headroom. The 10-word description cost 42% of the confidence and bought nothing. The `insert prompt` placeholder (accidentally used in a real run once) survived on a 0.05 margin — pure luck, as the Nike set below demonstrates.

### Handbag (red glossy croc-embossed leather) — 14 frames

| variant | prompt | det | conf | min | margin | IoU | \|Δw\|% | \|Δh\|% |
|---|---|---|---|---|---|---|---|---|
| color + label | `red handbag` | 14/14 | **0.907** | 0.900 | **0.55** | 1.0 | 0.02 | 0.0 |
| wrong color ⚠ | `blue handbag` | 14/14 | 0.867 | 0.862 | 0.51 | 1.0 | 0.0 | 0.0 |
| short label | `bag` | 14/14 | 0.834 | 0.830 | 0.48 | 1.0 | 0.0 | 0.0 |
| full description | `red glossy leather handbag` | 14/14 | 0.804 | 0.777 | 0.43 | 1.0 | 0.0 | 0.0 |
| category | `handbag` | 14/14 | 0.769 | 0.762 | 0.41 | 1.0 | 0.0 | 0.0 |
| over-specified | `shiny dark red crocodile-embossed leather handbag with two shoulder straps` | 14/14 | 0.511 | 0.506 | 0.16 | 1.0 | 0.01 | 0.0 |
| placeholder ⚠ | `insert prompt` | 14/14 | 0.441 | 0.423 | 0.07 | 1.0 | 0.01 | 0.0 |

**Implications.** Same shape as the Converse — with one twist: `blue handbag`, a color the bag is not, out-scored four of the honest prompts. Color words boost the detector's attention on the object, but the color is **not verified** against the pixels. Confidence measures "how strongly the prompt locked onto *an* object", not "how accurate your description was". Note also that the plain `bag` beat the more formal `handbag` — commoner words ground better.

### Nike running shoe (white/light-gray mesh) — 11 frames

| variant | prompt | det | conf | min | margin | IoU | \|Δw\|% | \|Δh\|% |
|---|---|---|---|---|---|---|---|---|
| color + label | `white shoe` | 11/11 | **0.904** | 0.898 | **0.55** | 1.0 | 0.0 | 0.04 |
| short label | `shoe` | 11/11 | 0.899 | 0.890 | 0.54 | 1.0 | 0.0 | 0.0 |
| wrong color ⚠ | `black shoe` | 11/11 | 0.898 | 0.888 | 0.54 | 1.0 | 0.0 | 0.0 |
| category | `sneaker` | 11/11 | 0.862 | 0.851 | 0.50 | 1.0 | 0.0 | 0.02 |
| over-specified | `brand new white nike running sneaker with white laces and gray mesh panels` | 11/11 | 0.595 | 0.590 | 0.24 | 1.0 | 0.01 | 0.05 |
| full description | `white mesh running sneaker` | 11/11 | 0.592 | 0.586 | 0.24 | 1.0 | 0.01 | 0.0 |
| placeholder ⚠ | `insert prompt` | **9/11** | 0.359 | 0.351 | **0.001** | 1.0 | 0.01 | 0.0 |

**Implications.** Three findings, all important:

1. **The placeholder finally failed** — it missed 2 of 11 frames outright, and its worst surviving frame cleared the threshold by 0.001. This is what a margin of ~0.05 on the other products was warning about: garbage prompts live at the edge of the threshold, and which side of it they land on varies frame to frame.
2. **`white shoe` did not grab the white A4 sheet** — the scene contained another large white object, and the color word still boxed the shoe every time (IoU 1.0). In a single-salient-product scene, object salience dominates attribute matching.
3. **The 4-word description scored as badly as the 12-word one** (0.59 both). Description length isn't the whole story — uncommon or scene-mismatched words (`mesh`, `running`) dilute the match as much as sheer word count. Descriptive prompts are *unpredictable*: the same style scored 0.85 on the Converse and 0.59 here. The simple noun forms were the only style that scored ~0.9 on all three products.

---

## What this means for operating the pipeline

1. **Write a simple noun plus at most one attribute** — `black shoe`, `red handbag`. This form scored 0.90–0.92 on every product; it is the only style that was consistently strong.
2. **Do not stack adjectives.** Every added word either did nothing or drained confidence — up to −45% — and never improved the box. Longer ≠ more precise.
3. **High confidence is not proof your description matched.** Wrong-color prompts scored 0.86–0.90 on all three products. The score means "locked onto an object", not "your words were right". Always check the `*_detections.jpg` debug images.
4. **Prompt choice affects reliability, not accuracy.** With the capture protocol followed (one product, clean background), every prompt that detected produced the same box, hence the same centimeters. The risk of a bad prompt is silently losing frames, not mismeasuring.
5. **If the summary line shows missed frames, simplify the prompt before touching thresholds.** Rewording toward a plain noun recovers margin; lowering `BOX_THRESHOLD` just moves the cliff closer to the noise floor (the placeholder's 0.35–0.37 scores show what lives just above the current threshold).
6. **Untested: scenes with two similar objects.** Attributes were never verified against pixels here, so do not count on `black shoe` to pick the right one of two shoes. Remove the second object from the frame instead — that keeps you in the regime where all of the above holds.

## Method notes

- The consensus reference box is the per-frame median of the five sensible prompts' final SAM boxes; the two controls are measured against it but excluded from it.
- All metrics use the **SAM mask's tight box** (what measurement actually reads), not Grounding DINO's raw box.
- All three sets share one room, one camera, one lighting setup, and one salient product per scene. Confidence *values* will shift in other conditions; the *ranking* of prompt styles is the transferable result.
- To test prompts for a new product: drop its frames into `prompt_experiments/frames/<product>/`, add a `PRODUCTS` entry in `prompt_robustness.py`, and run it. Delete `results_<product>.json` to re-run.
