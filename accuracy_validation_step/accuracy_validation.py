import numpy as np
import os
import json
import glob
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))  # folder this script lives in, so paths work from any cwd
MEASUREMENTS_DIR  = os.path.join(SCRIPT_DIR, '..', 'measurement_extraction_step', 'output')
GROUND_TRUTH_FILE = os.path.join(SCRIPT_DIR, 'ground_truth.json')
OUTPUT_DIR        = os.path.join(SCRIPT_DIR, 'output')
HISTORY_FILE      = os.path.join(OUTPUT_DIR, 'accuracy_history.jsonl')

TOLERANCE_CM            = 2.0   # target error tolerance from the project brief
MIN_VALIDATION_SUBJECTS = 3     # brief recommends validating against 3-5 known objects

# If the pipeline's own reported ± (frame-to-frame precision) is smaller than the
# actual deviation from tape-measure truth by more than this much, flag it as
# "overconfident" -- tight agreement between frames does not mean the measurement
# is correct, it just means the error is systematic rather than random.
PRECISION_ACCURACY_GAP_CM = 1.0

# add real dimensions (in cm) into ground_truth.json for each product you want to validate against
DIMENSIONS = ['width', 'height', 'depth']

# ─── Load data ─────────────────────────────────────────────────────────────────

def load_ground_truth(path):
    """
    Load tape-measure ground truth values.

    Keys starting with '_' (e.g. '_README') are documentation entries and are
    skipped -- this lets the file ship with a usable template despite JSON
    having no comment syntax.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith('_')}


def load_pipeline_measurements(directory):
    """Load every measurements_*.json produced by Step 5, keyed by subject_id."""
    measurements = {}
    for path in sorted(glob.glob(os.path.join(directory, 'measurements_*.json'))):
        with open(path) as f:
            data = json.load(f)
        measurements[data['subject_id']] = data
    return measurements


# ─── Compare one subject ────────────────────────────────────────────────────────

def compare_subject(subject_id, ground_truth, pipeline_result):
    """
    Compare one product's pipeline output against its tape-measure ground truth,
    dimension by dimension.

    Flags two distinct failure modes that are easy to conflate:
      - inaccurate:     predicted value is far from the true value.
      - overconfident:  the reported ± error estimate (precision, from
        frame-to-frame agreement) is small, but the deviation from truth
        (accuracy) is large -- the signature of a systematic bias (e.g. a
        calibration or scale-anchoring problem) rather than random noise.
    """
    results = []
    predicted_cm      = pipeline_result.get('measurements_cm', {})
    reported_error_cm = pipeline_result.get('error_estimates_cm', {})

    for dim in DIMENSIONS:
        true_value = ground_truth.get(dim)
        pred_value = predicted_cm.get(dim)

        if true_value is None or pred_value is None:
            continue  # not measured for this subject/dimension -- nothing to validate

        signed_error_cm  = pred_value - true_value
        abs_error_cm     = abs(signed_error_cm)
        pct_error        = (abs_error_cm / true_value * 100) if true_value else None
        within_tolerance = abs_error_cm <= TOLERANCE_CM

        reported_precision_cm = reported_error_cm.get(dim)
        overconfident = (
            reported_precision_cm is not None
            and (abs_error_cm - reported_precision_cm) > PRECISION_ACCURACY_GAP_CM
        )

        results.append({
            'subject_id':           subject_id,
            'dimension':            dim,
            'true_cm':              true_value,
            'predicted_cm':         pred_value,
            'signed_error_cm':      round(signed_error_cm, 2),
            'abs_error_cm':         round(abs_error_cm, 2),
            'pct_error':            round(pct_error, 1) if pct_error is not None else None,
            'reported_precision_cm': reported_precision_cm,
            'within_tolerance':     within_tolerance,
            'overconfident':        overconfident,
        })

    return results


# ─── Aggregate across all validated subjects ───────────────────────────────────

def aggregate(all_results):
    """
    Roll per-measurement comparisons up into per-dimension statistics.

    Reports both mean absolute error (overall accuracy) and mean SIGNED error
    (systematic bias -- consistently over- or under-measuring in one direction,
    which random noise would NOT produce; random error averages back towards
    zero, bias does not).
    """
    summary = {}
    for dim in DIMENSIONS:
        dim_results = [r for r in all_results if r['dimension'] == dim]
        if not dim_results:
            continue

        abs_errors    = [r['abs_error_cm'] for r in dim_results]
        signed_errors = [r['signed_error_cm'] for r in dim_results]
        pass_count    = sum(1 for r in dim_results if r['within_tolerance'])
        overconfident_count = sum(1 for r in dim_results if r['overconfident'])

        summary[dim] = {
            'n':                   len(dim_results),
            'mean_abs_error_cm':   round(float(np.mean(abs_errors)), 2),
            'max_abs_error_cm':    round(float(np.max(abs_errors)), 2),
            'mean_signed_bias_cm': round(float(np.mean(signed_errors)), 2),
            'pass_rate':           round(pass_count / len(dim_results), 2),
            'overconfident_count': overconfident_count,
        }

    return summary


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 7 — Accuracy Validation ===\n")

    ground_truth = load_ground_truth(GROUND_TRUTH_FILE)
    if not ground_truth:
        print(f"[!] No ground truth entries found in '{GROUND_TRUTH_FILE}'.")
        print("    Tape-measure at least 3-5 known products and add their real")
        print("    dimensions to that file, keyed by subject_id, before validating.")
        return

    pipeline_measurements = load_pipeline_measurements(MEASUREMENTS_DIR)
    if not pipeline_measurements:
        print(f"[!] No pipeline output found in '{MEASUREMENTS_DIR}'.")
        print("    Run measurement_extraction.py first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []
    skipped = []

    for subject_id, truth in ground_truth.items():
        pipeline_result = pipeline_measurements.get(subject_id)
        if pipeline_result is None:
            skipped.append(subject_id)
            continue

        subject_results = compare_subject(subject_id, truth, pipeline_result)
        all_results.extend(subject_results)

        print(f"Subject: {subject_id}")
        for r in subject_results:
            flag = "PASS" if r['within_tolerance'] else "FAIL"
            warn = "  [!] overconfident — tight ± but far from truth" if r['overconfident'] else ""
            print(f"  {r['dimension']:<8} true={r['true_cm']:.1f}cm  pred={r['predicted_cm']:.1f}cm  "
                  f"error={r['signed_error_cm']:+.2f}cm  ({flag}){warn}")
        print()

    if skipped:
        print(f"[!] No pipeline output for: {', '.join(skipped)} — skipped.\n")

    if not all_results:
        print("[!] No overlapping subject_id + dimension between ground truth and pipeline output.")
        print("    Check that SUBJECT_ID in measurement_extraction.py matches a key in ground_truth.json.")
        return

    validated_subjects = len(set(r['subject_id'] for r in all_results))
    if validated_subjects < MIN_VALIDATION_SUBJECTS:
        print(f"[!] Only {validated_subjects} validated subject(s) — the brief recommends at least "
              f"{MIN_VALIDATION_SUBJECTS} for a reliable accuracy estimate. Treat this report as provisional.\n")

    summary = aggregate(all_results)

    print("─" * 60)
    print("Summary by dimension:")
    overall_pass = True
    for dim, s in summary.items():
        verdict = "PASS" if s['mean_abs_error_cm'] <= TOLERANCE_CM else "FAIL"
        if verdict == "FAIL":
            overall_pass = False
        print(f"  {dim:<8} n={s['n']:<3} MAE={s['mean_abs_error_cm']:.2f}cm  "
              f"max={s['max_abs_error_cm']:.2f}cm  bias={s['mean_signed_bias_cm']:+.2f}cm  "
              f"pass_rate={s['pass_rate']*100:.0f}%  ({verdict})")
        if s['overconfident_count'] > 0:
            print(f"           [!] {s['overconfident_count']} measurement(s) were overconfident — "
                  f"reported error bars looked tight but missed the true value. That usually means "
                  f"a systematic bias (calibration or scale anchoring), not random noise — check "
                  f"those before adding more frames.")

    print()
    if overall_pass:
        print(f"Overall: PASS — all dimensions within {TOLERANCE_CM}cm target on average.")
    else:
        print(f"Overall: FAIL — one or more dimensions exceed the {TOLERANCE_CM}cm target.")
        print("  Per the brief's accuracy levers, check in this order:")
        print("  1. Camera calibration reprojection error")
        print("  2. A4 sheet detection reliability / scale anchoring")
        print("  3. Frame count / capture protocol discipline")

    # Save full report
    report = {
        'validated_at':         datetime.now().isoformat(),
        'tolerance_cm':         TOLERANCE_CM,
        'validated_subjects':   validated_subjects,
        'overall_pass':         overall_pass,
        'summary_by_dimension': summary,
        'measurements':         all_results,
        'skipped_subjects':     skipped,
    }
    report_path = os.path.join(OUTPUT_DIR, 'accuracy_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Append a one-line summary to the history log so accuracy can be tracked
    # across pipeline changes over time (e.g. did last week's calibration fix help?)
    history_entry = {
        'validated_at':       report['validated_at'],
        'validated_subjects':  validated_subjects,
        'overall_pass':        overall_pass,
        'mean_abs_error_cm':  {dim: s['mean_abs_error_cm'] for dim, s in summary.items()},
    }
    with open(HISTORY_FILE, 'a') as f:
        f.write(json.dumps(history_entry) + '\n')

    print(f"\nFull report saved to: {report_path}")
    print(f"Run history appended to: {HISTORY_FILE}")


if __name__ == '__main__':
    main()
