"""
Diagnostic evaluation: run the *exact same* inference code the API uses
(app.inference.TBClassifier) against every image in data/val, and print a
confusion matrix plus the raw probability for each prediction.

Why this matters: if the model is systematically predicting "Normal" for
everything (not just occasionally missing TB cases), the raw probabilities
will tell us whether that's a genuine model problem (probabilities spread
out normally, just biased toward low) or an inference bug (probabilities
suspiciously uniform, or all exactly the same value, which usually means
something upstream - preprocessing, wrong file being loaded, etc. - is
broken rather than the model itself being bad).

Usage:
    python evaluate.py
    python evaluate.py --val-dir data/val --model model.h5
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.inference import TBClassifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", default="data/val")
    parser.add_argument("--model", default="model.h5")
    parser.add_argument("--max-per-class", type=int, default=None, help="Cap images per class (for a quick run)")
    args = parser.parse_args()

    clf = TBClassifier(model_path=args.model)
    clf.load()
    if not clf.is_loaded:
        raise SystemExit(f"Could not load model from {args.model}")

    classes = sorted(
        d for d in os.listdir(args.val_dir) if os.path.isdir(os.path.join(args.val_dir, d))
    )
    print(f"Classes found: {classes}")

    confusion = defaultdict(lambda: defaultdict(int))  # confusion[true][predicted] = count
    tb_probs_by_true_class = defaultdict(list)
    misclassified_examples = []
    bad_files = []
    all_tb_probs = []  # (true_class, tb_pct) for every evaluated image - used by the threshold sweep below

    total = 0
    for true_class in classes:
        class_dir = os.path.join(args.val_dir, true_class)
        files = sorted(
            f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if args.max_per_class:
            files = files[: args.max_per_class]

        evaluated = 0
        for fname in files:
            path = os.path.join(class_dir, fname)
            try:
                with open(path, "rb") as f:
                    raw = f.read()
                result = clf.predict(raw)
            except Exception as exc:
                bad_files.append((path, str(exc)))
                continue

            confusion[true_class][result.label] += 1
            tb_probs_by_true_class[true_class].append(result.tb_pct)
            all_tb_probs.append((true_class, result.tb_pct))
            total += 1
            evaluated += 1

            if result.label != true_class and len(misclassified_examples) < 15:
                misclassified_examples.append((path, true_class, result.label, result.tb_pct))

        print(f"  {true_class}: {evaluated}/{len(files)} images evaluated"
              + (f"  ({len(files) - evaluated} failed to open)" if evaluated != len(files) else ""))

    print(f"\nTotal images evaluated: {total}\n")

    print("Confusion matrix (rows = true class, columns = predicted):")
    header = "".ljust(16) + "".join(c.ljust(16) for c in classes)
    print(header)
    for true_class in classes:
        row = true_class.ljust(16)
        for pred_class in classes:
            row += str(confusion[true_class][pred_class]).ljust(16)
        print(row)

    print("\nPer-class metrics:")
    for c in classes:
        tp = confusion[c][c]
        total_true = sum(confusion[c].values())
        recall = tp / total_true if total_true else float("nan")
        total_pred = sum(confusion[other][c] for other in classes)
        precision = tp / total_pred if total_pred else float("nan")
        print(f"  {c}: recall={recall:.1%} (caught {tp}/{total_true}), precision={precision:.1%}")

    print("\nTuberculosis-probability stats by true class (should be LOW for Normal, HIGH for Tuberculosis):")
    for c in classes:
        probs = tb_probs_by_true_class[c]
        if probs:
            print(f"  {c}: min={min(probs):.1f}%  max={max(probs):.1f}%  avg={sum(probs)/len(probs):.1f}%")

    if misclassified_examples:
        print(f"\nFirst {len(misclassified_examples)} misclassified examples (true -> predicted, TB probability):")
        for path, true_c, pred_c, tb_pct in misclassified_examples:
            print(f"  {os.path.basename(path)}: {true_c} -> {pred_c}  (TB probability: {tb_pct:.1f}%)")

    if bad_files:
        print(f"\n{len(bad_files)} file(s) could not be opened/read (skipped, not counted above):")
        for path, err in bad_files[:15]:
            print(f"  {path}: {err}")
        if len(bad_files) > 15:
            print(f"  ... and {len(bad_files) - 15} more")

    _print_threshold_sweep(all_tb_probs, classes)


def _print_threshold_sweep(all_tb_probs: list[tuple[str, float]], classes: list[str]) -> None:
    """The model itself only needs to be run once - everything below is just
    re-counting the SAME predictions against different cutoffs, so this is
    instant and doesn't require re-evaluating any images.

    Useful when, like here, the model correctly separates the two classes
    (Tuberculosis probabilities are genuinely low for Normal and high for
    Tuberculosis) but the default 50% cutoff produces too many false
    "Tuberculosis" alarms on Normal x-rays. Raising the threshold trades
    some of that Normal precision back in exchange for how close to the
    edge a Tuberculosis case would have to be before it's missed - for a
    screening tool you generally want to keep Tuberculosis recall as close
    to 100% as safely possible, and only push the threshold up as far as
    that allows.
    """
    if "Tuberculosis" not in classes:
        return
    tb_true_probs = [p for c, p in all_tb_probs if c == "Tuberculosis"]
    normal_true_probs = [p for c, p in all_tb_probs if c == "Normal"]
    if not tb_true_probs or not normal_true_probs:
        return

    print("\nThreshold sweep (recomputed instantly from the same predictions above - no rerun needed):")
    print("  threshold   Normal recall   Tuberculosis recall")
    for pct in (50, 60, 70, 80, 85, 90, 92, 94, 95, 96, 97, 98):
        normal_recall = sum(1 for p in normal_true_probs if p < pct) / len(normal_true_probs)
        tb_recall = sum(1 for p in tb_true_probs if p >= pct) / len(tb_true_probs)
        print(f"  {pct:>3}%        {normal_recall:>6.1%}          {tb_recall:>6.1%}")

    print(
        "\n  -> Pick the smallest threshold where Tuberculosis recall is still 100% (or as close as you're"
        "\n     comfortable with for a screening tool) - that row's Normal recall is the least-false-alarms"
        "\n     you can get without missing real Tuberculosis cases. Then set DECISION_THRESHOLD to that"
        "\n     value (as a fraction, e.g. 0.9 for 90%) - either as an env var, or by editing"
        "\n     `decision_threshold` in app/config.py - and both the API and this script will use it."
    )


if __name__ == "__main__":
    main()
