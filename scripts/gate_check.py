"""S0 gate — does template differencing actually produce false calls?

The whole project rests on one assumption: that a cheap AOI-style detector
flags far more regions than are genuinely defective, leaving a real
re-verification problem to solve. DeepPCB contains only true defects, so the
false calls have to come from the detector itself.

This script measures whether they do. It sweeps the detector threshold and
reports, for each setting, how many real defects are caught and how many
spurious regions come along for the ride.

Pass criteria (from the plan):
    recall >= 95%  AND  mean false calls per image >= 2

Usage::

    uv run python scripts/gate_check.py
    uv run python scripts/gate_check.py --limit 500 --perturb
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.aoi.matching import match  # noqa: E402
from aoi_agent.aoi.simulator import (  # noqa: E402
    DetectorConfig,
    Perturbation,
    detect,
)
from aoi_agent.data.deeppcb import CLASS_NAMES, load_split  # noqa: E402

RECALL_TARGET = 0.95
FALSE_CALLS_TARGET = 2.0


def evaluate(pairs, config: DetectorConfig, perturbation: Perturbation | None):
    """Run the detector over every pair and summarise the outcome."""
    total_annotations = 0
    total_detected = 0
    false_calls_per_image: list[int] = []
    candidates_per_image: list[int] = []
    detected_by_class: Counter[str] = Counter()
    total_by_class: Counter[str] = Counter()

    for index, pair in enumerate(pairs):
        template = pair.load_template()
        test = pair.load_test()
        annotations = pair.load_annotations()

        # Vary the perturbation per image so the whole run does not share one
        # registration error.
        per_image = (
            None
            if perturbation is None
            else Perturbation(
                max_shift_px=perturbation.max_shift_px,
                noise_sigma=perturbation.noise_sigma,
                gain=perturbation.gain,
                seed=index,
            )
        )

        candidates = detect(template, test, config, per_image)
        result = match(candidates, annotations)

        total_annotations += len(annotations)
        total_detected += len(result.detected_annotations)
        false_calls_per_image.append(len(result.false_calls))
        candidates_per_image.append(len(candidates))

        for annotation in annotations:
            total_by_class[annotation.class_name] += 1
        for annotation in result.detected_annotations:
            detected_by_class[annotation.class_name] += 1

    recall = total_detected / total_annotations if total_annotations else 0.0
    return {
        "recall": recall,
        "total_annotations": total_annotations,
        "total_detected": total_detected,
        "mean_false_calls": statistics.mean(false_calls_per_image),
        "median_false_calls": statistics.median(false_calls_per_image),
        "max_false_calls": max(false_calls_per_image),
        "images_without_false_calls": sum(1 for n in false_calls_per_image if n == 0),
        "mean_candidates": statistics.mean(candidates_per_image),
        "recall_by_class": {
            name: (detected_by_class[name] / total_by_class[name])
            if total_by_class[name]
            else None
            for name in CLASS_NAMES.values()
        },
        "count_by_class": dict(total_by_class),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="trainval", choices=["trainval", "test"])
    parser.add_argument("--limit", type=int, default=200, help="pairs to evaluate")
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[30, 45, 60, 80, 100],
        help="grey-level difference thresholds to sweep",
    )
    parser.add_argument(
        "--perturb",
        action="store_true",
        help="degrade the template to production conditions before differencing",
    )
    parser.add_argument("--shift", type=int, default=2, help="max registration shift px")
    parser.add_argument("--noise", type=float, default=6.0, help="sensor noise sigma")
    parser.add_argument("--gain", type=float, default=1.03, help="illumination gain")
    parser.add_argument("--out", type=Path, default=Path("eval/results/gate_check.json"))
    args = parser.parse_args()

    pairs = load_split(args.split)[: args.limit]
    perturbation = (
        Perturbation(max_shift_px=args.shift, noise_sigma=args.noise, gain=args.gain)
        if args.perturb
        else None
    )

    print(f"S0 gate — {len(pairs)} pairs from {args.split}")
    print(
        "perturbation: "
        + (f"shift +/-{args.shift}px, noise sigma {args.noise}, gain {args.gain}"
           if perturbation
           else "none (dataset as shipped)")
    )
    print()
    header = f"{'thresh':>7} {'recall':>8} {'detected':>14} {'FC/img':>8} {'median':>7} {'max':>5} {'clean imgs':>11}"
    print(header)
    print("-" * len(header))

    runs = []
    for threshold in args.thresholds:
        config = DetectorConfig(threshold=threshold)
        summary = evaluate(pairs, config, perturbation)
        summary["threshold"] = threshold
        summary["config"] = asdict(config)
        summary["perturbation"] = asdict(perturbation) if perturbation else None
        runs.append(summary)
        print(
            f"{threshold:>7} "
            f"{summary['recall']:>7.1%} "
            f"{summary['total_detected']:>6}/{summary['total_annotations']:<7} "
            f"{summary['mean_false_calls']:>8.2f} "
            f"{summary['median_false_calls']:>7.1f} "
            f"{summary['max_false_calls']:>5} "
            f"{summary['images_without_false_calls']:>11}"
        )

    passing = [
        r
        for r in runs
        if r["recall"] >= RECALL_TARGET and r["mean_false_calls"] >= FALSE_CALLS_TARGET
    ]

    print()
    if passing:
        best = max(passing, key=lambda r: r["recall"])
        print(
            f"GATE PASSED — threshold {best['threshold']}: "
            f"recall {best['recall']:.1%}, {best['mean_false_calls']:.2f} false calls/image"
        )
        print("  recall by defect class:")
        for name, value in best["recall_by_class"].items():
            count = best["count_by_class"].get(name, 0)
            shown = f"{value:.1%}" if value is not None else "n/a"
            print(f"    {name:<12} {shown:>7}  (n={count})")
    else:
        best_recall = max(runs, key=lambda r: r["recall"])
        most_fc = max(runs, key=lambda r: r["mean_false_calls"])
        print("GATE NOT PASSED")
        print(
            f"  best recall      : {best_recall['recall']:.1%} "
            f"at threshold {best_recall['threshold']} "
            f"({best_recall['mean_false_calls']:.2f} false calls/image)"
        )
        print(
            f"  most false calls : {most_fc['mean_false_calls']:.2f}/image "
            f"at threshold {most_fc['threshold']} "
            f"(recall {most_fc['recall']:.1%})"
        )
        if perturbation is None:
            print("  next step: re-run with --perturb")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "split": args.split,
                "pairs": len(pairs),
                "passed": bool(passing),
                "runs": runs,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
