"""Why are the trainval boards easier than the official test boards?

Three measurements now agree that they are -- a cross-validated threshold
(2026-08-31), a gradient-boosted tree over hand features and a per-seed
validation choice (both 2026-09-01) -- and none of them shares the others'
failure mode. What none of them says is *why*, and the answer changes what to
do next:

- **a few hard boards** -- the gap is about particular boards, and a different
  or larger split would dissolve it; the honest response is a bigger test set;
- **uniformly harder** -- it is a property of the official split, and the
  honest response is to state it and carry it;
- **composition** -- the two sides differ in class mix, defect size or
  candidates per board, and the gap is arithmetic rather than difficulty.

Four readings, in that order. The third is the decisive one: test defects are
re-weighted to trainval's own (class x size) distribution and the escape rate
recomputed. If direct standardisation closes the gap, the gap was composition.

The two sides are paired the way the shipped procedure pairs them -- out-of-fold
predictions over trainval against the shipped checkpoint on test -- because that
is the pairing whose gap is being decomposed, not an arbitrary one.

    uv run python scripts/gap_decomposition.py            # ~6 min (5 folds)
    uv run python scripts/gap_decomposition.py --oof models/cv_oof.npz
                                                          # reuse a previous run

Writes `models/gap_decomposition.json` and caches the out-of-fold matrix;
appends to docs/benchmarks.md unless `--dry-run`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aoi_agent.vision.inference import DEFAULT_DISMISS_THRESHOLD  # noqa: E402
from aoi_agent.vision.model import select_device  # noqa: E402
from aoi_agent.vision.patches import PatchSet  # noqa: E402
from threshold_cv import select_threshold  # noqa: E402
from train import DEFAULTS  # noqa: E402


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def side(scores, labels, boards, boxes, false_call_index, threshold):
    """One side of the comparison, reduced to what the four readings need."""
    defect = labels != false_call_index
    escaped = defect & (scores >= threshold)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return {
        "candidates": int(len(labels)),
        "defects": int(defect.sum()),
        "escapes": int(escaped.sum()),
        "escape_rate": float(escaped.sum() / max(defect.sum(), 1)),
        "boards": int(len(np.unique(boards))),
        "defect_scores": scores[defect],
        "defect_labels": labels[defect],
        "defect_boards": boards[defect],
        "defect_areas": areas[defect],
        "escaped_boards": boards[escaped],
    }


def concentration(escaped_boards: np.ndarray, boards: np.ndarray) -> dict:
    """How few boards hold how much of the escaping."""
    if len(escaped_boards) == 0:
        return {"boards_with_any": 0, "top1": 0.0, "top5": 0.0, "worst_board_count": 0}
    counts = np.bincount(escaped_boards)
    ordered = np.sort(counts[counts > 0])[::-1]
    total = ordered.sum()
    return {
        "boards_with_any": int((counts > 0).sum()),
        "boards_total": int(len(np.unique(boards))),
        "top1": float(ordered[:1].sum() / total),
        "top5": float(ordered[:5].sum() / total),
        "worst_board_count": int(ordered[0]),
    }


def boards_to_remove(test, target_rate: float) -> dict:
    """How many of the worst test boards must go before the gap closes."""
    counts = np.bincount(test["escaped_boards"],
                         minlength=int(test["defect_boards"].max()) + 1)
    defects = np.bincount(test["defect_boards"], minlength=len(counts))
    order = np.argsort(counts)[::-1]
    escapes, total = test["escapes"], test["defects"]
    for removed, board in enumerate(order, 1):
        escapes -= int(counts[board])
        total -= int(defects[board])
        if total <= 0:
            break
        if escapes / total <= target_rate:
            return {"boards": removed, "of": test["boards"],
                    "fraction": removed / test["boards"],
                    "rate_after": escapes / total}
    return {"boards": None, "of": test["boards"]}


def standardised(test, trainval, label_names, false_call_index, threshold, buckets=3) -> dict:
    """Test's escape rate re-weighted to trainval's own (class x size) mix.

    Direct standardisation: if the two sides only differ in what they contain,
    matching the mix removes the difference. If it does not, the difference is
    in the candidates themselves.
    """
    edges = np.quantile(trainval["defect_areas"], np.linspace(0, 1, buckets + 1)[1:-1])

    def key(row_labels, row_areas):
        return row_labels * (buckets + 1) + np.digitize(row_areas, edges)

    train_key = key(trainval["defect_labels"], trainval["defect_areas"])
    test_key = key(test["defect_labels"], test["defect_areas"])
    test_escaped = test["defect_scores"] >= threshold

    weighted, covered = 0.0, 0.0
    per_bucket = []
    for bucket in np.unique(train_key):
        weight = float((train_key == bucket).mean())
        inside = test_key == bucket
        if not inside.any():
            continue
        rate = float(test_escaped[inside].mean())
        weighted += weight * rate
        covered += weight
        per_bucket.append({
            "class": label_names[bucket // (buckets + 1)],
            "size_bucket": int(bucket % (buckets + 1)),
            "trainval_share": weight,
            "test_share": float(inside.mean()),
            "test_escape_rate": rate,
        })
    return {"standardised_rate": weighted / max(covered, 1e-9),
            "weight_covered": covered, "buckets": per_bucket}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=Path, default=ROOT / "data" / "patches")
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "models" / "test_predictions.npz")
    parser.add_argument("--oof", type=Path, default=ROOT / "models" / "cv_oof.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "gap_decomposition.json")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "docs" / "benchmarks.md")
    parser.add_argument("--threshold", type=float, default=DEFAULT_DISMISS_THRESHOLD)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--budget", type=float, default=DEFAULTS["escape_budget"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    trainval_set = PatchSet.load(args.patches / "trainval.npz")
    test_set = PatchSet.load(args.patches / "test.npz")
    label_names = list(trainval_set.label_names)
    false_call_index = label_names.index("false_call")
    started = time.perf_counter()

    if args.oof.exists():
        print(f"reusing out-of-fold predictions from {args.oof}")
        oof = np.load(args.oof)["probabilities"]
    else:
        print(f"running {args.folds} folds to get out-of-fold predictions")
        oof, _optimistic, _guarded = select_threshold(
            trainval_set, folds=args.folds, budget=args.budget, epochs=args.epochs,
            seed=args.seed, device=select_device(args.device), log=print)
        np.savez_compressed(args.oof, probabilities=oof)

    stored = np.load(args.predictions)
    threshold = args.threshold
    trainval = side(oof[:, false_call_index], trainval_set.labels, trainval_set.image_index,
                    trainval_set.boxes, false_call_index, threshold)
    test = side(stored["probabilities"][:, false_call_index], stored["labels"],
                test_set.image_index, test_set.boxes, false_call_index, threshold)
    assert (stored["labels"] == test_set.labels).all(), "predictions and patches disagree"

    quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
    record = {
        "commit": commit(), "threshold": threshold, "folds": args.folds,
        "seed": args.seed, "wall_seconds": round(time.perf_counter() - started, 1),
        "trainval": {k: v for k, v in trainval.items() if not isinstance(v, np.ndarray)},
        "test": {k: v for k, v in test.items() if not isinstance(v, np.ndarray)},
        "defect_score_quantiles": {
            "quantiles": quantiles,
            "trainval": [float(x) for x in np.quantile(trainval["defect_scores"], quantiles)],
            "test": [float(x) for x in np.quantile(test["defect_scores"], quantiles)],
        },
        "concentration": {
            "trainval": concentration(trainval["escaped_boards"], trainval["defect_boards"]),
            "test": concentration(test["escaped_boards"], test["defect_boards"]),
        },
        "leave_out": boards_to_remove(test, trainval["escape_rate"]),
        "standardisation": standardised(test, trainval, label_names, false_call_index, threshold),
        "class_mix": {
            "trainval": {label_names[c]: float((trainval["defect_labels"] == c).mean())
                         for c in range(len(label_names)) if c != false_call_index},
            "test": {label_names[c]: float((test["defect_labels"] == c).mean())
                     for c in range(len(label_names)) if c != false_call_index},
        },
        "defects_per_board": {
            "trainval": trainval["defects"] / trainval["boards"],
            "test": test["defects"] / test["boards"],
        },
        "median_defect_area": {
            "trainval": float(np.median(trainval["defect_areas"])),
            "test": float(np.median(test["defect_areas"])),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")

    gap = test["escape_rate"] - trainval["escape_rate"]
    closed = (test["escape_rate"] - record["standardisation"]["standardised_rate"]) / max(gap, 1e-9)
    print(json.dumps({k: v for k, v in record.items()
                      if k in ("trainval", "test", "leave_out")}, indent=2))

    q = record["defect_score_quantiles"]
    lines = [
        f"## {date.today().isoformat()} · commit {commit()}",
        "",
        "### Why are the trainval boards easier? Four readings",
        "",
        f"Three measurements agree the official test boards are harder than "
        f"trainval and none says why. Here the two sides are paired the way the "
        f"shipped procedure pairs them -- {args.folds}-fold out-of-fold "
        f"predictions over all {trainval['candidates']:,} trainval candidates "
        f"against the shipped checkpoint on the {test['candidates']:,} test "
        f"candidates -- and read at the shipped threshold {threshold}. "
        f"`scripts/gap_decomposition.py`.",
        "",
        "| | candidates | defects | boards | escapes | escape rate |",
        "|---|---|---|---|---|---|",
        f"| trainval (out-of-fold) | {trainval['candidates']:,} | "
        f"{trainval['defects']:,} | {trainval['boards']:,} | {trainval['escapes']} | "
        f"**{trainval['escape_rate']:.3%}** |",
        f"| test (shipped checkpoint) | {test['candidates']:,} | {test['defects']:,} | "
        f"{test['boards']:,} | {test['escapes']} | **{test['escape_rate']:.3%}** |",
        "",
        "**Reading 1 — the score a defect is given.** The threshold cuts "
        "`P(false_call)`, so the gap has to appear in what that number is on "
        "defect-labelled candidates. Quantiles:",
        "",
        "| quantile | " + " | ".join(f"{p:.0%}" for p in quantiles) + " |",
        "|---|" + "---|" * len(quantiles),
        "| trainval | " + " | ".join(f"{x:.4f}" for x in q["trainval"]) + " |",
        "| test | " + " | ".join(f"{x:.4f}" for x in q["test"]) + " |",
        "",
        f"**Reading 2 — how few boards hold the escaping.** On test "
        f"{record['concentration']['test']['boards_with_any']} of "
        f"{record['concentration']['test']['boards_total']} boards escape "
        f"anything at all, the worst one holds "
        f"{record['concentration']['test']['worst_board_count']} of "
        f"{test['escapes']}, and the worst five hold "
        f"{record['concentration']['test']['top5']:.0%}. On trainval: "
        f"{record['concentration']['trainval']['boards_with_any']} of "
        f"{record['concentration']['trainval']['boards_total']}, worst five "
        f"{record['concentration']['trainval']['top5']:.0%}.",
        "",
        (f"**Reading 3 — how many boards would have to go.** Removing the "
         f"{record['leave_out']['boards']} worst test boards "
         f"({record['leave_out']['fraction']:.1%} of them) brings test down to "
         f"{record['leave_out']['rate_after']:.3%}, trainval's own rate."
         if record["leave_out"]["boards"] is not None else
         "**Reading 3 — how many boards would have to go.** No number of "
         "removals reaches trainval's rate: the remaining boards keep escaping."),
        "",
        f"**Reading 4 — composition, and this is the decisive one.** Test's "
        f"defects re-weighted to trainval's own (class x size-tercile) mix give "
        f"**{record['standardisation']['standardised_rate']:.3%}** against the "
        f"raw {test['escape_rate']:.3%} -- direct standardisation closes "
        f"{closed:.0%} of the gap. Class mix and defect size are therefore "
        f"{'most of the story' if closed > 0.5 else 'not the story'}. "
        f"Median defect area {record['median_defect_area']['trainval']:.0f} px "
        f"against {record['median_defect_area']['test']:.0f} px; defects per "
        f"board {record['defects_per_board']['trainval']:.1f} against "
        f"{record['defects_per_board']['test']:.1f}.",
        "",
        "**What this does not establish.** One checkpoint, one seed, one "
        "threshold, and an out-of-fold side whose predictions come from fold "
        "models rather than the shipped one -- that pairing is the procedure's "
        "own, but it means the two sides differ by more than the boards. "
        "Standardisation over (class x size) can only rule those two out; a "
        "difference in what a defect of a given class and size *looks like* on "
        "the test boards would pass through it untouched.",
        "",
    ]
    body = "\n".join(lines)
    if args.dry_run:
        print("\n" + body)
    else:
        with args.benchmarks.open("a") as handle:
            handle.write("\n" + body)
        print(f"\nappended to {args.benchmarks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
