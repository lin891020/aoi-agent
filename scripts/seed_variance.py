"""What a re-run of this pipeline produces: the headline over several seeds.

Every figure this project publishes comes from one seed. That is a point
estimate of a *procedure*, and the procedure has variance the escape rate's
Wilson interval does not describe: the seed moves the by-image split, the
initialisation, the shuffling, and -- since 2026-08-31 -- the threshold the
cross-validated selection lands on. Two different kinds of uncertainty, and
quoting only the first reads as though re-running would give the same number.

So each seed runs the whole thing, not the training half of it:

1. `threshold_cv.select_threshold` over five folds of trainval, choosing on the
   interval's upper bound at the budget;
2. a final model on that seed's own by-image split, exactly as `train.py` does;
3. the test split read once, at that seed's threshold.

The shipped checkpoint stays seed 0. What the spread describes is what a
re-run would land on, not an error bar on the model that is deployed. And it
covers only what the seed moves: one dataset, one architecture, one recipe.

    uv run python scripts/seed_variance.py                 # ~40 min on the M5 Air
    uv run python scripts/seed_variance.py --seeds 0 1 2   # a shorter read

Writes `models/seed_variance.json`; appends to docs/benchmarks.md unless
`--dry-run`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aoi_agent.stats import wilson  # noqa: E402
from aoi_agent.vision.dataset import CandidateDataset  # noqa: E402
from aoi_agent.vision.model import select_device  # noqa: E402
from aoi_agent.vision.operating_point import best_at_escape_budget, sweep  # noqa: E402
from aoi_agent.vision.patches import PatchSet  # noqa: E402
from threshold_cv import select_threshold  # noqa: E402
from train import DEFAULTS, fit, predict, split_by_image  # noqa: E402


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def one_seed(trainval: PatchSet, test: PatchSet, seed: int, args, device, log) -> dict:
    label_names = trainval.label_names
    false_call_index = label_names.index("false_call")

    log(f"\n=== seed {seed}: selecting the threshold over {args.folds} folds ===")
    _oof, _optimistic, guarded = select_threshold(
        trainval, folds=args.folds, budget=args.budget, epochs=args.epochs,
        seed=seed, device=device, log=log)
    if guarded is None:
        return {"seed": seed, "threshold": None}

    log(f"\n=== seed {seed}: the final model, on its own by-image split ===")
    train_idx, val_idx = split_by_image(trainval, DEFAULTS["val_fraction"], seed)
    model, _history, _best = fit(
        Subset(CandidateDataset(trainval, augment=True), train_idx),
        Subset(CandidateDataset(trainval, augment=False), val_idx),
        trainval, label_names,
        epochs=args.epochs, batch_size=DEFAULTS["batch_size"], lr=DEFAULTS["lr"],
        seed=seed, device=device, escape_budget=args.budget,
        checkpoint=None, log=lambda line: log("  " + line),
    )
    loader = DataLoader(CandidateDataset(test, augment=False), batch_size=256)
    probabilities, labels = predict(model, loader, device)
    del model

    scores = probabilities[:, false_call_index]
    point = sweep(scores, labels, false_call_index,
                  thresholds=np.array([guarded.threshold]))[0]
    low, high = wilson(point.escapes, point.defects_total)
    # The same seed read at its own oracle on this split. Not deployable -- it is
    # chosen here -- but it separates two things the deployed column mixes: how
    # good the model is, and where the selection rule happened to land on its
    # curve. Without it a seed that removes less review looks like a worse model
    # when it may only be sitting further right.
    oracle = best_at_escape_budget(sweep(scores, labels, false_call_index), args.budget)
    row = {
        "seed": seed,
        "threshold": guarded.threshold,
        "oof_escape_rate": guarded.escape_rate,
        "oof_review_reduction": guarded.review_reduction,
        "test_escape_rate": point.escape_rate,
        "test_escapes": int(point.escapes),
        "test_defects": int(point.defects_total),
        "test_escape_ci": [low, high],
        "test_review_reduction": point.review_reduction,
        "oracle_threshold": None if oracle is None else oracle.threshold,
        "oracle_review_reduction": None if oracle is None else oracle.review_reduction,
        "oracle_escape_rate": None if oracle is None else oracle.escape_rate,
    }
    log(f"\nseed {seed}: thr {row['threshold']:.4f}  out-of-fold {row['oof_escape_rate']:.3%}  "
        f"test {row['test_escape_rate']:.3%} ({point.escapes}/{point.defects_total})  "
        f"review {row['test_review_reduction']:.2%}")
    return row


def spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "n": len(values)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=Path, default=ROOT / "data" / "patches")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "seed_variance.json")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "docs" / "benchmarks.md")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--budget", type=float, default=DEFAULTS["escape_budget"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    trainval = PatchSet.load(args.patches / "trainval.npz")
    test = PatchSet.load(args.patches / "test.npz")
    device = select_device(args.device)
    started = time.perf_counter()
    rows = [one_seed(trainval, test, seed, args, device, print) for seed in args.seeds]
    rows = [r for r in rows if r.get("threshold") is not None]
    wall = time.perf_counter() - started
    if not rows:
        print("no seed produced a threshold within the budget", file=sys.stderr)
        return 1

    thresholds = spread([r["threshold"] for r in rows])
    escapes = spread([r["test_escape_rate"] for r in rows])
    reviews = spread([r["test_review_reduction"] for r in rows])
    oof = spread([r["oof_escape_rate"] for r in rows])
    oracles = spread([r["oracle_review_reduction"] for r in rows
                      if r["oracle_review_reduction"] is not None])
    record = {"commit": commit(), "seeds": args.seeds, "folds": args.folds,
              "epochs": args.epochs, "budget": args.budget,
              "wall_seconds": round(wall, 1), "rows": rows,
              "threshold": thresholds, "test_escape_rate": escapes,
              "test_review_reduction": reviews, "oof_escape_rate": oof,
              "oracle_review_reduction": oracles}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")

    lines = [
        f"## {date.today().isoformat()} · commit {commit()}",
        "",
        "### Seed variance — what a re-run of the whole pipeline lands on",
        "",
        f"Every figure elsewhere in this file is one seed. The seed moves the "
        f"by-image split, the initialisation, the shuffling and the threshold the "
        f"cross-validated selection returns, so it is a source of uncertainty the "
        f"escape rate's Wilson interval says nothing about. {len(rows)} seeds, each "
        f"running the whole procedure -- {args.folds}-fold selection, then a final "
        f"model on that seed's own split, then the test split read once at that "
        f"seed's threshold. {wall / 60:.0f} min. `scripts/seed_variance.py`.",
        "",
        "| seed | threshold | out-of-fold escape | test escape | 95% interval | "
        "test review removed | oracle on this split |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['threshold']:.4f} | {r['oof_escape_rate']:.3%} | "
            f"{r['test_escape_rate']:.3%} ({r['test_escapes']}/{r['test_defects']}) | "
            f"{r['test_escape_ci'][0]:.2%}–{r['test_escape_ci'][1]:.2%} | "
            f"{r['test_review_reduction']:.2%} | "
            f"{r['oracle_review_reduction']:.2%} @ {r['oracle_threshold']:.4f} |")
    lines += [
        f"| **median** | **{thresholds['median']:.4f}** | {oof['median']:.3%} | "
        f"**{escapes['median']:.3%}** | — | **{reviews['median']:.2%}** | "
        f"{oracles['median']:.2%} |",
        f"| range | {thresholds['min']:.4f}–{thresholds['max']:.4f} | "
        f"{oof['min']:.3%}–{oof['max']:.3%} | {escapes['min']:.3%}–{escapes['max']:.3%} | — | "
        f"{reviews['min']:.2%}–{reviews['max']:.2%} | "
        f"{oracles['min']:.2%}–{oracles['max']:.2%} |",
        "",
        f"**How much of the spread is the model, and how much is where the "
        f"threshold landed?** The last column is each seed read at its own oracle "
        f"on this split -- not deployable, since it is chosen here, but it holds "
        f"the operating point fixed at the budget and so varies only with the "
        f"model. Its range is {oracles['min']:.2%}–{oracles['max']:.2%} against "
        f"{reviews['min']:.2%}–{reviews['max']:.2%} for the deployed column: what "
        f"that comparison separates is a model that got better or worse from a "
        f"selection rule that landed further along the same curve.",
        "",
        f"**Two intervals, and they are not the same interval.** Within one seed the "
        f"escape rate carries a Wilson interval, which is sampling error on a fixed "
        f"model. Across seeds the whole procedure moves, and the spread above is what "
        f"a re-run lands on. Quoting only the first reads as though re-running would "
        f"return the same number.",
        "",
        f"**What this does not cover.** One dataset, one architecture, one recipe, and "
        f"one test split -- every seed is read on the same 3,018 defect-labelled "
        f"candidates, so the spread says nothing about a different set of boards. The "
        f"shipped checkpoint remains seed 0; this is the variance of the procedure, "
        f"not an error bar on the model that is deployed.",
        "",
    ]
    body = "\n".join(lines)
    print("\n" + body)
    if args.dry_run:
        print("(dry run -- nothing appended)")
        return 0
    existing = args.benchmarks.read_text() if args.benchmarks.exists() else "# Benchmarks\n"
    args.benchmarks.write_text(existing.rstrip() + "\n\n" + body.rstrip() + "\n")
    print(f"appended to {args.benchmarks.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
