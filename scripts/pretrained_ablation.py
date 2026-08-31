"""Does the ImageNet initialisation buy anything on binarised board scans?

The re-verifier starts from ImageNet-1K weights. That was a deliberate choice --
a known baseline, so an improvement attributes to the pipeline and the operating
point rather than to architecture search -- but it was never a *measured* one,
and `train.py` has carried a `--no-pretrained` switch nobody pulled. ImageNet is
photographs of objects; DeepPCB is a binarised difference of two board scans.
The features may transfer, or the initialisation may be doing nothing a random
one would not.

The measurement this needs is not "run it once". `scripts/seed_variance.py`
showed the same recipe spans 3.1 points of review-removed at its own oracle
across five seeds, so a single pair whose arms differ by less than that says
nothing. Both arms therefore run over the same seeds, on the same by-image
split per seed, with everything else held: same patches, same recipe, same
epochs, same class weights.

Both arms are read at their own best threshold on the test split. That is the
selection-on-the-reporting-split defect the deployed threshold avoids, and it is
deliberate here for the reason the model-free floor gives: the comparison is
between two engines, each inherits the same optimism, and correcting one side
only would flatter the other. Nothing here is a deployment number.

    uv run python scripts/pretrained_ablation.py                 # ~25 min, 3 seeds
    uv run python scripts/pretrained_ablation.py --seeds 0 1     # a shorter read

Writes `models/pretrained_ablation.json`; appends to docs/benchmarks.md unless
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

from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aoi_agent.vision.dataset import CandidateDataset  # noqa: E402
from aoi_agent.vision.model import select_device  # noqa: E402
from aoi_agent.vision.operating_point import best_at_escape_budget, sweep  # noqa: E402
from aoi_agent.vision.patches import PatchSet  # noqa: E402
from train import DEFAULTS, fit, predict, split_by_image  # noqa: E402

ARMS = {True: "ImageNet", False: "random"}


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def one_arm(trainval: PatchSet, test: PatchSet, seed: int, pretrained: bool,
            args, device, log) -> dict:
    """One model, one seed, one initialisation, read at its own oracle."""
    label_names = trainval.label_names
    false_call_index = label_names.index("false_call")

    log(f"\n=== seed {seed}, {ARMS[pretrained]} initialisation ===")
    # The same split on both arms of a seed: the split is what the seed moves,
    # and moving it between arms would put that variance inside the comparison.
    train_idx, val_idx = split_by_image(trainval, DEFAULTS["val_fraction"], seed)
    model, history, val_best = fit(
        Subset(CandidateDataset(trainval, augment=True), train_idx),
        Subset(CandidateDataset(trainval, augment=False), val_idx),
        trainval, label_names,
        epochs=args.epochs, batch_size=DEFAULTS["batch_size"], lr=DEFAULTS["lr"],
        seed=seed, device=device, escape_budget=args.budget,
        pretrained=pretrained, checkpoint=None, log=lambda line: log("  " + line),
    )
    loader = DataLoader(CandidateDataset(test, augment=False), batch_size=256)
    probabilities, labels = predict(model, loader, device)
    del model

    scores = probabilities[:, false_call_index]
    oracle = best_at_escape_budget(sweep(scores, labels, false_call_index), args.budget)
    accuracy = float((probabilities.argmax(1) == labels).mean())
    row = {
        "seed": seed,
        "pretrained": pretrained,
        "arm": ARMS[pretrained],
        "test_accuracy": accuracy,
        "val_review_reduction": None if val_best is None else val_best.review_reduction,
        "oracle_threshold": None if oracle is None else oracle.threshold,
        "oracle_review_reduction": None if oracle is None else oracle.review_reduction,
        "oracle_escape_rate": None if oracle is None else oracle.escape_rate,
        "epochs_to_best": max(history, key=lambda h: h["val_review_reduction"])["epoch"],
        "first_epoch_val_reduction": history[0]["val_review_reduction"],
    }
    log(f"\nseed {seed} {ARMS[pretrained]}: review {row['oracle_review_reduction']:.2%} "
        f"@ {row['oracle_escape_rate']:.3%}  (acc {accuracy:.1%}, "
        f"best at epoch {row['epochs_to_best']}, epoch 1 was "
        f"{row['first_epoch_val_reduction']:.1%})")
    return row


def pair_by_seed(rows: list[dict], seeds: list[int]) -> list[dict]:
    """ImageNet minus random, *within* a seed, for the seeds that have both.

    Differencing one arm's median against the other's would compare across
    seeds, and the seed spread on this recipe is 3.1 points with nothing
    changed -- wide enough to invent a difference or hide one. A seed missing an
    arm drops out rather than pairing with a neighbour: a comparison over an
    unstated set of seeds is worse than a shorter one.
    """
    by_seed: dict[int, dict[str, float]] = {}
    for row in rows:
        if row.get("oracle_review_reduction") is None:
            continue
        by_seed.setdefault(row["seed"], {})[row["arm"]] = row["oracle_review_reduction"]
    return [
        {"seed": seed,
         "delta_review_removed": by_seed[seed][ARMS[True]] - by_seed[seed][ARMS[False]]}
        for seed in seeds
        if len(by_seed.get(seed, {})) == len(ARMS)
    ]


def spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "n": len(values)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=Path, default=ROOT / "data" / "patches")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "pretrained_ablation.json")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "docs" / "benchmarks.md")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--budget", type=float, default=DEFAULTS["escape_budget"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    trainval = PatchSet.load(args.patches / "trainval.npz")
    test = PatchSet.load(args.patches / "test.npz")
    device = select_device(args.device)
    print(f"device: {device}")
    print(f"trainval {len(trainval.labels)} / test {len(test.labels)} patches")

    started = time.perf_counter()
    rows = [one_arm(trainval, test, seed, pretrained, args, device, print)
            for seed in args.seeds for pretrained in (True, False)]
    wall = time.perf_counter() - started
    rows = [r for r in rows if r["oracle_review_reduction"] is not None]
    if len(rows) < 2:
        print("too few arms produced a point inside the budget", file=sys.stderr)
        return 1

    by_arm = {name: [r for r in rows if r["arm"] == name] for name in ARMS.values()}
    summary = {name: {
        "review_removed": spread([r["oracle_review_reduction"] for r in arm]),
        "test_accuracy": spread([r["test_accuracy"] for r in arm]),
        "first_epoch_val_reduction": spread([r["first_epoch_val_reduction"] for r in arm]),
    } for name, arm in by_arm.items() if arm}
    paired = pair_by_seed(rows, args.seeds)
    record = {"commit": commit(), "seeds": args.seeds, "epochs": args.epochs,
              "budget": args.budget, "wall_seconds": round(wall, 1),
              "rows": rows, "summary": summary, "paired_deltas": paired}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")

    deltas = [p["delta_review_removed"] for p in paired]
    if not deltas:
        print("no seed produced both arms; nothing to pair", file=sys.stderr)
        return 1
    lines = [
        f"## {date.today().isoformat()} · commit {commit()}",
        "",
        "### Does the ImageNet initialisation buy anything here?",
        "",
        f"`--no-pretrained` has been a switch in `train.py` since the first "
        f"training run and had never been pulled, so the choice of ImageNet "
        f"weights was documented as deliberate and never as measured. Both arms "
        f"over {len(args.seeds)} seeds, the same by-image split within each seed, "
        f"everything else held; each arm read at its own best threshold on the "
        f"test split, so both inherit the same optimism and neither figure is a "
        f"deployment number. {wall / 60:.0f} min. `scripts/pretrained_ablation.py`.",
        "",
        "| seed | init | review removed @ budget | achieved escape | test accuracy | "
        "epoch 1 val review | best epoch |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['arm']} | **{row['oracle_review_reduction']:.2%}** | "
            f"{row['oracle_escape_rate']:.3%} | {row['test_accuracy']:.1%} | "
            f"{row['first_epoch_val_reduction']:.1%} | {row['epochs_to_best']} |")
    for name, stats in summary.items():
        review = stats["review_removed"]
        lines.append(
            f"| **median** | **{name}** | **{review['median']:.2%}** | — | "
            f"{stats['test_accuracy']['median']:.1%} | "
            f"{stats['first_epoch_val_reduction']['median']:.1%} | — |")
    lines += [
        "",
        f"Paired by seed, ImageNet minus random: "
        + ", ".join(f"{d:+.2%}" for d in deltas)
        + f" (median {statistics.median(deltas):+.2%}).",
        "",
        "**The yardstick is the seed spread, not zero.** The 2026-08-31 seed entry "
        "measured this same recipe at 49.73%-52.79% review removed across five "
        "seeds at their own oracles -- 3.1 points wide with nothing changed at "
        "all. A paired difference smaller than that is not a result about the "
        "initialisation, and the pairing above is what makes the comparison "
        "readable at all: within a seed both arms see the same split.",
        "",
        "**What this does not establish.** One dataset, one architecture, one "
        "recipe, one test split, and a fixed 10 epochs -- an initialisation that "
        "only costs convergence time would show up in the epoch-1 column and be "
        "gone by epoch 10, which is why that column is here rather than a "
        "headline. Nothing about a colour or photographic front end, where the "
        "ImageNet features have something to be about.",
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
