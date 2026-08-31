"""Choose the dismissal threshold by cross-validation, never on the test split.

The threshold this project shipped until 2026-08-31 was swept on the test
predictions. As a comparison between engines that is fair -- each gets its own
oracle threshold -- but as a deployment number it has seen the answers.

Choosing on the single validation split instead is honest and, measured, not
enough: 969 defects make one defect worth 0.10%, so the point estimate is
granular and noisy. Its choice (0.610) escapes 0.93% on test, nearly twice the
budget; the interval-upper-bound choice on the same split needs zero escapes in
969 and costs twenty points of review. Both readings are in docs/benchmarks.md.

So the threshold is chosen on **out-of-fold** predictions over the whole
trainval set: k folds by image, each fold predicted by a model that never saw
it, pooled into one sweep with roughly three thousand defects behind it. Epoch
selection happens on an inner split of each fold's own training images, so the
held-out fold leaks into nothing.

    uv run python scripts/threshold_cv.py                 # ~9 min on the M5 Air
    uv run python scripts/threshold_cv.py --folds 5 --budget 0.005

Writes `models/oof_predictions.npz` and `models/cv_threshold.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aoi_agent.stats import wilson  # noqa: E402
from aoi_agent.vision.dataset import CandidateDataset  # noqa: E402
from aoi_agent.vision.model import select_device  # noqa: E402
from aoi_agent.vision.operating_point import OperatingPoint, sweep  # noqa: E402
from aoi_agent.vision.patches import PatchSet  # noqa: E402
from train import DEFAULTS, fit, predict  # noqa: E402


def folds_by_image(patch_set: PatchSet, folds: int, seed: int) -> list[np.ndarray]:
    """Whole images per fold, for the reason `split_by_image` gives."""
    images = np.unique(patch_set.image_index)
    rng = np.random.default_rng(seed)
    rng.shuffle(images)
    return [np.asarray(part) for part in np.array_split(images, folds)]


def indices_for(patch_set: PatchSet, images: set) -> list[int]:
    return [i for i, img in enumerate(patch_set.image_index) if img in images]


def inner_split(images: np.ndarray, val_fraction: float, seed: int):
    """The same by-image hold-out `train.py` uses, over a subset of images."""
    shuffled = images.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - val_fraction))
    return set(shuffled[:cut]), set(shuffled[cut:])


def choose(points: list[OperatingPoint], budget: float) -> tuple[OperatingPoint | None,
                                                                 OperatingPoint | None]:
    """(point-estimate choice, upper-bound choice) at this budget.

    The second is the deployable one: a budget is a promise about defects nobody
    has seen yet, so the rate has to clear it with its interval, not with its
    point estimate on the sample that chose it.
    """
    within = [p for p in points if p.escape_rate <= budget]
    optimistic = max(within, key=lambda p: p.review_reduction) if within else None
    guarded = [p for p in points if wilson(p.escapes, p.defects_total)[1] <= budget]
    conservative = max(guarded, key=lambda p: p.review_reduction) if guarded else None
    return optimistic, conservative


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=Path, default=ROOT / "data" / "patches")
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--budget", type=float, default=DEFAULTS["escape_budget"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    trainval = PatchSet.load(args.patches / "trainval.npz")
    label_names = trainval.label_names
    false_call_index = label_names.index("false_call")
    device = select_device(args.device)
    parts = folds_by_image(trainval, args.folds, args.seed)
    print(f"device {device}; {len(np.unique(trainval.image_index))} images "
          f"in {args.folds} folds of {[len(p) for p in parts]}")

    oof_probabilities = np.zeros((len(trainval.labels), len(label_names)), dtype=np.float32)
    covered = np.zeros(len(trainval.labels), dtype=bool)
    started = time.perf_counter()

    for k, held_out in enumerate(parts, 1):
        rest = np.array([i for i in np.unique(trainval.image_index) if i not in set(held_out)])
        inner_train, inner_val = inner_split(rest, DEFAULTS["val_fraction"], args.seed)
        train_idx = indices_for(trainval, inner_train)
        val_idx = indices_for(trainval, inner_val)
        out_idx = indices_for(trainval, set(held_out))
        print(f"\nfold {k}/{args.folds}: train {len(train_idx)} / inner val {len(val_idx)} "
              f"/ held out {len(out_idx)} patches")

        model, _history, _best = fit(
            Subset(CandidateDataset(trainval, augment=True), train_idx),
            Subset(CandidateDataset(trainval, augment=False), val_idx),
            trainval, label_names,
            epochs=args.epochs, batch_size=DEFAULTS["batch_size"], lr=DEFAULTS["lr"],
            seed=args.seed, device=device, escape_budget=args.budget,
            log=lambda line: print("  " + line),
        )
        loader = DataLoader(Subset(CandidateDataset(trainval, augment=False), out_idx),
                            batch_size=256)
        probabilities, _labels = predict(model, loader, device)
        oof_probabilities[out_idx] = probabilities
        covered[out_idx] = True
        del model

    assert covered.all(), "every candidate must be predicted by a model that did not train on it"
    labels = trainval.labels
    points = sweep(oof_probabilities[:, false_call_index], labels, false_call_index)
    optimistic, conservative = choose(points, args.budget)
    wall = time.perf_counter() - started

    print(f"\nout-of-fold: {len(labels)} candidates, "
          f"{int((labels != false_call_index).sum())} defects, {wall / 60:.1f} min")
    for name, point in (("point estimate", optimistic), ("interval upper bound", conservative)):
        if point is None:
            print(f"  {name:22} no threshold meets ≤{args.budget:.2%}")
            continue
        lo, hi = wilson(point.escapes, point.defects_total)
        print(f"  {name:22} thr {point.threshold:.4f}  escape {point.escape_rate:.3%} "
              f"({point.escapes}/{point.defects_total}, 95% {lo:.2%}-{hi:.2%})  "
              f"review {point.review_reduction:.2%}")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "oof_predictions.npz", probabilities=oof_probabilities,
                        labels=labels, label_names=np.array(label_names))
    record = {
        "folds": args.folds, "epochs": args.epochs, "seed": args.seed,
        "budget": args.budget, "wall_seconds": round(wall, 1),
        "oof_candidates": int(len(labels)),
        "oof_defects": int((labels != false_call_index).sum()),
    }
    for name, point in (("point_estimate", optimistic), ("upper_bound", conservative)):
        record[name] = None if point is None else {
            "threshold": point.threshold, "escape_rate": point.escape_rate,
            "escapes": int(point.escapes), "defects_total": int(point.defects_total),
            "review_reduction": point.review_reduction,
        }
    (args.out / "cv_threshold.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {(args.out / 'cv_threshold.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
