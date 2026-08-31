"""Derive the dismissal threshold on the validation split, not the test split.

`scripts/report.py` sweeps the test predictions and reports the best threshold
each escape budget can reach. As a comparison between engines that is fair --
every engine gets its own oracle threshold. As a *deployment* number it is
optimistic, because the threshold has seen the answers.

The split this reads is the one `train.py` already trains against: whole
images held out, `--val-fraction 0.15`, seeded, so it rebuilds exactly without
retraining. That matters -- retraining to recover a threshold would move the
weights and the threshold together, and the comparison would have two
variables.

    uv run python scripts/validation_threshold.py
    uv run python scripts/validation_threshold.py --budget 0.0025

Writes `models/val_predictions.npz` and `models/val_threshold.json`; prints the
threshold `vision/inference.py` should carry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aoi_agent.vision.dataset import CandidateDataset  # noqa: E402
from aoi_agent.vision.model import build_model, select_device  # noqa: E402
from aoi_agent.vision.operating_point import best_at_escape_budget, sweep  # noqa: E402
from aoi_agent.vision.patches import PatchSet  # noqa: E402
from train import predict, split_by_image  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=Path, default=ROOT / "data" / "patches")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "reverifier.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=float, default=0.005)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"{args.checkpoint} not found; train first", file=sys.stderr)
        return 2

    trainval = PatchSet.load(args.patches / "trainval.npz")
    label_names = trainval.label_names
    false_call_index = label_names.index("false_call")
    _, val_idx = split_by_image(trainval, args.val_fraction, args.seed)
    val_data = Subset(CandidateDataset(trainval, augment=False), val_idx)
    loader = DataLoader(val_data, batch_size=256)

    device = select_device(args.device)
    saved = torch.load(args.checkpoint, map_location=device)
    model = build_model(len(label_names)).to(device)
    model.load_state_dict(saved["state_dict"])

    probabilities, labels = predict(model, loader, device)
    p_false_call = probabilities[:, false_call_index]
    points = sweep(p_false_call, labels, false_call_index)
    best = best_at_escape_budget(points, args.budget)
    if best is None:
        print(f"no threshold meets a ≤{args.budget:.2%} budget on validation", file=sys.stderr)
        return 1

    boards = len(np.unique(trainval.image_index[val_idx]))
    print(f"validation split: {len(labels)} candidates from {boards} held-out images "
          f"({int((labels != false_call_index).sum())} defects)")
    print(f"threshold at ≤{args.budget:.2%}: {best.threshold:.4f}")
    print(f"  validation escape rate {best.escape_rate:.2%} "
          f"({best.escapes}/{best.defects_total}), review removed {best.review_reduction:.1%}")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "val_predictions.npz", probabilities=probabilities,
                        labels=labels, label_names=np.array(label_names))
    (args.out / "val_threshold.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "budget": args.budget,
        "threshold": best.threshold,
        "val_candidates": int(len(labels)),
        "val_images": int(boards),
        "val_escape_rate": best.escape_rate,
        "val_escapes": int(best.escapes),
        "val_defects": int(best.defects_total),
        "val_review_reduction": best.review_reduction,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
    }, indent=2) + "\n")
    print(f"\nwrote {(args.out / 'val_threshold.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
