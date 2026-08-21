"""Train the AOI re-verification model.

Usage::

    uv run python scripts/train.py
    uv run python scripts/train.py --epochs 12 --no-pretrained
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.vision.dataset import CandidateDataset, class_weights  # noqa: E402
from aoi_agent.vision.model import build_model, select_device  # noqa: E402
from aoi_agent.vision.operating_point import (  # noqa: E402
    best_at_escape_budget,
    sweep,
)
from aoi_agent.vision.patches import PatchSet  # noqa: E402


def split_by_image(patch_set: PatchSet, val_fraction: float, seed: int):
    """Hold out whole images, never individual patches.

    Patches cropped from the same board share lighting, registration error and
    often the same defect, so splitting at patch level would leak the
    validation set into training and inflate every number.
    """
    images = np.unique(patch_set.image_index)
    rng = np.random.default_rng(seed)
    rng.shuffle(images)
    cut = int(len(images) * (1 - val_fraction))
    train_images, val_images = set(images[:cut]), set(images[cut:])

    train_idx = [i for i, img in enumerate(patch_set.image_index) if img in train_images]
    val_idx = [i for i, img in enumerate(patch_set.image_index) if img in val_images]
    return train_idx, val_idx


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return per-class probabilities and true labels."""
    model.eval()
    probabilities, labels = [], []
    for batch, target in loader:
        logits = model(batch.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(target.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=Path, default=Path("data/patches"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--escape-budget", type=float, default=0.005)
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_set = PatchSet.load(args.patches / "trainval.npz")
    test_set = PatchSet.load(args.patches / "test.npz")
    label_names = train_set.label_names
    false_call_index = label_names.index("false_call")

    train_idx, val_idx = split_by_image(train_set, args.val_fraction, args.seed)
    train_data = Subset(CandidateDataset(train_set, augment=True), train_idx)
    val_data = Subset(CandidateDataset(train_set, augment=False), val_idx)
    test_data = CandidateDataset(test_set, augment=False)

    device = select_device(args.device)
    print(f"device: {device}")
    print(f"train {len(train_data)} / val {len(val_data)} / test {len(test_data)} patches")
    print(f"classes: {label_names}\n")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256)
    test_loader = DataLoader(test_data, batch_size=256)

    model = build_model(len(label_names), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_set).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_reduction = -1.0
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "reverifier.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        running = 0.0
        for batch, target in train_loader:
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), target)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(target)
        scheduler.step()
        train_loss = running / len(train_data)

        probabilities, labels = predict(model, val_loader, device)
        accuracy = float((probabilities.argmax(1) == labels).mean())
        points = sweep(probabilities[:, false_call_index], labels, false_call_index)
        best = best_at_escape_budget(points, args.escape_budget)
        reduction = best.review_reduction if best else 0.0

        elapsed = time.perf_counter() - started
        print(
            f"epoch {epoch:>2}/{args.epochs}  loss {train_loss:.4f}  "
            f"val acc {accuracy:.1%}  "
            f"review -{reduction:.1%} @ escape<={args.escape_budget:.1%}  "
            f"({elapsed:.0f}s)"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": accuracy,
                "val_review_reduction": reduction,
                "seconds": elapsed,
            }
        )

        if reduction > best_reduction:
            best_reduction = reduction
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "label_names": label_names,
                    "epoch": epoch,
                    "pretrained": not args.no_pretrained,
                },
                checkpoint,
            )

    print(f"\nbest validation review reduction {best_reduction:.1%} -> {checkpoint}")

    model.load_state_dict(torch.load(checkpoint)["state_dict"])
    probabilities, labels = predict(model, test_loader, device)
    np.savez_compressed(
        args.out / "test_predictions.npz",
        probabilities=probabilities,
        labels=labels,
        label_names=np.array(label_names),
    )

    accuracy = float((probabilities.argmax(1) == labels).mean())
    print(f"\nheld-out test accuracy: {accuracy:.1%}")
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print("run `uv run python scripts/report.py` for the operating-point table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
