"""Train the detector front end on PCB-AoI.

YOLO26n, from the released weights, on the dataset's augmentation set with a
by-stem validation split; the 60 test images are never read here. Writes
``models/detector_pcbaoi.pt`` and ``models/detector_history.json``, and
refuses to start beside a busy GPU -- a training run taken next to another
torch job on this fanless machine produces a wall time that describes the
contention rather than the model, which is the failure the latency skill
exists to catch.

Not part of `retraining-the-reverifier`'s chain: this trains a different
model on a different dataset and carries no threshold across. Its only gate
is the report, `scripts/detector_report.py`, which reads the checkpoint this
writes.

Run:
    uv run python scripts/train_detector.py                  # ~20-40 min on the M5 Air (MPS)
    uv run python scripts/train_detector.py --epochs 1 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aoi_agent.data import pcbaoi  # noqa: E402
from aoi_agent.vision.detector import DEFAULT_CHECKPOINT  # noqa: E402

BASE_WEIGHTS = Path("models/yolo26n.pt")
DEFAULT_HISTORY = Path("models/detector_history.json")


def history_path(out: Path) -> Path:
    """Where this run's record goes.

    A second run at another input size is a second model, and writing both
    records to one path leaves the surviving checkpoint described by whichever
    run finished last. The default checkpoint keeps the original filename so
    nothing that reads it has to change; anything else carries its own.
    """
    # resolved, or a relative path off the command line never equals the
    # absolute default and the record silently becomes a row of question marks
    if out.resolve() == DEFAULT_CHECKPOINT.resolve():
        return DEFAULT_HISTORY
    return out.with_name(out.stem + "_history.json")


def restrict(items: list[pcbaoi.Item], stems_file: Path) -> list[pcbaoi.Item]:
    """The items whose board is named in ``stems_file``, and a refusal if any is not.

    A stem that matches nothing is a typo or a stale list, and training on the
    silently smaller remainder would produce a detector describing a fold
    nobody chose.
    """
    wanted = {line.strip() for line in stems_file.read_text().splitlines() if line.strip()}
    kept = [item for item in items if item.base_stem in wanted]
    missing = wanted - {item.base_stem for item in kept}
    if missing:
        raise SystemExit(f"{stems_file}: {len(missing)} base stem(s) match no image, "
                         f"first {sorted(missing)[:3]}")
    return kept


def machine_is_quiet() -> list[str]:
    """What is competing for the GPU, by the latency skill's own check.

    The parent is excluded along with this process: a report script that runs
    this one as a step -- `scripts/detector_fold_report.py` -- sits in the
    process table at whatever CPU it spent getting here, and a guard that
    counts its own caller refuses every run it is part of. Anything else busy
    still stops the training, which is the failure the guard is for.
    """
    try:
        from reverifier_latency import competing_processes, ollama_ps, process_table, resident_models

        table = process_table()
        busy = competing_processes(table, os.getpid())
        parent = str(os.getppid())
        return resident_models(ollama_ps()) + [
            line for line in busy if line.split(None, 1)[0] != parent]
    except ImportError:  # pragma: no cover
        return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--out", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--name", default=None,
                        help="ultralytics run directory; defaults to the checkpoint's stem, "
                             "so a second run does not overwrite the first's curves")
    parser.add_argument("--base-stems", type=Path, default=None,
                        help="a file of base stems, one per line: train on these boards only. "
                             "For a fold experiment that needs a detector which has not seen "
                             "the rest; the split inside them is this script's usual one")
    parser.add_argument("--dry-run", action="store_true", help="export the split and stop")
    parser.add_argument("--ignore-contention", action="store_true")
    args = parser.parse_args()

    contended = machine_is_quiet()
    if contended and not args.ignore_contention:
        print("refusing to train beside: " + "; ".join(contended), file=sys.stderr)
        print("the wall time would measure the contention, not the model", file=sys.stderr)
        return 2

    items = pcbaoi.load("train_data_augmentation")
    if args.base_stems is not None:
        items = restrict(items, args.base_stems)
    train, val = pcbaoi.split_by_stem(items, seed=args.seed)
    test = pcbaoi.load("test_data")
    assert not pcbaoi.leaks(train, val) and not pcbaoi.leaks(train + val, test)
    yaml = pcbaoi.export(train, val, test)
    print(f"exported {len(train)} train / {len(val)} val images ({len({i.base_stem for i in val})} held-out boards) "
          f"and {len(test)} test -> {yaml}")
    if args.dry_run:
        return 0

    if not BASE_WEIGHTS.exists():
        print(f"{BASE_WEIGHTS} not found; ultralytics will download yolo26n.pt into cwd", file=sys.stderr)

    from ultralytics import YOLO

    model = YOLO(str(BASE_WEIGHTS) if BASE_WEIGHTS.exists() else "yolo26n.pt")
    started = time.time()
    results = model.train(
        data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, seed=args.seed, deterministic=True,
        project="models/detector_runs",
        name=args.name or ("pcbaoi" if args.out == DEFAULT_CHECKPOINT else args.out.stem),
        exist_ok=True,
        verbose=False, plots=False,
    )
    wall = time.time() - started

    best = Path(results.save_dir) / "weights" / "best.pt"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.out)

    history = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "base_weights": str(BASE_WEIGHTS),
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "device": args.device, "seed": args.seed,
        "train_images": len(train), "val_images": len(val),
        "base_stems": str(args.base_stems) if args.base_stems else None,
        "val_boards": len({i.base_stem for i in val}),
        "wall_seconds": round(wall, 1),
        "contended": contended,
        "checkpoint": str(args.out),
    }
    record = history_path(args.out)
    record.write_text(json.dumps(history, indent=2))
    print(f"\nwrote {args.out} after {wall / 60:.1f} min; history -> {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
