"""Build the re-verification training set from AOI candidates.

Runs the AOI simulator over a split, labels every candidate against ground
truth, and writes the resulting patches to ``data/patches/<split>.npz``.

Usage::

    uv run python scripts/build_patches.py --split trainval
    uv run python scripts/build_patches.py --split test
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.aoi.matching import IOU_THRESHOLD, match  # noqa: E402
from aoi_agent.aoi.simulator import DetectorConfig, detect  # noqa: E402
from aoi_agent.data.deeppcb import CLASS_NAMES, FALSE_CALL, load_split  # noqa: E402
from aoi_agent.vision.patches import PATCH_SIZE, PatchSet, patches_for_image  # noqa: E402

LABEL_NAMES = [FALSE_CALL, *CLASS_NAMES.values()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="trainval", choices=["trainval", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--size", type=int, default=PATCH_SIZE)
    parser.add_argument("--out-dir", type=Path, default=Path("data/patches"))
    args = parser.parse_args()

    pairs = load_split(args.split)
    if args.limit:
        pairs = pairs[: args.limit]

    config = DetectorConfig(threshold=args.threshold, register=True)
    label_index = {name: i for i, name in enumerate(LABEL_NAMES)}

    all_patches: list[np.ndarray] = []
    all_labels: list[int] = []
    all_image_index: list[int] = []
    all_boxes: list[tuple[int, int, int, int]] = []
    missed_total = 0
    annotation_total = 0

    started = time.perf_counter()
    for index, pair in enumerate(pairs):
        template = pair.load_template()
        test = pair.load_test()
        annotations = pair.load_annotations()

        candidates = detect(template, test, config)
        result = match(candidates, annotations)

        patches, labels, boxes = patches_for_image(
            template, test, result.labelled, args.size
        )
        all_patches.extend(patches)
        all_labels.extend(label_index[name] for name in labels)
        all_image_index.extend([index] * len(patches))
        all_boxes.extend(boxes)

        annotation_total += len(annotations)
        missed_total += len(result.missed_annotations)

        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(pairs)} images, {len(all_labels)} patches")

    elapsed = time.perf_counter() - started

    patch_set = PatchSet(
        patches=np.stack(all_patches).astype(np.uint8),
        labels=np.array(all_labels, dtype=np.int64),
        label_names=LABEL_NAMES,
        image_index=np.array(all_image_index, dtype=np.int64),
        boxes=np.array(all_boxes, dtype=np.int64),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.split}.npz"
    patch_set.save(out)

    counts = Counter(patch_set.label_names[i] for i in patch_set.labels)
    total = len(patch_set)

    print(f"\n{args.split}: {len(pairs)} images -> {total} patches in {elapsed:.1f}s")
    print(f"{'label':<12} {'count':>7} {'share':>8}")
    print("-" * 30)
    for name in LABEL_NAMES:
        print(f"{name:<12} {counts[name]:>7} {counts[name] / total:>7.1%}")
    print()
    print(f"unmatched at IoU {IOU_THRESHOLD}: {missed_total}/{annotation_total} "
          f"({missed_total / annotation_total:.1%})")
    print("  A box-tightness figure, not an escape rate. Most of these defects "
          "have a candidate sitting on them that simply drew a looser box; it is "
          "labelled `fragment` and held out of the patch set, which is why they "
          "read as missed here. This line was quoted as a 5.0% AOI escape rate "
          "until 2026-08-23, when the same split measured 0.22%. For the escape "
          "rate run scripts/escape_accounting.py.")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
