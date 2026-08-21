"""Are the false calls real, or are they fragments of real defects?

A differencing detector can split one defect into several blobs. If only the
largest blob clears the IoU threshold, the rest get labelled ``false_call``
even though they sit on top of a genuine defect. That would be a labelling
bug, not a false call, and it would quietly inflate every number downstream.

This script measures the distance from each false call to the nearest ground
truth box so the two populations can be told apart:

  - overlapping or touching a defect  -> fragment, a labelling artefact
  - far from every defect             -> genuine false call
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.aoi.matching import iou, match  # noqa: E402
from aoi_agent.aoi.simulator import DetectorConfig, detect  # noqa: E402
from aoi_agent.data.deeppcb import load_split  # noqa: E402


def box_gap(a, b) -> float:
    """Euclidean gap between two boxes; 0 when they overlap."""
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return float(np.hypot(dx, dy))


def main() -> None:
    pairs = load_split("trainval")[:100]
    config = DetectorConfig()

    buckets: Counter[str] = Counter()
    areas: dict[str, list[int]] = {"fragment": [], "near": [], "genuine": []}
    per_image_genuine: list[int] = []

    for pair in pairs:
        annotations = pair.load_annotations()
        candidates = detect(pair.load_template(), pair.load_test(), config)
        result = match(candidates, annotations)

        genuine_here = 0
        for labelled in result.false_calls:
            box = labelled.candidate.box
            gap = min((box_gap(box, a.box) for a in annotations), default=1e9)
            best = max((iou(box, a.box) for a in annotations), default=0.0)

            if best > 0 or gap == 0:
                kind = "fragment"      # touches or overlaps a real defect
            elif gap <= 20:
                kind = "near"          # in the halo around a defect
            else:
                kind = "genuine"       # nowhere near anything real
                genuine_here += 1

            buckets[kind] += 1
            areas[kind].append(labelled.candidate.area)

        per_image_genuine.append(genuine_here)

    total = sum(buckets.values())
    print(f"false calls over {len(pairs)} images: {total}\n")
    print(f"{'kind':<10} {'count':>7} {'share':>8} {'median area':>12}")
    print("-" * 40)
    for kind in ("fragment", "near", "genuine"):
        count = buckets[kind]
        median = int(np.median(areas[kind])) if areas[kind] else 0
        print(f"{kind:<10} {count:>7} {count / total:>7.1%} {median:>12}")

    print()
    print(f"genuine false calls per image: mean {np.mean(per_image_genuine):.2f}, "
          f"median {np.median(per_image_genuine):.1f}, max {max(per_image_genuine)}")
    print(f"images with zero genuine false calls: "
          f"{sum(1 for n in per_image_genuine if n == 0)}/{len(pairs)}")


if __name__ == "__main__":
    main()
