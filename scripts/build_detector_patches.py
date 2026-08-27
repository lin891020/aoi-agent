"""Turn the detector's boxes on PCB-AoI into re-verifier patches.

The detector front end localises and does not discriminate: at the escape
budget its confidence removes 1.2% of the queue. What both transfer datasets
pointed at is a re-verifier over the detector's crops -- the ResNet-18 already
here, minus the template channel, because there is no template to difference
against on a solder-paste line. This script builds that model's training set
in the same ``PatchSet`` schema ``scripts/train.py`` already reads, so the
training loop, the by-image split and the operating-point sweep are reused
unchanged::

    uv run python scripts/build_detector_patches.py                # -> data/patches_pcbaoi/
    uv run python scripts/train.py --patches data/patches_pcbaoi --out models/pcbaoi_reverifier
    uv run python scripts/crop_reverifier_report.py --dry-run

Three things this set is, said up front because each one bounds the result.

* **The three channels are RGB, not template/test/difference.** A patch is a
  64 px window centred on the detector's box in the inspected image, and
  nothing else. The model sees the same context the detector saw and has to
  tell a defect from a plausible false call on appearance alone.
* **The training candidates come from a detector that has seen the training
  images.** There is one trained detector and it was trained on ``train_data``;
  its boxes on those images are in-sample -- fewer and more confident false
  calls than it draws on unseen boards. The crop re-verifier is therefore
  trained on an easier candidate distribution than it is tested on, and the
  test figure is the one to read.
* **``image_index`` is the base stem, not the file.** The augmentation set is
  six transforms of each training capture; a split by file would put a
  board's rotation in validation while its original trains, which is the leak
  ``split_by_image`` exists to prevent. Grouping by base stem makes that split
  hold out whole boards.

Beside ``test.npz`` a sidecar ``test_detector.npz`` keeps, per test patch, the
detector's own ``P(false_call)`` and, per image, how many annotated defects
the detector never boxed -- so the report can put the crop re-verifier's
ordering and the detector's on the same candidates, and state the unflagged
defects above both, where the sweep cannot see them.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.aoi.matching import match  # noqa: E402
from aoi_agent.data import pcbaoi  # noqa: E402
from aoi_agent.vision.detector import CONF_FLOOR, DEFAULT_CHECKPOINT, Detector  # noqa: E402
from aoi_agent.vision.patches import PATCH_SIZE, PatchSet, crop  # noqa: E402

FALSE_CALL = "false_call"
OUT = Path("data/patches_pcbaoi")


def rgb_patch(image: np.ndarray, candidate, size: int = PATCH_SIZE) -> np.ndarray:
    """A ``(3, size, size)`` uint8 window centred on the candidate, one channel
    per colour plane -- ``crop`` pads with the image's own edge, so a box at the
    border gets no artificial frame. The channel order is the image's, which is
    the only order that means anything without a template."""
    return np.stack([crop(image[..., c], candidate, size) for c in range(3)], axis=0)


def build(detector: Detector, items: list[pcbaoi.Item], floor: float) -> tuple[PatchSet, dict]:
    classes = list(pcbaoi.CLASS_NAMES) + [FALSE_CALL]
    groups = {stem: i for i, stem in enumerate(sorted({item.base_stem for item in items}))}
    patches, labels, image_index, boxes, detector_pfc = [], [], [], [], []
    per_image_candidates, unflagged, defects_total = [], Counter(), Counter()
    fragments = 0  # boxes that split a defect: neither a defect nor a false call, held out
    for item in items:
        image = np.array(Image.open(item.image_path).convert("RGB"))
        annotations = item.load_boxes()
        candidates = detector.detect(image, floor=floor)
        result = match(candidates, annotations)
        per_image_candidates.append(len(candidates))
        for box in annotations:
            defects_total[box.class_name] += 1
        for box in result.missed_annotations:
            unflagged[box.class_name] += 1
        for labelled, candidate in zip(result.labelled, candidates, strict=True):
            if not labelled.trainable:
                fragments += 1  # a fragment of a defect is neither a defect nor a false call
                continue
            target = labelled.matched_annotation
            patches.append(rgb_patch(image, candidate))
            labels.append(classes.index(target.class_name if target is not None else FALSE_CALL))
            image_index.append(groups[item.base_stem])
            boxes.append(candidate.box)
            detector_pfc.append(candidate.false_call_probability)
    patch_set = PatchSet(
        patches=np.stack(patches).astype(np.uint8),
        labels=np.array(labels, dtype=np.int64),
        label_names=classes,
        image_index=np.array(image_index, dtype=np.int64),
        boxes=np.array(boxes, dtype=np.int64),
    )
    sidecar = {
        "detector_false_call_probability": np.array(detector_pfc, dtype=np.float64),
        "candidates_per_image": np.array(per_image_candidates, dtype=np.int64),
        "unflagged_by_class": np.array([unflagged[c] for c in pcbaoi.CLASS_NAMES], dtype=np.int64),
        "defects_by_class": np.array([defects_total[c] for c in pcbaoi.CLASS_NAMES], dtype=np.int64),
        "class_names": np.array(list(pcbaoi.CLASS_NAMES)),
        "floor": np.array(floor),
        "fragments": np.array(fragments),
    }
    return patch_set, sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--floor", type=float, default=CONF_FLOOR)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    detector = Detector(args.checkpoint, device=args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    trainval = pcbaoi.load("train_data") + pcbaoi.load("train_data_augmentation")
    test = pcbaoi.load("test_data")
    if pcbaoi.leaks(trainval, test):
        raise SystemExit("a test base stem appears in the training set; refusing to build")

    for name, items in (("trainval", trainval), ("test", test)):
        patch_set, sidecar = build(detector, items, args.floor)
        patch_set.save(args.out / f"{name}.npz")
        np.savez_compressed(args.out / f"{name}_detector.npz", **sidecar)
        counts = Counter(patch_set.label_names[i] for i in patch_set.labels)
        print(
            f"{name}: {len(items)} images -> {len(patch_set)} patches "
            f"({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}); "
            f"{int(sidecar['unflagged_by_class'].sum())} of "
            f"{int(sidecar['defects_by_class'].sum())} defects never boxed at floor {args.floor}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
