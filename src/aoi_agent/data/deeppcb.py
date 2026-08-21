"""DeepPCB dataset access.

Layout on disk (after `git clone` into ``data/DeepPCB``)::

    PCBData/
        trainval.txt                  999 lines, one pair per line
        test.txt                      499 lines
        group12000/
            12000/12000001_test.jpg   tested image (has defects)
            12000/12000001_temp.jpg   defect-free template, already aligned
            12000_not/12000001.txt    annotations, one per line

Each split line names the image stem and the annotation file, e.g.::

    group20085/20085/20085000.jpg group20085/20085_not/20085000.txt

Note the ``.jpg`` in the split file is a *stem* — the real files carry
``_test`` / ``_temp`` suffixes.

Annotations are space separated ``x1 y1 x2 y2 type`` with the top-left and
bottom-right corner of the defect, and an integer class id.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# 0 is background and unused by the dataset.
CLASS_NAMES: dict[int, str] = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "pin-hole",
}

FALSE_CALL = "false_call"

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "DeepPCB" / "PCBData"


@dataclass(frozen=True)
class Annotation:
    """One ground-truth defect box."""

    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]


@dataclass(frozen=True)
class Pair:
    """A template/tested image pair and its ground truth."""

    stem: str
    template_path: Path
    test_path: Path
    annotation_path: Path

    def load_template(self) -> np.ndarray:
        return _load_gray(self.template_path)

    def load_test(self) -> np.ndarray:
        return _load_gray(self.test_path)

    def load_annotations(self) -> list[Annotation]:
        rows = []
        for line in self.annotation_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            x1, y1, x2, y2, class_id = (int(v) for v in line.split())
            rows.append(Annotation(x1, y1, x2, y2, class_id))
        return rows


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("L"))


def load_split(split: str, root: Path | None = None) -> list[Pair]:
    """Load the official ``trainval`` or ``test`` split.

    The official split is used as-is so results stay comparable with the
    numbers published alongside the dataset.
    """
    if split not in {"trainval", "test"}:
        raise ValueError(f"split must be 'trainval' or 'test', got {split!r}")

    root = root or DEFAULT_ROOT
    split_file = root / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(
            f"{split_file} not found. Clone the dataset first:\n"
            "  git clone --depth 1 https://github.com/tangsanli5201/DeepPCB.git data/DeepPCB"
        )

    pairs = []
    for line in split_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        image_field, annotation_field = line.split()
        stem_path = root / image_field
        stem = stem_path.stem
        pairs.append(
            Pair(
                stem=stem,
                template_path=stem_path.with_name(f"{stem}_temp.jpg"),
                test_path=stem_path.with_name(f"{stem}_test.jpg"),
                annotation_path=root / annotation_field,
            )
        )
    return pairs
