"""PCB-AoI, the dataset the detector front end is proven on.

Real solder-paste inspection images from a line (KubeEdge-Ianvs, 2016–2018):
600x600 RGB, two classes, a median box 17 px long, and **no template** --
placement has tolerance, so pixel differencing would flag every component.
On this data a detector is not a preference; it is the only front end that
can exist. See docs/superpowers/specs/2026-08-26-detector-front-end-design.md.

This module does three things and writes nothing the repository tracks:

* reads the VOC XML the dataset ships as boxes, in this project's own
  ``Annotation``-shaped record, under the dataset's own class names --
  ``Bad_podu`` and ``Bad_qiaojiao`` are not DeepPCB's classes and are not
  mapped onto them;
* splits by **stem**, never by image: the augmentation set is six transforms
  of each original, and a transform of a validation board in the training set
  is the same board leaking across the boundary that the DeepPCB split rule
  exists to prevent;
* exports the split in YOLO's layout under ``data/pcbaoi_yolo/`` (gitignored,
  rebuilt by `scripts/train_detector.py`), because that is the only layout
  the trainer reads.

The test set is the dataset's own 60 images and is touched once, by the
report.
"""

from __future__ import annotations

import random
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "PCB-AoI"
DEFAULT_EXPORT = Path(__file__).resolve().parents[3] / "data" / "pcbaoi_yolo"

#: The dataset's classes, in the index order the detector is trained with.
CLASS_NAMES: tuple[str, ...] = ("Bad_podu", "Bad_qiaojiao")

#: The suffixes the augmentation set adds to a stem. Anything after the last
#: underscore that is one of these marks a transform of the base stem.
AUGMENTATION_SUFFIXES = frozenset({"90", "180", "270", "shuiping", "suofang", "shuzhi"})

_STEM = re.compile(r"^(?P<base>\d{8}-SPI-AOI-\d+)(?:_(?P<suffix>[A-Za-z0-9]+))?$")


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class Item:
    """One image and its boxes."""

    stem: str
    image_path: Path
    annotation_path: Path

    @property
    def base_stem(self) -> str:
        """The original capture this image is, or is a transform of."""
        return base_stem(self.stem)

    def load_boxes(self) -> list[Box]:
        root = ET.parse(self.annotation_path).getroot()
        out = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name not in CLASS_NAMES:
                raise ValueError(f"{self.annotation_path}: unknown class {name!r}")
            b = obj.find("bndbox")
            x1, y1, x2, y2 = (int(round(float(b.findtext(k)))) for k in ("xmin", "ymin", "xmax", "ymax"))
            out.append(Box(x1, y1, x2, y2, name))
        return out

    def image_size(self) -> tuple[int, int]:
        root = ET.parse(self.annotation_path).getroot()
        return int(root.findtext("size/width")), int(root.findtext("size/height"))


def base_stem(stem: str) -> str:
    m = _STEM.match(stem)
    if not m:
        raise ValueError(f"not a PCB-AoI stem: {stem!r}")
    suffix = m.group("suffix")
    return m.group("base") if suffix is None or suffix in AUGMENTATION_SUFFIXES else stem


def _items(root: Path, subset: str) -> list[Item]:
    folder = root / subset
    index = folder / "index.txt"
    if not index.exists():
        raise FileNotFoundError(
            f"{index} not found. Download the dataset first:\n"
            "  kaggle datasets download -d kubeedgeianvs/pcb-aoi -p data/PCB-AoI --unzip"
        )
    out = []
    for line in index.read_text().splitlines():
        if not line.strip():
            continue
        image_rel, annotation_rel = line.split()
        image_path = folder / image_rel
        out.append(Item(stem=image_path.stem, image_path=image_path,
                        annotation_path=folder / annotation_rel))
    return out


def load(subset: str, root: Path | None = None) -> list[Item]:
    """``train_data``, ``train_data_augmentation`` or ``test_data``, as shipped."""
    if subset not in ("train_data", "train_data_augmentation", "test_data"):
        raise ValueError(f"unknown subset {subset!r}")
    return _items(root or DEFAULT_ROOT, subset)


def split_by_stem(
    items: list[Item], validation_share: float = 0.2, seed: int = 20260826
) -> tuple[list[Item], list[Item]]:
    """Train/validation, with every transform of a base stem on one side.

    The share is of *base stems*, not of images, so the validation set holds
    whole boards. Deterministic for a seed, so a retrain sees the same split.
    """
    bases = sorted({item.base_stem for item in items})
    rng = random.Random(seed)
    rng.shuffle(bases)
    held = set(bases[: int(round(len(bases) * validation_share))])
    train = [i for i in items if i.base_stem not in held]
    val = [i for i in items if i.base_stem in held]
    return train, val


def leaks(a: list[Item], b: list[Item]) -> set[str]:
    """Base stems present on both sides. Empty is the only acceptable answer."""
    return {i.base_stem for i in a} & {i.base_stem for i in b}


def to_yolo_line(box: Box, width: int, height: int) -> str:
    """``class cx cy w h``, normalised, the way YOLO reads a label."""
    cx = (box.x1 + box.x2) / 2 / width
    cy = (box.y1 + box.y2) / 2 / height
    w = (box.x2 - box.x1) / width
    h = (box.y2 - box.y1) / height
    return f"{CLASS_NAMES.index(box.class_name)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def from_yolo_line(line: str, width: int, height: int) -> Box:
    idx, cx, cy, w, h = line.split()
    cx, cy, w, h = (float(v) for v in (cx, cy, w, h))
    return Box(
        int(round((cx - w / 2) * width)), int(round((cy - h / 2) * height)),
        int(round((cx + w / 2) * width)), int(round((cy + h / 2) * height)),
        CLASS_NAMES[int(idx)],
    )


def export(
    train: list[Item], val: list[Item], test: list[Item], out: Path | None = None
) -> Path:
    """Write the YOLO layout and its ``data.yaml``; returns the yaml path.

    Images are copied, not moved, so the dataset as downloaded is untouched.
    The output is derived and gitignored; rerunning overwrites it.
    """
    out = out or DEFAULT_EXPORT
    if out.exists():
        shutil.rmtree(out)
    for name, items in (("train", train), ("val", val), ("test", test)):
        (out / "images" / name).mkdir(parents=True)
        (out / "labels" / name).mkdir(parents=True)
        for item in items:
            width, height = item.image_size()
            shutil.copy2(item.image_path, out / "images" / name / item.image_path.name)
            (out / "labels" / name / f"{item.stem}.txt").write_text(
                "\n".join(to_yolo_line(b, width, height) for b in item.load_boxes()) + "\n"
            )
    yaml = out / "data.yaml"
    yaml.write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    return yaml
