"""HRIPCB, presented to the pipeline the way DeepPCB is.

The second dataset this project's own notes asked for: "a second dataset with
a different prevalence and its own registration problem". HRIPCB is ten
photographed bare boards, one defect-free template each, and 693 images in
which defects were drawn onto that template -- so the template/test pair the
differencing detector needs exists, and the pair is pixel-near-identical
outside the defects (mean |diff| 1.29 on a 3034x1586 frame). The registration
problem is the dataset's own ``rotation/`` set: the same 693 images rotated by an
integer angle in -10..+10 degrees about the centre, with the angle recorded
beside each -- 35 of the 693 recorded as 0, which is a rotated image that
happens not to be turned and is kept as such.

Everything here is a *view*: nothing is written to disk, and nothing about
DeepPCB changes. In particular ``CLASS_NAMES`` is not extended. HRIPCB's sixth
class is ``missing_hole`` where DeepPCB's is ``pin-hole``, and those are
opposite defects (copper where a hole should be, against a hole where copper
should be). A seventh entry in that table would put a class into the standards
retrieval with no document behind it, which `tests/test_standards_retrieval.py`
exists to refuse. So the annotation type is its own, duck-compatible with
``deeppcb.Annotation`` on the four fields and two properties the pipeline
reads, and the re-verifier -- which has never seen a missing hole -- is asked
only the question it can answer: is this a false call or not.

**Scale.** The images are downscaled by ``SCALE`` before anything sees them.
The number is chosen to match *defect size*, not image size: the median defect
long side is 78 px here and 39 px on DeepPCB, and the re-verifier's 64 px patch
was sized against the latter. At 1.0 a typical HRIPCB defect fills the window.

**What this makes comparable, and what it does not.** The detector, the patch
geometry and the checkpoint are unchanged, so an escape rate measured here is
the shipped operating point applied to a population it was not swept on --
which is the question. What is not comparable is prevalence: HRIPCB contains
no false calls at all, so every false call in the queue is one the differencing
stage manufactured, and the prevalence is a property of the detector on these
photographs rather than of the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "HRIPCB" / "PCB_DATASET"

SCALE = 0.5
"""Downscale applied to every image and box. See the module docstring."""

#: HRIPCB's own class vocabulary. Five of the six are DeepPCB's under a
#: different spelling; ``missing_hole`` is not ``pin-hole`` and is not mapped.
CLASS_NAMES: dict[str, str] = {
    "open_circuit": "open",
    "short": "short",
    "mouse_bite": "mousebite",
    "spur": "spur",
    "spurious_copper": "copper",
    "missing_hole": "missing_hole",
}

#: Directory name under ``images/`` and ``Annotations/`` for each class.
FOLDERS = {
    "open_circuit": "Open_circuit",
    "short": "Short",
    "mouse_bite": "Mouse_bite",
    "spur": "Spur",
    "spurious_copper": "Spurious_copper",
    "missing_hole": "Missing_hole",
}

SETS = ("aligned", "rotated")


@dataclass(frozen=True)
class HripcbAnnotation:
    """One ground-truth box, in the (scaled, and for ``rotated`` rotated) frame.

    Duck-compatible with ``deeppcb.Annotation``: the matcher reads ``box`` and
    the escape accounting reads ``class_name`` and the corners.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class HripcbPair:
    """One test image against its board's template."""

    stem: str
    board: str
    template_path: Path
    test_path: Path
    annotation_path: Path
    angle: float = 0.0
    """Degrees the ``rotated`` set turned this image by; 0 for ``aligned``."""

    def load_template(self) -> np.ndarray:
        template = _load_gray(self.template_path)
        if self.angle == 0.0:
            return template
        # The rotated test sits on a canvas rotate_bound() expanded to hold the
        # turned frame, filled with one flat grey. The template goes onto the
        # same canvas, centred, with the same fill -- so the only difference
        # between the two is the rotation the registration stage is being
        # asked about. The fill is read off the rotated image's corner rather
        # than copied from rotate.py's borderValue: derived, never declared.
        rotated_shape, fill = _rotated_canvas(self.test_path)
        return _centre_on_canvas(template, rotated_shape, fill)

    def load_test(self) -> np.ndarray:
        return _load_gray(self.test_path)

    def load_annotations(self) -> list[HripcbAnnotation]:
        root = ET.parse(self.annotation_path).getroot()
        width = int(root.findtext("size/width"))
        height = int(root.findtext("size/height"))
        rows = []
        for obj in root.findall("object"):
            box = obj.find("bndbox")
            corners = [
                float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")
            ]
            if self.angle != 0.0:
                corners = _rotate_box(corners, width, height, self.angle)
            x1, y1, x2, y2 = (int(round(v * SCALE)) for v in corners)
            rows.append(
                HripcbAnnotation(x1, y1, x2, y2, CLASS_NAMES[obj.findtext("name")])
            )
        return rows


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        gray = np.array(im.convert("L"))
    if SCALE == 1.0:
        return gray
    size = (int(round(gray.shape[1] * SCALE)), int(round(gray.shape[0] * SCALE)))
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def _rotated_canvas(rotated_path: Path) -> tuple[tuple[int, int], int]:
    rotated = _load_gray(rotated_path)
    return rotated.shape, int(rotated[0, 0])


def _centre_on_canvas(image: np.ndarray, shape: tuple[int, int], fill: int) -> np.ndarray:
    canvas = np.full(shape, fill, dtype=image.dtype)
    top = (shape[0] - image.shape[0]) // 2
    left = (shape[1] - image.shape[1]) // 2
    canvas[top:top + image.shape[0], left:left + image.shape[1]] = image
    return canvas


def _rotate_box(corners, width: int, height: int, angle: float) -> list[float]:
    """Where a box lands after ``rotate.py``'s ``rotate_bound_white_bg``.

    Reproduces that function's matrix exactly -- rotation about the image
    centre by ``-angle``, then a translation onto the expanded canvas -- and
    takes the axis-aligned bounds of the four turned corners. Full-resolution
    coordinates in, full-resolution out; the caller scales.
    """
    cx, cy = width // 2, height // 2
    matrix = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(height * sin + width * cos)
    new_h = int(height * cos + width * sin)
    matrix[0, 2] += new_w / 2 - cx
    matrix[1, 2] += new_h / 2 - cy

    x1, y1, x2, y2 = corners
    points = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float64)
    turned = points @ matrix.T
    return [turned[:, 0].min(), turned[:, 1].min(), turned[:, 0].max(), turned[:, 1].max()]


def _angles(root: Path) -> dict[str, float]:
    angles: dict[str, float] = {}
    for path in (root / "rotation").glob("*_angles.txt"):
        for line in path.read_text().splitlines():
            if line.strip():
                stem, angle = line.split()
                angles[stem] = float(angle)
    return angles


def load(subset: str = "aligned", root: Path | None = None) -> list[HripcbPair]:
    """Every annotated image, as a template/test pair.

    ``aligned`` is the 693 images as shipped. ``rotated`` is the same 693 from
    the dataset's ``rotation/`` directory, each turned by its recorded angle
    (an integer in -10..+10, sometimes 0),
    against the un-rotated template -- which is the registration problem.
    """
    if subset not in SETS:
        raise ValueError(f"subset must be one of {SETS}, got {subset!r}")
    root = root or DEFAULT_ROOT
    if not (root / "PCB_USED").exists():
        raise FileNotFoundError(
            f"{root / 'PCB_USED'} not found. Download the dataset first:\n"
            "  kaggle datasets download -d akhatova/pcb-defects -p data/HRIPCB --unzip"
        )
    angles = _angles(root) if subset == "rotated" else {}

    pairs = []
    for name, folder in FOLDERS.items():
        for annotation_path in sorted((root / "Annotations" / folder).glob("*.xml")):
            stem = annotation_path.stem
            board = stem.split("_")[0]
            if subset == "aligned":
                test_path = root / "images" / folder / f"{stem}.jpg"
                angle = 0.0
            else:
                test_path = root / "rotation" / f"{folder}_rotation" / f"{stem}.jpg"
                angle = angles[stem]
            pairs.append(
                HripcbPair(
                    stem=stem,
                    board=board,
                    template_path=root / "PCB_USED" / f"{board}.JPG",
                    test_path=test_path,
                    annotation_path=annotation_path,
                    angle=angle,
                )
            )
    return pairs
