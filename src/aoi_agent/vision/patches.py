"""Turn AOI candidates into training patches.

Each candidate the AOI simulator flags becomes one fixed-size patch with three
channels:

    0. the golden template at that location
    1. the board under test at that location
    2. the absolute difference between them

Giving the model all three rather than just the difference matters: the same
difference blob means different things depending on whether copper was added
or removed, and only the template/test pair carries that.

Patches are a fixed window centred on the candidate rather than a resized crop
of it. Defect classes have characteristic sizes -- a pin-hole is small and
round, a short is long and thin -- and resizing every candidate to the same
scale throws that signal away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aoi_agent.aoi.matching import LabelledCandidate
from aoi_agent.aoi.simulator import Candidate

PATCH_SIZE = 64
"""Window side in pixels. Ground-truth defects run roughly 30-45 px, so this
keeps the whole defect plus a margin of surrounding circuitry for context."""


def crop(image: np.ndarray, candidate: Candidate, size: int = PATCH_SIZE) -> np.ndarray:
    """Cut a ``size``x``size`` window centred on the candidate.

    Windows that run past the image edge are padded with the image's own
    border value so the model never sees an artificial black frame.
    """
    cx = (candidate.x1 + candidate.x2) // 2
    cy = (candidate.y1 + candidate.y2) // 2
    half = size // 2

    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + size, y1 + size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - image.shape[1])
    pad_bottom = max(0, y2 - image.shape[0])

    window = image[
        max(0, y1) : min(image.shape[0], y2),
        max(0, x1) : min(image.shape[1], x2),
    ]

    if pad_left or pad_top or pad_right or pad_bottom:
        window = np.pad(
            window,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="edge",
        )
    return window


def build_patch(
    template: np.ndarray,
    test: np.ndarray,
    candidate: Candidate,
    size: int = PATCH_SIZE,
) -> np.ndarray:
    """Return a ``(3, size, size)`` uint8 patch for one candidate."""
    template_patch = crop(template, candidate, size)
    test_patch = crop(test, candidate, size)
    diff_patch = np.abs(
        test_patch.astype(np.int16) - template_patch.astype(np.int16)
    ).astype(np.uint8)
    return np.stack([template_patch, test_patch, diff_patch], axis=0)


@dataclass
class PatchSet:
    """Patches plus their labels and enough provenance to trace one back."""

    patches: np.ndarray       # (n, 3, size, size) uint8
    labels: np.ndarray        # (n,) int64, index into label_names
    label_names: list[str]
    image_index: np.ndarray   # (n,) int64, which source image
    boxes: np.ndarray         # (n, 4) int64, candidate box in image coordinates

    def __len__(self) -> int:
        return len(self.labels)

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            patches=self.patches,
            labels=self.labels,
            label_names=np.array(self.label_names),
            image_index=self.image_index,
            boxes=self.boxes,
        )

    @classmethod
    def load(cls, path) -> "PatchSet":
        data = np.load(path, allow_pickle=False)
        return cls(
            patches=data["patches"],
            labels=data["labels"],
            label_names=[str(n) for n in data["label_names"]],
            image_index=data["image_index"],
            boxes=data["boxes"],
        )


def patches_for_image(
    template: np.ndarray,
    test: np.ndarray,
    labelled: list[LabelledCandidate],
    size: int = PATCH_SIZE,
) -> tuple[list[np.ndarray], list[str], list[tuple[int, int, int, int]]]:
    """Build patches for every trainable candidate in one image."""
    patches, labels, boxes = [], [], []
    for item in labelled:
        if not item.trainable:
            continue
        patches.append(build_patch(template, test, item.candidate, size))
        labels.append(item.label)
        boxes.append(item.candidate.box)
    return patches, labels, boxes
