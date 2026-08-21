"""Torch dataset over AOI candidate patches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from aoi_agent.vision.patches import PatchSet


class CandidateDataset(Dataset):
    """Patches produced by the AOI simulator, labelled against ground truth.

    Augmentation is limited to flips and 90-degree rotations. A PCB has no
    canonical orientation under a line-scan camera, so those are label
    preserving. Anything that changes intensity is not: the template and test
    channels only mean something relative to each other, and shifting one would
    manufacture a difference that never existed.
    """

    def __init__(self, patch_set: PatchSet, augment: bool = False):
        self.patch_set = patch_set
        self.augment = augment

    @classmethod
    def from_file(cls, path: Path | str, augment: bool = False) -> "CandidateDataset":
        return cls(PatchSet.load(path), augment=augment)

    @property
    def label_names(self) -> list[str]:
        return self.patch_set.label_names

    def __len__(self) -> int:
        return len(self.patch_set)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        patch = self.patch_set.patches[index]
        label = int(self.patch_set.labels[index])

        if self.augment:
            patch = _augment(patch)

        tensor = torch.from_numpy(np.ascontiguousarray(patch)).float().div_(255.0)
        return tensor, label


def _augment(patch: np.ndarray) -> np.ndarray:
    """Random flips and quarter turns, applied to all three channels alike."""
    if np.random.rand() < 0.5:
        patch = patch[:, :, ::-1]
    if np.random.rand() < 0.5:
        patch = patch[:, ::-1, :]
    turns = np.random.randint(4)
    if turns:
        patch = np.rot90(patch, turns, axes=(1, 2))
    return patch


def class_weights(patch_set: PatchSet) -> torch.Tensor:
    """Inverse-frequency weights.

    Roughly half of every batch is ``false_call``. Left unweighted the model
    can score well by dismissing everything, which is precisely the failure
    mode that lets defects escape.
    """
    counts = np.bincount(patch_set.labels, minlength=len(patch_set.label_names))
    weights = counts.sum() / np.maximum(counts, 1)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)
