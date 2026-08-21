"""The re-verification model.

ResNet-18 off the shelf, not a bespoke architecture. The interesting work in
this project is the operating point, not the backbone, and starting from a
known quantity makes it obvious how much of the result comes from the pipeline
rather than from architecture tuning.

Input is a 3x64x64 stack of (template, test, difference), so the standard
3-channel stem is used unchanged.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet-18 with the classifier resized to ``num_classes``."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def select_device(prefer: str | None = None) -> torch.device:
    """Pick the best available device, preferring Apple's GPU."""
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
