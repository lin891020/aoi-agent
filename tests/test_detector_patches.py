"""The crop re-verifier's training set, at the two places it could quietly lie.

A patch here is an RGB window with no template channel, and the split that
`scripts/train.py` makes is by `image_index`. Both are one line each, and both
would fail silently: a crop that mixed channel order, or an index that named
the file rather than the board, trains a model that scores well on a
validation set it has already seen.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from aoi_agent.aoi.simulator import Candidate
from aoi_agent.data import pcbaoi

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "build_detector_patches", ROOT / "scripts" / "build_detector_patches.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_an_rgb_patch_keeps_the_image_channels_in_order():
    """Channel 0 is red, not a template. A model trained on a stack whose
    planes were shuffled would still converge; it would just be learning the
    wrong colour."""
    script = _load_script()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30
    candidate = Candidate(x1=40, y1=40, x2=60, y2=60, area=400)

    patch = script.rgb_patch(image, candidate)

    assert patch.shape == (3, 64, 64) and patch.dtype == np.uint8
    assert (patch[0] == 10).all() and (patch[1] == 20).all() and (patch[2] == 30).all()


def test_a_patch_at_the_border_is_padded_with_the_image_edge():
    script = _load_script()
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    candidate = Candidate(x1=0, y1=0, x2=10, y2=10, area=100)

    patch = script.rgb_patch(image, candidate)

    assert patch.shape == (3, 64, 64)
    assert (patch == 200).all(), "edge padding must not introduce a black frame"


def test_the_image_index_groups_a_board_with_its_augmentations():
    """`split_by_image` holds out whole `image_index` values. If a rotation of
    a training board carried a different index from its original, one could
    train while the other validated -- the leak the by-image split exists to
    prevent, reintroduced by the dataset's own augmentation set."""
    stems = ["20220101-SPI-AOI-1", "20220101-SPI-AOI-1_90", "20220101-SPI-AOI-1_shuiping",
             "20220102-SPI-AOI-2"]
    bases = [pcbaoi.base_stem(s) for s in stems]

    assert bases[0] == bases[1] == bases[2] != bases[3]
