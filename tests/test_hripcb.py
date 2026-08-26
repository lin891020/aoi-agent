"""HRIPCB presented as template/test pairs, and the geometry that makes the
rotated subset an honest registration test.

Two things can go quietly wrong in a dataset adapter, and both look like a
recall number rather than like a bug. The first is a box that lands in the
wrong frame: the rotated images sit on an expanded canvas, and a box rotated
about the wrong centre or without the canvas offset is a box drawn on the
board next door. The second is a class table that leaks: HRIPCB's
``missing_hole`` is not DeepPCB's ``pin-hole``, and the two must never be
folded together, because the escape accounting is per class and the standards
retrieval refuses a class with no document.

The geometry tests need no dataset. The rest are behind ``-m dataset`` and
skip when `data/HRIPCB` is absent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aoi_agent.data import deeppcb, hripcb
from aoi_agent.data.hripcb import _centre_on_canvas, _rotate_box

# ---- pure geometry ---------------------------------------------------------


def test_a_zero_rotation_leaves_a_box_where_it_was():
    assert _rotate_box([100, 200, 140, 260], 1000, 800, 0.0) == pytest.approx(
        [100, 200, 140, 260], abs=1e-6
    )


def test_a_box_at_the_centre_stays_at_the_centre_of_the_expanded_canvas():
    """rotate_bound() expands the canvas and moves the centre with it. A box on
    the original centre must land on the new centre, whatever the angle."""
    width, height = 1000, 800
    box = [width / 2 - 10, height / 2 - 10, width / 2 + 10, height / 2 + 10]
    for angle in (-9.0, -1.0, 4.0, 9.0):
        turned = _rotate_box(box, width, height, angle)
        cos, sin = abs(math.cos(math.radians(angle))), abs(math.sin(math.radians(angle)))
        new_w, new_h = int(height * sin + width * cos), int(height * cos + width * sin)
        cx, cy = (turned[0] + turned[2]) / 2, (turned[1] + turned[3]) / 2
        assert cx == pytest.approx(new_w / 2, abs=1.0), angle
        assert cy == pytest.approx(new_h / 2, abs=1.0), angle


def test_a_rotated_box_grows_and_never_shrinks():
    """The axis-aligned bounds of a turned rectangle contain it, so every side
    is at least as long as before. A box that got smaller was rotated about
    the wrong point or with the wrong sign somewhere."""
    box = [100, 200, 160, 230]
    for angle in (-9.0, 3.0, 9.0):
        x1, y1, x2, y2 = _rotate_box(box, 1000, 800, angle)
        assert x2 - x1 >= 60 - 1e-6 and y2 - y1 >= 30 - 1e-6


def test_the_template_is_centred_on_the_rotated_canvas_with_the_same_fill():
    image = np.full((80, 100), 200, dtype=np.uint8)
    canvas = _centre_on_canvas(image, (100, 120), fill=17)
    assert canvas.shape == (100, 120)
    assert canvas[10:90, 10:110].min() == 200
    assert canvas[0, 0] == 17 and canvas[-1, -1] == 17
    assert (canvas == 200).sum() == image.size


# ---- the class boundary ----------------------------------------------------


def test_missing_hole_is_not_pin_hole_and_deeppcbs_table_is_untouched():
    """The seventh class stays out of ``deeppcb.CLASS_NAMES``. A row there
    would put a class into the standards retrieval with no document behind it,
    and it would fold two opposite defects into one escape figure."""
    assert hripcb.CLASS_NAMES["missing_hole"] == "missing_hole"
    assert "missing_hole" not in deeppcb.CLASS_NAMES.values()
    assert len(deeppcb.CLASS_NAMES) == 6
    shared = set(hripcb.CLASS_NAMES.values()) & set(deeppcb.CLASS_NAMES.values())
    assert shared == {"open", "short", "mousebite", "spur", "copper"}


def test_an_unknown_subset_is_refused():
    with pytest.raises(ValueError, match="subset"):
        hripcb.load("test")


# ---- against the files -----------------------------------------------------


@pytest.fixture(scope="module")
def pairs():
    if not (hripcb.DEFAULT_ROOT / "PCB_USED").exists():
        pytest.skip("HRIPCB not downloaded; see data/hripcb.py")
    return {subset: hripcb.load(subset) for subset in hripcb.SETS}


@pytest.mark.dataset
def test_every_annotated_image_is_a_pair_in_both_subsets(pairs):
    assert len(pairs["aligned"]) == len(pairs["rotated"]) == 693
    assert {p.stem for p in pairs["aligned"]} == {p.stem for p in pairs["rotated"]}
    assert len({p.board for p in pairs["aligned"]}) == 10


@pytest.mark.dataset
def test_template_and_test_share_a_frame_in_both_subsets(pairs):
    """detect() and align() need the two images the same shape. On ``rotated``
    that only holds because the template was padded onto the turned canvas."""
    for subset in hripcb.SETS:
        for pair in pairs[subset][::97]:
            assert pair.load_template().shape == pair.load_test().shape, (subset, pair.stem)


@pytest.mark.dataset
def test_the_aligned_pairs_really_are_aligned(pairs):
    """The premise of using the differencing detector at all: outside the
    defects, template and test are the same photograph."""
    pair = pairs["aligned"][0]
    template = pair.load_template().astype(np.int16)
    test = pair.load_test().astype(np.int16)
    outside = np.abs(template - test)
    for a in pair.load_annotations():
        outside[a.y1:a.y2, a.x1:a.x2] = 0
    assert np.percentile(outside, 99.9) <= 4


@pytest.mark.dataset
def test_every_rotated_pair_carries_its_recorded_angle(pairs):
    assert all(p.angle == 0.0 for p in pairs["aligned"])
    angles = [p.angle for p in pairs["rotated"]]
    # The file records integers in -10..+10 and 35 of them are 0: an image
    # from the rotated set that was not turned. It stays in the subset as it
    # is; dropping it would make the subset look harder than it was shipped.
    assert all(a == int(a) for a in angles)
    assert -10.0 <= min(angles) and max(angles) <= 10.0
    assert sum(1 for a in angles if a == 0.0) < len(angles) / 10


@pytest.mark.dataset
def test_a_rotated_box_still_sits_on_a_difference(pairs):
    """The end-to-end check on the geometry: after rotation, padding and
    scaling, the box must still contain the defect -- which shows up as the
    template/test difference being larger inside the box than outside it."""
    pair = [p for p in pairs["rotated"] if abs(p.angle) >= 6][0]
    template = pair.load_template().astype(np.int16)
    test = pair.load_test().astype(np.int16)
    diff = np.abs(template - test)
    # Rotation puts difference along every edge, so compare against the
    # frame's own background level rather than against zero.
    background = float(np.median(diff))
    inside = [float(diff[a.y1:a.y2, a.x1:a.x2].mean()) for a in pair.load_annotations()]
    assert all(v > background for v in inside), (pair.stem, pair.angle, inside, background)


@pytest.mark.dataset
def test_scaling_matches_deeppcbs_defect_size_not_its_image_size():
    """The 0.5 is a defect-size decision and this pins the reason: at that
    scale HRIPCB's median defect long side is within a few pixels of
    DeepPCB's, which is what the 64 px patch was sized against."""
    if not (hripcb.DEFAULT_ROOT / "PCB_USED").exists():
        pytest.skip("HRIPCB not downloaded")
    sides = sorted(
        max(a.x2 - a.x1, a.y2 - a.y1)
        for pair in hripcb.load("aligned")[::7]
        for a in pair.load_annotations()
    )
    median = sides[len(sides) // 2]
    assert 34 <= median <= 44, median
