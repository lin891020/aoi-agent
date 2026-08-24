"""The registration stage: what it recovers, and the three things it refuses.

Built after `scripts/registration_report.py` measured what misalignment costs.
What these tests hold is not "registration works" -- that is a curve, and it is
in the benchmarks -- but the four decisions the stage makes, each of which was
wrong in the first draft and corrected by measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from aoi_agent.aoi.registration import (
    MAX_SHIFT_FRACTION,
    MIN_SHIFT_PX,
    Alignment,
    align,
    estimate_shift,
)


def board(seed: int = 0, size: int = 256) -> np.ndarray:
    """A binarised board: traces on a dark ground, no texture to key on."""
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size), np.uint8)
    for _ in range(14):
        y = int(rng.integers(10, size - 10))
        image[y:y + 3, 8:size - 8] = 255
    for _ in range(14):
        x = int(rng.integers(10, size - 10))
        image[8:size - 8, x:x + 3] = 255
    return image


def shifted(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return np.roll(np.roll(image, dy, axis=0), dx, axis=1)


def test_it_finds_a_shift_it_was_not_told_about():
    template = board()
    found = estimate_shift(template, shifted(template, 6, -4))

    assert found.dx == pytest.approx(6, abs=0.5)
    assert found.dy == pytest.approx(-4, abs=0.5)


def test_the_test_image_moves_and_the_template_never_does():
    """The template is what the defect boxes are expressed against. Warping it
    would move the ground truth along with it, and every measurement here would
    look better than it is by construction."""
    template = board()
    test = shifted(template, 5, 0)
    # A mark that exists only on the tested board -- a defect, in miniature.
    # If `align` warps the template instead, the mark is not in what comes
    # back, and `array_equal(template, board())` would not have noticed:
    # `warpAffine` returns a new array and mutates nothing, so checking the
    # caller's template is unchanged tests nothing at all. Found by mutation.
    # Somewhere the board is provably empty, so the mark is distinguishable
    # from a trace. Asserting `max() == 255` over a patch that already held one
    # passed whichever image was warped -- the first version of this test did
    # exactly that.
    # Blank in *both*: the two images are 5 px apart, so a window empty in one
    # can hold a trace in the other -- which is what the first attempt at this
    # tripped over.
    def blank(image):
        return (
            np.lib.stride_tricks.sliding_window_view(image, (12, 12))
            .max(axis=(2, 3)) == 0
        )

    empty = np.argwhere(blank(test) & blank(template))
    assert len(empty), "the fixture has no region blank in both images"
    y, x = empty[len(empty) // 2]
    test = test.copy()
    test[y + 2:y + 10, x + 2:x + 10] = 255

    corrected, found = align(template, test)

    assert found.refused is None
    assert corrected.shape == template.shape
    assert template[y:y + 12, x:x + 12].max() == 0, "the region really is blank"
    assert corrected[y:y + 12, x:x + 12].max() == 255, (
        "the defect only the tested board carries has to survive -- if it does "
        "not, the template was warped and the test image was discarded"
    )


# ---------------------------------------------------------------------------
# The three refusals
# ---------------------------------------------------------------------------

def test_a_sub_pixel_correction_is_declined():
    """These images are binarised, so warping by half a pixel writes grey along
    every edge and the detector reads it as difference. Measured: registering
    already-aligned pairs on their 0.48 px median estimate made 17 of 60
    worse; declining takes that to 3."""
    template = board()
    corrected, found = align(template, template.copy())

    assert found.refused == "already_aligned"
    assert np.array_equal(corrected, template), "nothing was warped"


def test_an_implausible_shift_is_declined():
    """A board is misplaced by millimetres, not by a third of the frame. Four
    of 60 aligned DeepPCB pairs produced estimates of 240-355 px on a 640 px
    frame -- correlation failures, not boards."""
    # An absolute figure, not one derived from the constant under test: the
    # first version computed its input from `MAX_SHIFT_FRACTION`, so raising
    # the constant raised the test's own expectation with it and the assertion
    # could not fail. Found by mutation. 200 px on a 256 px board is a third of
    # the frame either way.
    template = board(size=256)
    _, found = align(
        template, template.copy(), Alignment(dx=200.0, dy=0.0, confidence=0.9)
    )

    assert found.refused == "implausible"
    assert MAX_SHIFT_FRACTION * 256 < 200, "and the guard has to be below it"


def test_confidence_alone_would_not_have_caught_it():
    """The correction that produced `MAX_SHIFT_FRACTION`, kept as a test.

    The first draft refused on confidence only, and its docstring said that was
    sufficient. Two of those four failures came back at 0.134 and 0.076 --
    above any floor low enough to admit the real cases. A test that only ever
    fed low-confidence failures would have agreed with the docstring.
    """
    template = board(size=256)
    confident_but_wrong = Alignment(dx=200.0, dy=0.0, confidence=0.134)

    _, found = align(template, template.copy(), confident_but_wrong)

    assert found.refused == "implausible", (
        "a confident, implausible estimate has to be refused on magnitude"
    )


def test_a_real_correction_is_not_declined():
    """The guards must not swallow the case the stage exists for."""
    template = board()
    _, found = align(template, shifted(template, 5, 3))

    assert found.refused is None
    assert found.magnitude > MIN_SHIFT_PX
