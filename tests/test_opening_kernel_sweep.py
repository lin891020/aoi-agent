"""The opening-kernel trade: the arithmetic, and the sentence it licenses.

No board is loaded. What can go wrong quietly here is not the morphology --
that is OpenCV's -- it is the price. Dividing added false calls by candidates
recovered instead of by defects the model then keeps makes every smaller kernel
look cheaper than it is, and the number that comes out still prints.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from aoi_agent.aoi.simulator import DetectorConfig, detect, opening_element

from opening_kernel_sweep import SHIPPED, Row, cost_per_defect, is_noop, render


# --------------------------------------------------------------------------
# the structuring element the sweep needed
# --------------------------------------------------------------------------


def test_the_default_element_is_the_square_that_always_shipped():
    element = opening_element(DetectorConfig())
    assert np.array_equal(element, np.ones((3, 3), np.uint8))


def test_an_unknown_shape_is_refused_rather_than_silently_squared():
    with pytest.raises(ValueError, match="open_shape"):
        opening_element(DetectorConfig(open_shape="diamond"))


def test_a_cross_keeps_a_plus_the_square_erases(blank_board):
    """The whole reason the cross is in the sweep. A 3x3 square opening survives
    only where a full 3x3 block of difference sits; a cross survives on the five
    cells of a plus. Anything shaped between the two lives or dies on that
    choice, and cannot be reached by changing the kernel *size*."""
    board = blank_board
    marked = board.copy()
    marked[49:52, 50] = 255           # a plus of difference: three tall,
    marked[50, 49:52] = 255           # three wide, corners empty

    square = detect(board, marked, DetectorConfig(open_kernel=3))
    cross = detect(board, marked, DetectorConfig(open_kernel=3, open_shape="cross"))

    assert square == []
    assert cross


def test_the_shape_is_only_a_shape_and_the_size_still_applies():
    element = opening_element(DetectorConfig(open_kernel=5, open_shape="cross"))
    assert element.shape == (5, 5)
    assert np.array_equal(
        element, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    )


# --------------------------------------------------------------------------
# is_noop
# --------------------------------------------------------------------------


def test_a_1px_element_is_reported_as_a_no_op_not_as_a_data_point():
    """OpenCV's 1x1 opening returns the mask unchanged, so a row for it would
    duplicate the no-opening row and read as evidence."""
    assert is_noop(DetectorConfig(open_kernel=0))
    assert is_noop(DetectorConfig(open_kernel=1))
    assert not is_noop(DetectorConfig(open_kernel=2))


# --------------------------------------------------------------------------
# the price
# --------------------------------------------------------------------------


def _row(name, false_calls, candidates=None, recovered=0, kept=0, unflagged=0,
         config=None) -> Row:
    boards = 500
    return Row(
        name=name,
        config=config or DetectorConfig(),
        defects=3140,
        unflagged=unflagged,
        missed_at_cut=157,
        false_calls=[false_calls] * boards,
        candidates=[candidates or false_calls] * boards,
        perturbed_false_calls=[false_calls * 3] * boards,
        perturbed_candidates=[false_calls * 3] * boards,
        recovered=recovered,
        recovered_kept=kept,
    )


def test_the_price_is_per_defect_kept_not_per_candidate_recovered():
    """A setting that recovers seven and has five survive the model costs the
    same total and buys five, not seven."""
    shipped = _row(SHIPPED, 10.0, unflagged=7)
    smaller = _row("2x2 square", 39.0, recovered=7, kept=5)

    assert cost_per_defect(smaller, shipped, smaller.recovered_kept) == pytest.approx(
        (39.0 - 10.0) * 500 / 5
    )
    # Priced against candidates recovered it would look 29% cheaper.
    assert cost_per_defect(smaller, shipped, smaller.recovered) < cost_per_defect(
        smaller, shipped, smaller.recovered_kept
    )


def test_a_setting_the_model_dismisses_everything_from_has_no_price_at_all():
    """Not zero, and not infinity dressed as a number -- it bought nothing."""
    shipped = _row(SHIPPED, 10.0, unflagged=7)
    useless = _row("2x2 square", 39.0, recovered=7, kept=0)
    assert cost_per_defect(useless, shipped, useless.recovered_kept) is None


def test_the_row_aggregates_are_means_over_boards_not_totals():
    row = _row("3x3 square", 10.0, unflagged=7)
    assert row.boards == 500
    assert row.mean_false_calls == pytest.approx(10.0)
    assert row.unflagged_rate == pytest.approx(7 / 3140)


# --------------------------------------------------------------------------
# the prose
# --------------------------------------------------------------------------


def _rows() -> list[Row]:
    return [
        _row("none", 123.24, candidates=153.38, recovered=7, kept=4, unflagged=7,
             config=DetectorConfig(open_kernel=0)),
        _row("2x2 square", 39.17, candidates=51.50, recovered=7, kept=5, unflagged=0),
        _row("3x3 cross", 19.47, candidates=28.83, recovered=5, kept=4, unflagged=2),
        _row(SHIPPED, 10.29, candidates=18.14, unflagged=7),
        _row("4x4 square", 3.52, candidates=9.79, unflagged=255),
        _row("5x5 square", 0.74, candidates=5.66, unflagged=799),
    ]


def _forensics() -> list[dict]:
    return [
        {"board": f"board{i}", "class": "open", "box": "27x30",
         "diff_pixels": 24 + i, "after_opening": 0, "half_width": 1.37}
        for i in range(7)
    ]


def test_the_section_states_the_decision_in_its_heading():
    text = "\n".join(render(_rows(), _forensics(), "test", 0.915))
    assert "Decision: `open_kernel` stays at 3" in text


def test_the_section_prices_the_trade_rather_than_asserting_it():
    text = "\n".join(render(_rows(), _forensics(), "test", 0.915))
    assert "per defect the model then keeps" in text
    assert "misregistration" in text


def test_the_section_warns_the_iou_column_is_not_a_recall_improvement():
    text = "\n".join(render(_rows(), _forensics(), "test", 0.915))
    assert "means almost" in text
    assert "reaching an operator either way" in text


def test_the_section_carries_the_out_of_distribution_caveat():
    """The 'model keeps' column decides the question and is the weakest number
    in the run. It must not travel without its caveat."""
    text = "\n".join(render(_rows(), _forensics(), "test", 0.915))
    assert "out of its distribution" in text
    assert "trained on `open_kernel=3` patches" in text


def test_the_section_names_what_would_reopen_the_decision():
    text = "\n".join(render(_rows(), _forensics(), "test", 0.915))
    assert "Revisit this if" in text
