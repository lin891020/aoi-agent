"""The escape accounting, and the labelling behaviour that made it necessary.

Nothing here loads a checkpoint or a board. The mistake being corrected was
never in the arithmetic -- ``system_escape_rate`` computed exactly what it was
asked for -- it was in which population got fed to it, and that is decided by
two pure functions and one labelling rule. Those are what is pinned.
"""

from __future__ import annotations

import numpy as np
import pytest

from aoi_agent.aoi.matching import FRAGMENT, match
from aoi_agent.vision.operating_point import system_escape_rate
from conftest import make_annotation, make_candidate

from escape_accounting import (
    DISMISSED,
    ESCAPES,
    REVIEWED_MATCHED,
    REVIEWED_SUB_CUT,
    UNFLAGGED,
    Accounting,
    covering_candidates,
    defect_outcome,
    render,
)
from collections import Counter


# --------------------------------------------------------------------------
# the labelling rule that produced the wrong number
# --------------------------------------------------------------------------


def test_a_loose_candidate_on_a_defect_is_a_fragment_and_the_defect_reads_missed():
    """Both halves of the accounting error, in one assertion each.

    A candidate that lands squarely on a real defect but draws a tighter box
    than the annotator did clears neither the IoU cut nor the coverage rule --
    on the test split a matched candidate is a median 0.51x the ground truth's
    area, so this is the ordinary case rather than a contrived one. The
    candidate is labelled `fragment`, which holds it out of the patch set and
    therefore out of the denominator of every operating point. The annotation
    lands in `missed_annotations`, which is where the 5.0% "AOI escape rate"
    came from.

    So the region is removed from the only measurement that could show the model
    handling it, and charged in full to the stage that did in fact flag it.
    """
    defect = make_annotation(50, 50, 90, 90)        # 40x40, as an annotator drew it
    tight = make_candidate(60, 60, 80, 72)          # on it, 0.15 of its area

    result = match([tight], [defect])

    assert result.labelled[0].label == FRAGMENT
    assert not result.labelled[0].trainable        # out of the model's score
    assert result.missed_annotations == [defect]   # in the line's escape count
    assert result.labelled[0].best_iou < 0.33


def test_the_same_candidate_matches_once_the_cut_is_loosened():
    """Nothing about the detector changed -- only the number it is judged by."""
    defect = make_annotation(50, 50, 90, 90)
    tight = make_candidate(60, 60, 80, 72)

    assert match([tight], [defect]).missed_annotations
    assert not match([tight], [defect], iou_threshold=0.10).missed_annotations


# --------------------------------------------------------------------------
# covering_candidates -- the loosest possible "did anything get flagged"
# --------------------------------------------------------------------------


def test_covering_candidates_finds_a_box_that_overlaps_by_a_sliver():
    defect = make_annotation(50, 50, 60, 60)
    grazing = make_candidate(59, 59, 90, 90)
    assert covering_candidates(defect.box, [grazing]) == [0]


def test_covering_candidates_finds_a_candidate_swallowed_by_the_defect():
    """IoU can be tiny in both directions. A dot inside a loose ground-truth box
    still means the detector saw something there."""
    defect = make_annotation(0, 0, 100, 100)
    dot = make_candidate(48, 48, 52, 52)
    assert covering_candidates(defect.box, [dot]) == [0]


def test_covering_candidates_is_empty_when_nothing_touches():
    defect = make_annotation(50, 50, 60, 60)
    elsewhere = make_candidate(200, 200, 220, 220)
    assert covering_candidates(defect.box, [elsewhere]) == []


def test_covering_candidates_rejects_a_box_that_only_abuts():
    """Touching edges share no pixel, and an operator would see nothing."""
    defect = make_annotation(50, 50, 60, 60)
    abutting = make_candidate(60, 50, 70, 60)
    assert covering_candidates(defect.box, [abutting]) == []


# --------------------------------------------------------------------------
# defect_outcome -- what actually became of one defect
# --------------------------------------------------------------------------


def test_a_defect_with_no_candidate_is_unflagged():
    defect = make_annotation(50, 50, 60, 60)
    assert defect_outcome(defect, [], np.array([], dtype=bool), False) == UNFLAGGED


def test_a_defect_whose_only_candidate_is_dismissed_escapes():
    defect = make_annotation(50, 50, 60, 60)
    candidate = make_candidate(48, 48, 62, 62)
    kept = np.array([False])
    assert defect_outcome(defect, [candidate], kept, True) == DISMISSED


def test_one_kept_candidate_is_enough_to_put_the_region_on_a_screen():
    """A defect escapes only when *every* candidate covering it is dismissed."""
    defect = make_annotation(50, 50, 60, 60)
    candidates = [make_candidate(48, 48, 62, 62), make_candidate(52, 52, 58, 58)]
    assert defect_outcome(defect, candidates, np.array([False, True]), True) == (
        REVIEWED_MATCHED
    )


def test_a_kept_sub_cut_candidate_is_reviewed_not_escaped():
    """The 4.65% the old number charged to the line. Kept, and below the cut."""
    defect = make_annotation(50, 50, 90, 90)
    tight = make_candidate(60, 60, 80, 72)
    assert defect_outcome(defect, [tight], np.array([True]), False) == REVIEWED_SUB_CUT


def test_only_the_two_escape_outcomes_count_as_escapes():
    assert set(ESCAPES) == {UNFLAGGED, DISMISSED}
    assert REVIEWED_SUB_CUT not in ESCAPES
    assert REVIEWED_MATCHED not in ESCAPES


# --------------------------------------------------------------------------
# the composition
# --------------------------------------------------------------------------


def _accounting(matched=2975, sub_cut=146, dismissed=12, unflagged=7) -> Accounting:
    outcomes = Counter({
        REVIEWED_MATCHED: matched,
        REVIEWED_SUB_CUT: sub_cut,
        DISMISSED: dismissed,
        UNFLAGGED: unflagged,
    })
    total = matched + sub_cut + dismissed + unflagged
    return Accounting(
        defects=total,
        outcomes=outcomes,
        escapes_by_class={UNFLAGGED: Counter({"open": unflagged}),
                          DISMISSED: Counter({"open": dismissed})},
        defects_by_class=Counter({"open": total}),
        miss_by_cut={0.50: 1489, 0.40: 487, 0.33: 157, 0.30: 98,
                     0.25: 58, 0.20: 38, 0.10: 17},
        unflagged=[(f"board{i}", "open") for i in range(unflagged)],
        best_iou_of_sub_cut=[0.29] * sub_cut,
        area_ratio_of_matched=[0.51] * matched,
    )


def test_the_escape_count_is_the_unflagged_plus_the_dismissed():
    a = _accounting()
    assert a.escapes == 19
    assert a.whole_line_rate == pytest.approx(19 / 3140)
    assert a.unflagged_rate == pytest.approx(7 / 3140)


def test_the_reverifier_rate_is_over_what_reached_it_not_over_everything():
    """Dividing by every defect would credit the model for regions it never saw."""
    a = _accounting()
    assert a.flagged == 3133
    assert a.reverifier_rate == pytest.approx(12 / 3133)


def test_the_compounded_rate_agrees_with_the_direct_count():
    """The formula was never wrong; it was fed the wrong stage rate."""
    a = _accounting()
    compounded = system_escape_rate(a.reverifier_rate, a.unflagged_rate)
    assert compounded == pytest.approx(a.whole_line_rate)


def test_feeding_the_iou_miss_rate_reproduces_the_number_that_was_published():
    """The old 5.4%, regenerated, so the size of the error is pinned rather than
    remembered. 5.0% was the IoU-0.33 miss rate and 0.47% the candidate-level
    escape rate at the +/-0.5% budget."""
    assert system_escape_rate(0.0047, 0.050) == pytest.approx(0.0545, abs=5e-4)


def test_a_line_with_a_perfect_model_still_escapes_what_was_never_flagged():
    a = _accounting(dismissed=0)
    assert a.reverifier_rate == 0.0
    assert a.escapes == 7
    assert system_escape_rate(0.0, a.unflagged_rate) == pytest.approx(a.unflagged_rate)


# --------------------------------------------------------------------------
# the prose
# --------------------------------------------------------------------------


def test_the_section_says_what_the_number_was_and_what_it_is():
    text = "\n".join(render(_accounting(), "test", 0.915, 500))
    assert "5.4%" in text
    assert "0.61%" in text
    assert "0.22%" in text


def test_the_section_refuses_to_call_the_sub_cut_defects_escapes():
    text = "\n".join(render(_accounting(), "test", 0.915, 500))
    assert "146" in text
    assert "operator's screen" in text


def test_the_section_reports_two_numbers_and_labels_which_one_moves():
    text = "\n".join(render(_accounting(), "test", 0.915, 500))
    assert "unrecoverable" in text
    assert "Two numbers, not one" in text
