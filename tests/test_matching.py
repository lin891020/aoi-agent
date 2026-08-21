import pytest

from aoi_agent.aoi.matching import FRAGMENT, box_gap, coverage, iou, match
from aoi_agent.data.deeppcb import FALSE_CALL

from conftest import make_annotation, make_candidate


def test_iou_of_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_iou_is_symmetric():
    a, b = (0, 0, 10, 10), (5, 5, 15, 15)
    assert iou(a, b) == pytest.approx(iou(b, a))


def test_coverage_rewards_containing_a_small_target():
    """A loose candidate around a tiny defect scores badly on IoU but covers it."""
    candidate, target = (0, 0, 100, 100), (40, 40, 50, 50)
    assert iou(candidate, target) < 0.05
    assert coverage(candidate, target) == pytest.approx(1.0)


def test_box_gap_is_zero_for_overlapping_boxes():
    assert box_gap((0, 0, 10, 10), (5, 5, 15, 15)) == 0.0


def test_box_gap_measures_separation():
    assert box_gap((0, 0, 10, 10), (13, 0, 20, 10)) == pytest.approx(3.0)


def test_candidate_on_a_defect_takes_its_class():
    result = match(
        [make_candidate(10, 10, 30, 30)],
        [make_annotation(12, 12, 28, 28, class_id=2)],
    )
    assert result.labelled[0].label == "short"
    assert result.detected_annotations
    assert not result.missed_annotations


def test_candidate_far_from_everything_is_a_false_call():
    result = match(
        [make_candidate(200, 200, 220, 220)],
        [make_annotation(10, 10, 30, 30)],
    )
    assert result.labelled[0].label == FALSE_CALL
    assert len(result.missed_annotations) == 1


def test_candidate_beside_a_defect_is_a_fragment_not_a_false_call():
    """Differencing splits defects; the pieces must not be labelled spurious."""
    result = match(
        [make_candidate(35, 10, 45, 30)],   # just past the annotation's right edge
        [make_annotation(10, 10, 30, 30)],
    )
    assert result.labelled[0].label == FRAGMENT
    assert result.false_calls == []
    assert not result.labelled[0].trainable


def test_fragments_are_excluded_from_the_trainable_set():
    result = match(
        [
            make_candidate(12, 12, 28, 28),    # on the defect
            make_candidate(35, 10, 45, 30),    # fragment beside it
            make_candidate(200, 200, 220, 220) # genuine false call
        ],
        [make_annotation(10, 10, 30, 30)],
    )
    labels = {c.label for c in result.trainable}
    assert FRAGMENT not in labels
    assert len(result.trainable) == 2


def test_one_defect_hit_twice_is_only_detected_once():
    result = match(
        [make_candidate(10, 10, 30, 30), make_candidate(11, 11, 31, 31)],
        [make_annotation(10, 10, 30, 30)],
    )
    assert len(result.detected_annotations) == 1
    assert len(result.true_detections) == 2


def test_no_annotations_means_every_candidate_is_a_false_call():
    result = match([make_candidate(0, 0, 10, 10)], [])
    assert result.labelled[0].label == FALSE_CALL
