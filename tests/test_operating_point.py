import numpy as np
import pytest

from aoi_agent.vision.operating_point import (
    best_at_escape_budget,
    sweep,
    system_escape_rate,
)


@pytest.fixture
def perfect_predictions():
    """A model that separates the two populations completely."""
    labels = np.array([0, 0, 0, 1, 1, 1])       # 0 = false_call
    probability = np.array([0.99, 0.98, 0.97, 0.02, 0.01, 0.03])
    return probability, labels


def test_a_perfect_model_removes_the_false_calls_without_escapes(perfect_predictions):
    probability, labels = perfect_predictions
    best = best_at_escape_budget(sweep(probability, labels, 0), max_escape_rate=0.0)

    assert best is not None
    assert best.escapes == 0
    assert best.review_reduction == pytest.approx(0.5)   # 3 of 6 dismissed
    assert best.false_call_recall == pytest.approx(1.0)


def test_dismissing_nothing_is_always_available():
    """Threshold above 1.0 keeps every candidate, so escape rate is zero."""
    labels = np.array([0, 1])
    probability = np.array([0.5, 0.5])
    points = sweep(probability, labels, 0, thresholds=np.array([1.01]))

    assert points[0].escapes == 0
    assert points[0].review_reduction == 0.0


def test_review_reduction_rises_as_the_threshold_falls():
    labels = np.array([0, 0, 1, 1])
    probability = np.array([0.9, 0.6, 0.4, 0.1])
    points = sweep(probability, labels, 0, thresholds=np.array([0.95, 0.7, 0.5, 0.2]))

    reductions = [p.review_reduction for p in points]
    assert reductions == sorted(reductions)


def test_escape_budget_is_respected():
    labels = np.array([0] * 10 + [1] * 10)
    probability = np.concatenate([np.linspace(0.5, 1.0, 10), np.linspace(0.0, 0.9, 10)])
    points = sweep(probability, labels, 0)

    for budget in (0.0, 0.1, 0.3):
        best = best_at_escape_budget(points, budget)
        assert best is None or best.escape_rate <= budget


def test_unreachable_budget_returns_none():
    """A model that cannot meet the tolerance says so instead of pretending."""
    labels = np.array([1, 1])          # only defects
    probability = np.array([1.0, 1.0]) # dismissed at every threshold
    points = sweep(probability, labels, 0, thresholds=np.array([0.0, 0.5]))

    assert best_at_escape_budget(points, max_escape_rate=0.0) is None


def test_system_escape_rate_compounds_both_stages():
    # 5% missed by AOI; of the 95% that got through, 1% wrongly dismissed
    assert system_escape_rate(0.01, 0.05) == pytest.approx(0.05 + 0.95 * 0.01)


def test_system_escape_rate_is_bounded_by_the_aoi_stage():
    """A flawless re-verifier cannot beat the detector feeding it."""
    assert system_escape_rate(0.0, 0.05) == pytest.approx(0.05)
    assert system_escape_rate(1.0, 0.05) == pytest.approx(1.0)
