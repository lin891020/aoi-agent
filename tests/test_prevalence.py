"""Prevalence: which of the published figures survive a line that is not this
dataset, and which do not.

The test split is 36.8% genuine defects. No line is. What this module holds is
the separation the report claims -- that prevalence moves exactly one of the
three quantities, and that it is not the threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aoi_agent.vision.operating_point import sweep  # noqa: E402
from prevalence_report import (  # noqa: E402
    reweighted_review_reduction,
    wilson,
)

rng = np.random.default_rng(20260824)


def population(defects: int, false_calls: int):
    """Scores whose two groups are separated but overlapping, like the real ones."""
    p = np.concatenate([
        rng.beta(2, 5, defects),        # defects: low P(false_call), with a tail
        rng.beta(5, 2, false_calls),    # false calls: high, with a tail
    ])
    labels = np.concatenate([np.ones(defects, int), np.zeros(false_calls, int)])
    return p, labels.astype(bool)


def test_the_escape_rate_is_the_same_at_every_prevalence():
    """The claim the report makes first, and the reason a threshold transfers.

    `escapes / defects_total` scales its numerator and denominator by the same
    factor when defects are re-weighted, so the rate cannot move. Asserted here
    over two populations built with different mixes rather than by repeating
    the algebra.
    """
    rich_p, rich_defect = population(3000, 5000)     # 37.5% defects
    sparse_p, sparse_defect = population(3000, 300000)  # 1.0% defects

    for threshold in (0.5, 0.8, 0.915, 0.99):
        rich = float((rich_p[rich_defect] >= threshold).mean())
        sparse = float((sparse_p[sparse_defect] >= threshold).mean())
        # Same generator for the defect group, so the two are the same draw.
        assert rich == pytest.approx(sparse, abs=0.03), (
            f"escape rate moved with prevalence at {threshold}"
        )


def test_the_escape_rate_is_exactly_invariant_under_reweighting():
    """The exact version: one population, re-weighted rather than resampled.
    Nothing here is approximate, so nothing here needs a tolerance."""
    p, is_defect = population(3000, 5000)
    points = sweep(p, is_defect.astype(int), false_call_index=0)

    for point in points[::100]:
        recomputed = float((p[is_defect] >= point.threshold).mean())
        assert recomputed == pytest.approx(point.escape_rate, abs=1e-12)


def test_review_reduction_rises_as_the_line_gets_cleaner():
    """The second claim: the 56.2% is a floor for a line cleaner than this
    dataset, not a ceiling. Fewer genuine defects means more false calls, and
    false calls are the population the model dismisses."""
    p, is_defect = population(3000, 5000)
    threshold = 0.915

    reductions = [
        reweighted_review_reduction(p, is_defect, threshold, prevalence)
        for prevalence in (0.368, 0.10, 0.01)
    ]

    # Strictly, and by a visible margin. `== sorted(...)` was the first form of
    # this and it is true of a constant list -- so a reweighting that ignored
    # its `prevalence` argument entirely passed it. Caught by mutation.
    assert reductions[0] < reductions[1] < reductions[2]
    assert reductions[2] > reductions[0] * 1.5, (
        "prevalence has to move this figure, not merely fail to reverse it"
    )


def test_the_reweighting_agrees_with_the_measurement_at_its_own_prevalence():
    """The check that the arithmetic is doing what it says: fed the dataset's
    own prevalence, it has to reproduce the swept figure."""
    p, is_defect = population(3000, 5000)
    threshold = 0.915
    measured = float((p >= threshold).mean())

    assert reweighted_review_reduction(
        p, is_defect, threshold, float(is_defect.mean())
    ) == pytest.approx(measured, abs=1e-12)


# ---------------------------------------------------------------------------
# The interval, which is the part that does not transfer
# ---------------------------------------------------------------------------

def test_the_projects_own_escape_figure_does_not_exclude_exceeding_its_budget():
    """14 escapes in 2,997 defects is 0.47% on this split and that is not in
    doubt. As an estimate of the rate on unseen defects -- the only reading
    that justifies deploying a threshold -- the interval runs past 0.5%.

    Every escape figure this project published before 2026-08-24 was a point
    estimate with no interval beside it.
    """
    low, high = wilson(14, 2997)

    assert low < 0.0047 < high
    assert high > 0.005, "the interval must not be read as confirming the budget"


def test_a_small_pilot_cannot_settle_the_budget_either_way():
    """Thirty defects is a month on a good line and says nothing at all."""
    low, high = wilson(0, 30)

    assert low == 0.0
    assert high > 0.05, "a month of a clean line bounds the rate at ~11%"


def test_the_interval_stays_inside_zero_and_one():
    """Why Wilson and not the normal approximation, which a spreadsheet reaches
    for: at a handful of successes it puts the lower bound below zero."""
    for successes, trials in [(0, 30), (1, 300), (3, 100), (14, 2997), (0, 1)]:
        low, high = wilson(successes, trials)
        assert 0.0 <= low <= high <= 1.0
