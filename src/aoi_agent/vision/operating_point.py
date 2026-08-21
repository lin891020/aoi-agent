"""Operating-point analysis for AOI re-verification.

Accuracy is the wrong headline for quality inspection. Dismissing a real
defect ships a bad board; keeping a false call costs an operator a few seconds.
The two errors are not interchangeable, so the model is reported as a curve
rather than a number:

    given an escape rate the line is willing to accept,
    how much of the manual review queue disappears?

Decision rule: a candidate is dismissed when the model's probability that it is
a false call reaches the threshold. Everything else goes to a human, which is
what happens today for every single candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OperatingPoint:
    """What one dismissal threshold buys and what it costs."""

    threshold: float
    escape_rate: float
    """Share of genuinely defective candidates the model wrongly dismissed."""

    review_reduction: float
    """Share of the manual review queue removed."""

    reviewed: int
    dismissed: int
    escapes: int
    defects_total: int
    false_calls_dismissed: int
    false_calls_total: int

    @property
    def false_call_recall(self) -> float:
        """Share of false calls correctly dismissed."""
        return (
            self.false_calls_dismissed / self.false_calls_total
            if self.false_calls_total
            else 0.0
        )


def sweep(
    false_call_probability: np.ndarray,
    labels: np.ndarray,
    false_call_index: int,
    thresholds: np.ndarray | None = None,
) -> list[OperatingPoint]:
    """Evaluate every dismissal threshold.

    ``false_call_probability`` is the model's P(false_call) per candidate and
    ``labels`` the true label indices.
    """
    if thresholds is None:
        thresholds = np.unique(
            np.concatenate([np.linspace(0.0, 1.0, 1001), false_call_probability])
        )

    is_defect = labels != false_call_index
    defects_total = int(is_defect.sum())
    false_calls_total = int((~is_defect).sum())
    total = len(labels)

    points = []
    for threshold in thresholds:
        dismissed = false_call_probability >= threshold
        escapes = int((dismissed & is_defect).sum())
        points.append(
            OperatingPoint(
                threshold=float(threshold),
                escape_rate=escapes / defects_total if defects_total else 0.0,
                review_reduction=int(dismissed.sum()) / total if total else 0.0,
                reviewed=total - int(dismissed.sum()),
                dismissed=int(dismissed.sum()),
                escapes=escapes,
                defects_total=defects_total,
                false_calls_dismissed=int((dismissed & ~is_defect).sum()),
                false_calls_total=false_calls_total,
            )
        )
    return points


def best_at_escape_budget(
    points: list[OperatingPoint], max_escape_rate: float
) -> OperatingPoint | None:
    """The threshold that clears the most review queue within an escape budget.

    Returns ``None`` when no threshold meets the budget -- which is a real
    answer, not an error: it means the model cannot be deployed at that
    tolerance.
    """
    affordable = [p for p in points if p.escape_rate <= max_escape_rate]
    if not affordable:
        return None
    return max(affordable, key=lambda p: p.review_reduction)


def system_escape_rate(
    reverification_escape_rate: float, aoi_stage_escape_rate: float
) -> float:
    """Total share of defects that reach the customer.

    The re-verification model only ever sees what the AOI stage flagged. Any
    defect the AOI missed is already gone, and no threshold recovers it. The
    honest number for the line is the union of both stages.
    """
    caught_by_aoi = 1.0 - aoi_stage_escape_rate
    return aoi_stage_escape_rate + caught_by_aoi * reverification_escape_rate
