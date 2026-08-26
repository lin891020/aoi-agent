"""Interval arithmetic the reports share, so two of them cannot disagree.

`wilson` lived in `scripts/prevalence_report.py` until 2026-08-26 and was
imported from nowhere else, which was fine while one report needed it. It moved
here when a second one did: `class_escape_report.py` publishes a per-class
interval, and a second copy of an interval formula is a second answer to "how
uncertain is this rate" -- the same argument that keeps the standing-disposition
rule in one place.
"""

from __future__ import annotations

import math


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """A Wilson score interval, which stays inside [0, 1] at small counts.

    The normal approximation is what a spreadsheet reaches for and it is wrong
    in exactly the regime these tables are about -- at 3 escapes in 100 it gives
    a lower bound below zero.
    """
    if trials == 0:
        return (0.0, 1.0)
    phat = successes / trials
    denominator = 1 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denominator
    spread = (
        z
        * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))
