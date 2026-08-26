"""The planted signals on the real annotations, without the model.

`tests/test_machine_events.py` holds the assignment planner on synthetic
boards. That was not enough: on the real split every control machine read as
an effect, because the split file lists boards design by design and the seed
had taken file order as time order. This runs the seeder's own ordering and
assignment over DeepPCB's real annotations -- no detector, no model, seconds
-- and asserts the three properties on ground-truth open share, which is the
quantity the seed actually moves.

Behind ``-m dataset``: it reads the annotation files.
"""

from __future__ import annotations

import random
import statistics

import pytest

from aoi_agent.data.deeppcb import load_split
from aoi_agent.store import seed


@pytest.fixture(scope="module")
def planted():
    try:
        pairs = load_split("test")[:500]
    except FileNotFoundError:
        pytest.skip("DeepPCB not cloned")
    rng = random.Random(7)
    pairs = [pairs[i] for i in seed.board_order(len(pairs), rng)]
    annotations = [p.load_annotations() for p in pairs]
    plan = seed._plan_assignments(annotations, rng)
    shares = [seed._open_share(a) for a in annotations]
    return plan, shares


def _sides(plan, shares, machine, position):
    before = [s for i, (m, s) in enumerate(zip(plan, shares)) if m[1] == machine and i < position]
    after = [s for i, (m, s) in enumerate(zip(plan, shares)) if m[1] == machine and i >= position]
    return before, after


def _gap_in_se(before, after):
    gap = statistics.mean(before) - statistics.mean(after)
    se = (statistics.pvariance(before) / len(before) + statistics.pvariance(after) / len(after)) ** 0.5
    return gap / se if se else float("inf")


@pytest.mark.dataset
def test_the_pool_carries_no_time_trend_once_shuffled(planted):
    """The defect this file exists for. Compare the first and second half of
    the run over every machine the seed does not touch by time."""
    plan, shares = planted
    untouched = [s for i, (m, s) in enumerate(zip(plan, shares)) if m[1] not in ("M32", "M11")]
    half = len(untouched) // 2
    assert abs(_gap_in_se(untouched[:half], untouched[half:])) < 3


@pytest.mark.dataset
def test_the_effect_reads_through_on_real_boards(planted):
    plan, shares = planted
    effect = next(e for e in seed.EVENTS if e.effect)
    before, after = _sides(plan, shares, effect.machine, effect.position)
    assert _gap_in_se(before, after) > 6, (statistics.mean(before), statistics.mean(after))


@pytest.mark.dataset
@pytest.mark.parametrize("event", [e for e in seed.EVENTS if not e.effect], ids=lambda e: e.machine)
def test_a_control_is_flat_on_real_boards(planted, event):
    plan, shares = planted
    before, after = _sides(plan, shares, event.machine, event.position)
    assert abs(_gap_in_se(before, after)) < 3, (event.machine, statistics.mean(before), statistics.mean(after))


@pytest.mark.dataset
def test_m22_still_holds_the_open_heaviest_fifth(planted):
    plan, shares = planted
    m22 = [s for m, s in zip(plan, shares) if m == seed.SUSPECT_MACHINE]
    rest = [s for m, s in zip(plan, shares) if m != seed.SUSPECT_MACHINE]
    assert min(m22) >= max(rest) - 1e-9 or statistics.mean(m22) > statistics.mean(rest) + 0.3
