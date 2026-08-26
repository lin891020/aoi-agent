"""Machine events, and the second planted signal that gives them something to say.

Two halves. The first is the store: an event is written only about a machine
the store has inspected on, ``after`` means the newest event of that kind,
and the validator's domain is what the table holds. The second is the seed:
the effect event changes *which* boards its machine receives and nothing
else, the three controls change nothing, and M22's signal is untouched. The
second half is the one that makes the tool measurable -- a tool read against
a seed with no controls is scored on "is there an event", not on "did it
matter".

The seed tests run the assignment planner on synthetic annotations. They do
not run the detector or the model, and they do not need the dataset.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pytest

from aoi_agent.store import events, seed
from aoi_agent.store.models import Board, create_all, make_session_factory
from aoi_agent.store import boards as boards_module

# ---- a store with three machines and no events --------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'events.db'}"
    create_all(url)
    factory = make_session_factory(url)
    monkeypatch.setattr(boards_module, "_session_factory", factory)
    with factory() as session:
        for n, (line, machine) in enumerate([("L1", "M11"), ("L2", "M22"), ("L3", "M32")]):
            session.add(Board(
                stem=f"2008529{n}", split="test", lot_id="LOT-1", line_id=line,
                machine_id=machine, shift="A",
                inspected_at=datetime(2026, 8, 1, 6) + timedelta(hours=n),
            ))
        session.commit()
    return factory


def test_an_event_about_a_machine_nobody_inspected_on_is_refused(store):
    with pytest.raises(ValueError, match="M99"):
        events.record("M99", "maintenance", datetime(2026, 8, 2))
    assert events.count() == 0


def test_an_event_needs_a_kind(store):
    with pytest.raises(ValueError, match="kind"):
        events.record("M32", "  ", datetime(2026, 8, 2))


def test_a_recorded_event_comes_back_newest_first_and_says_who_recorded_it(store):
    events.record("M32", "parameter_change", datetime(2026, 8, 2, 10))
    events.record("M32", "parameter_change", datetime(2026, 8, 5, 10), note="again")
    events.record("M11", "maintenance", datetime(2026, 8, 3, 10))

    rows = events.events_for("M32")
    assert [r["happened_at"][:10] for r in rows] == ["2026-08-05", "2026-08-02"]
    assert rows[0]["note"] == "again"
    assert {r["recorded_by"] for r in rows} == {events.SEEDED}
    assert len(events.events_for(kind="maintenance")) == 1
    assert len(events.events_for()) == 3


def test_after_means_the_newest_event_of_that_kind(store):
    """One rule for the anchor. Two tools resolving it differently would
    disagree only on machines with more than one event -- the ones asked about."""
    events.record("M32", "parameter_change", datetime(2026, 8, 2, 10))
    events.record("M32", "parameter_change", datetime(2026, 8, 5, 10))
    events.record("M32", "maintenance", datetime(2026, 8, 7, 10))

    assert events.anchor_for("M32", "parameter_change") == datetime(2026, 8, 5, 10)
    assert events.anchor_for("M32", "lamp_replaced") is None
    assert events.anchor_for("M11", "parameter_change") is None


def test_the_validators_domain_is_what_the_table_holds_not_the_tuple(store):
    assert events.kinds_present() == set()
    events.record("M32", "parameter_change", datetime(2026, 8, 2))
    assert events.kinds_present() == {"parameter_change"}
    assert "maintenance" in events.KINDS and "maintenance" not in events.kinds_present()


# ---- the seed: one effect, three controls, and M22 untouched -------------


class _Ann:
    def __init__(self, class_name):
        self.class_name = class_name


def _boards(n: int, rng: random.Random):
    """Synthetic annotation lists with a spread of open shares."""
    out = []
    for _ in range(n):
        k = rng.randint(1, 6)
        opens = rng.randint(0, k)
        out.append([_Ann("open")] * opens + [_Ann("short")] * (k - opens))
    return out


def _share(boards, idx):
    a = boards[idx]
    return sum(1 for x in a if x.class_name == "open") / len(a) if a else 0.0


@pytest.fixture
def planned():
    rng = random.Random(7)
    boards = _boards(500, rng)
    with_events = seed._plan_assignments(boards, random.Random(7), events=seed.EVENTS)
    without = seed._plan_assignments(boards, random.Random(7), events=())
    return boards, with_events, without


def test_exactly_one_planted_event_has_an_effect():
    assert sum(1 for e in seed.EVENTS if e.effect) == 1
    assert len(seed.EVENTS) == 4
    assert all(e.machine != seed.SUSPECT_MACHINE[1] for e in seed.EVENTS), (
        "M22 carries the first signal and gets no event"
    )


def test_m22s_signal_is_the_same_with_and_without_events(planned):
    """The first planted signal is not weakened by the second."""
    _boards_, with_events, without = planned
    m22_with = {i for i, m in enumerate(with_events) if m == seed.SUSPECT_MACHINE}
    m22_without = {i for i, m in enumerate(without) if m == seed.SUSPECT_MACHINE}
    assert m22_with == m22_without
    assert len(m22_with) == int(500 * seed.SUSPECT_TOP_SHARE)


def test_the_effect_machine_differs_before_and_after_its_event(planned):
    boards, with_events, _ = planned
    effect = next(e for e in seed.EVENTS if e.effect)
    target = (effect.line, effect.machine)
    before = [_share(boards, i) for i, m in enumerate(with_events) if m == target and i < effect.position]
    after = [_share(boards, i) for i, m in enumerate(with_events) if m == target and i >= effect.position]

    assert before and after
    assert sum(before) / len(before) > sum(after) / len(after) + 0.3, (
        "the effect is supposed to be large enough to read through a Wilson interval"
    )


def test_the_effect_moves_which_boards_not_how_many(planned):
    boards, with_events, without = planned
    effect = next(e for e in seed.EVENTS if e.effect)
    target = (effect.line, effect.machine)
    n_with = sum(1 for m in with_events if m == target)
    n_without = sum(1 for m in without if m == target)
    # Uniform draw over five gives ~20% of the non-suspect boards; the effect
    # gives exactly 20% per side. Same order of magnitude, by construction.
    assert abs(n_with - n_without) < 0.25 * n_without


@pytest.mark.parametrize("event", [e for e in seed.EVENTS if not e.effect], ids=lambda e: e.machine)
def test_a_control_machine_does_not_differ_across_its_event(planned, event):
    """The property that makes the tool falsifiable. If a control reads as
    an effect, the tool has nothing to be wrong about."""
    boards, with_events, _ = planned
    target = (event.line, event.machine)
    before = [_share(boards, i) for i, m in enumerate(with_events) if m == target and i < event.position]
    after = [_share(boards, i) for i, m in enumerate(with_events) if m == target and i >= event.position]
    assert before and after
    # A fixed bound is the wrong shape here: each side holds ~40 boards, so
    # the difference of two means has a standard error near 0.05 and a fixed
    # 0.15 is a coin at three sigma across three controls. The bound is the
    # samples' own: the gap must sit inside three standard errors, which is
    # what "no effect" means, and the effect's own gap sits far outside it.
    import statistics

    gap = abs(statistics.mean(before) - statistics.mean(after))
    se = (statistics.pvariance(before) / len(before)
          + statistics.pvariance(after) / len(after)) ** 0.5
    assert gap < 3 * se, (event.machine, gap, se, len(before), len(after))


def test_the_mirror_machine_changes_at_the_effects_date_and_has_no_event(planned):
    """Boards are conserved, so the effect's mirror has to land somewhere. It
    lands on one named machine with no event, not smeared over the controls --
    and a tool that anchored on M11 would have nothing to anchor on."""
    boards, with_events, _ = planned
    effect = next(e for e in seed.EVENTS if e.effect)
    mirror = seed.MIRROR_MACHINE
    before = [_share(boards, i) for i, m in enumerate(with_events) if m == mirror and i < effect.position]
    after = [_share(boards, i) for i, m in enumerate(with_events) if m == mirror and i >= effect.position]
    assert sum(after) / len(after) > sum(before) / len(before) + 0.3
    assert all(e.machine != mirror[1] for e in seed.EVENTS)


def test_two_planted_effects_are_refused():
    two = seed.EVENTS + (seed.PlantedEvent("L1", "M11", "maintenance", 100, True, "x"),)
    with pytest.raises(ValueError, match="one planted effect"):
        seed._plan_assignments(_boards(50, random.Random(1)), random.Random(1), events=two)


def test_planted_events_land_in_the_store_after_the_boards(store):
    with store() as session:
        # the fixture's machines: M11, M22, M32 -- so M21 and M12 would be refused
        # and this asserts the guard is really on the seed path.
        with pytest.raises(ValueError, match="M31|M21|M12"):
            seed.plant_events(session)
        session.rollback()
        only_known = tuple(e for e in seed.EVENTS if e.machine in ("M11", "M22", "M32"))
        assert seed.plant_events(session, events=only_known) == len(only_known)
        session.commit()
    assert events.count() == len(only_known)
    assert events.anchor_for("M32", "parameter_change") == next(
        e for e in seed.EVENTS if e.machine == "M32"
    ).happened_at
