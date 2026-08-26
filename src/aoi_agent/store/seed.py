"""Populate the store from DeepPCB boards plus simulated production context.

The boards, the defects and the AOI candidates are real. The factory context is
not -- public defect datasets ship no lot numbers or machine ids -- so it is
generated here.

The generation is not uniform noise. Machine ``L2-M22`` is given a raised share
of boards that genuinely contain ``open`` defects, standing in for a worn etch
station. Without a planted pattern the context queries would have nothing to
find and the agent's use of them could not be judged. The pattern is
deterministic and documented rather than hidden, so any conclusion the agent
draws from it can be checked against what was planted.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from aoi_agent.aoi.matching import match
from aoi_agent.aoi.simulator import DetectorConfig, detect
from aoi_agent.data.deeppcb import load_split
from aoi_agent.provenance import DecisionProvenance, code_version
from aoi_agent.store import events as machine_events
from aoi_agent.store.models import Board, CandidateRecord, ReviewDecision
from aoi_agent.vision.inference import DEFAULT_DISMISS_THRESHOLD, ReVerifier

LINES = {"L1": ["M11", "M12"], "L2": ["M21", "M22"], "L3": ["M31", "M32"]}
SHIFTS = ["A", "B", "C"]

#: The station with a planted problem: boards whose defects are most heavily
#: weighted towards ``open`` are assigned here, standing in for a worn etch
#: station.
#:
#: Assignment is by *rank on open share*, not by open count. Two earlier
#: attempts biased on count and on a share threshold, and both leaked: boards
#: carrying more opens carry more of every class, so the suspect machine came
#: out top for ``short`` as well and the planted cause was indistinguishable
#: from "this machine simply runs worse material". Ranking on share holds total
#: defect count roughly constant across machines by construction, which is what
#: makes the signal specific.
SUSPECT_MACHINE = ("L2", "M22")
SUSPECT_TOP_SHARE = 0.20
"""Fraction of boards, most open-weighted first, sent to the suspect machine."""

START = datetime(2026, 8, 1, 6, 0)
BOARD_INTERVAL = timedelta(hours=0.4)
"""One board every 24 minutes, so 500 boards span about nine days."""


@dataclass(frozen=True)
class PlantedEvent:
    """Something the seeder says happened to a machine, at board ``position``.

    ``effect`` is whether the assignment changes at that point. Exactly one
    event has one; the other three are controls, and they are the reason the
    tool that reads this table can be wrong -- without them it would be
    scored on "is there an event", not on "did the event matter".
    """

    line: str
    machine: str
    kind: str
    position: int
    effect: bool
    note: str

    @property
    def happened_at(self) -> datetime:
        return START + BOARD_INTERVAL * self.position


#: The second planted signal, and the three that are not signals. M22 carries
#: the first (above) and gets no event: splitting its boards in time would
#: weaken every measurement that already depends on it.
#:
#: The effect is implemented through the seeder's one lever -- which board
#: goes to which machine -- and nothing else. Before its event M32 receives
#: the open-heavy boards among those not already sent to M22; after it, the
#: open-light ones. The defects themselves are DeepPCB's and are not touched.
EVENTS: tuple[PlantedEvent, ...] = (
    PlantedEvent("L3", "M32", "parameter_change", 250, True,
                 "etch pressure re-set after C-shift handover"),
    PlantedEvent("L3", "M31", "lamp_replaced", 200, False,
                 "ring light replaced at scheduled hours"),
    PlantedEvent("L2", "M21", "maintenance", 300, False,
                 "quarterly PM"),
    PlantedEvent("L1", "M12", "nozzle_cleaned", 350, False,
                 "nozzle cleaned, no parameter change"),
)

#: What share of the non-suspect boards the effect machine receives on each
#: side of its event. One machine out of the five that are not M22, so this
#: is the share a uniform draw would give it -- the effect moves *which*
#: boards it gets, never how many.
EFFECT_SHARE = 1 / 5

#: The machine that absorbs the effect's mirror image, so the controls do not.
#:
#: Boards are conserved: if M32 takes the open-heavy fifth of the boards
#: inspected before its event, the boards left for everyone else are
#: open-light before the event and open-heavy after it -- the effect's mirror,
#: smeared over four machines. The first draft did exactly that and the
#: control tests caught it: every control read as an effect. So the mirror is
#: given to one named machine with no event of its own. It takes the
#: open-*light* fifth before and the open-heavy fifth after, which trims the
#: pool symmetrically and leaves the three controls unbiased in both windows.
#: M11 will therefore look like it changed at M32's date. It has no event, so
#: the tool must refuse to anchor on it -- which is a fourth thing the tool
#: can be wrong about, and is tested as one.
MIRROR_MACHINE = ("L1", "M11")


def _open_share(annotations) -> float:
    if not annotations:
        return 0.0
    return sum(1 for a in annotations if a.class_name == "open") / len(annotations)


def _other_machines() -> list[tuple[str, str]]:
    return [
        (line, machine)
        for line, machines in LINES.items()
        for machine in machines
        if (line, machine) != SUSPECT_MACHINE
    ]


def _effect_event(events) -> PlantedEvent | None:
    effects = [e for e in events if e.effect]
    if len(effects) > 1:
        raise ValueError("one planted effect, not several: a second one would be "
                         "indistinguishable from the first in any measurement")
    return effects[0] if effects else None


def _plan_assignments(
    all_annotations, rng: random.Random, events=EVENTS
) -> list[tuple[str, str]]:
    """Decide every board's machine up front, so the ranking is global.

    Two planted signals, both by assignment. The first is M22: the top
    ``SUSPECT_TOP_SHARE`` of boards by open share, ranked globally. The second
    is the effect event's machine: among the boards M22 did not take, the ones
    inspected *before* the event are ranked by open share and it receives the
    top ``EFFECT_SHARE`` of them; the ones inspected after are ranked the same
    way and it receives the *bottom* ``EFFECT_SHARE``. Every other board goes
    to a uniform draw over the remaining machines. Controls change nothing.
    """
    # Ties are broken at random, not by position. Open share is a coarse
    # quantity -- a board with three defects can only be 0, 1/3, 2/3 or 1 --
    # so a stable sort over board order would hand every tie to the earlier
    # board, and the top fifth would lean towards the start of the run. That
    # is a time trend in a seed that claims to plant none, and it was found
    # because the control machines drifted upward across every event date.
    order = list(range(len(all_annotations)))
    rng.shuffle(order)
    share = {i: _open_share(all_annotations[i]) for i in order}
    ranked = sorted(order, key=share.__getitem__, reverse=True)
    suspect = set(ranked[: int(len(ranked) * SUSPECT_TOP_SHARE)])

    assignments: dict[int, tuple[str, str]] = {i: SUSPECT_MACHINE for i in suspect}
    rest = {i for i in range(len(all_annotations)) if i not in suspect}

    effect = _effect_event(events)
    others = _other_machines()
    if effect is not None:
        target = (effect.line, effect.machine)
        if MIRROR_MACHINE in (target, SUSPECT_MACHINE):
            raise ValueError("the mirror machine must be neither the effect nor M22")
        others = [m for m in others if m not in (target, MIRROR_MACHINE)]
        before = sorted((i for i in ranked if i in rest and i < effect.position),
                        key=share.__getitem__, reverse=True)
        after = sorted((i for i in ranked if i in rest and i >= effect.position),
                       key=share.__getitem__, reverse=True)
        # Before: the effect machine takes the open-heavy end, the mirror the
        # open-light end. After: the reverse. Each takes the same count.
        for side in (before, after):
            take = int(len(side) * EFFECT_SHARE)
            heavy, light = side[:take], side[len(side) - take:]
            for i in (heavy if side is before else light):
                assignments[i] = target
            for i in (light if side is before else heavy):
                assignments[i] = MIRROR_MACHINE

    for i in sorted(rest):
        if i not in assignments:
            assignments[i] = rng.choice(others)
    return [assignments[i] for i in range(len(all_annotations))]


def plant_events(session, events=EVENTS) -> int:
    """Write the planted events. Call after the boards, since the guard reads them."""
    for event in events:
        machine_events.record(
            event.machine, event.kind, event.happened_at,
            note=event.note, recorded_by=machine_events.SEEDED, session=session,
        )
    return len(events)


def seed(
    session,
    split: str = "test",
    limit: int | None = None,
    reverifier: ReVerifier | None = None,
    seed_value: int = 7,
    progress_every: int = 50,
) -> dict[str, int]:
    """Fill the store with one split's boards, candidates and model verdicts."""
    rng = random.Random(seed_value)
    config = DetectorConfig(register=True)
    reverifier = reverifier or ReVerifier()

    # Every row this writes is an automated decision and carries what produced
    # it. Only the dismissal threshold is in force here: seeding records the
    # model's reading of each region, it does not route one, so listing the
    # graph's thresholds would claim a decision path this code never took.
    provenance = DecisionProvenance(
        model_digest=reverifier.checkpoint_digest,
        thresholds={"dismiss": DEFAULT_DISMISS_THRESHOLD},
        code_version=code_version(),
    )

    pairs = load_split(split)
    if limit:
        pairs = pairs[:limit]

    counts = {"boards": 0, "candidates": 0, "decisions": 0, "events": 0}

    assignments = _plan_assignments([p.load_annotations() for p in pairs], rng)

    for position, pair in enumerate(pairs):
        template = pair.load_template()
        test = pair.load_test()
        annotations = pair.load_annotations()

        candidates = detect(template, test, config)
        result = match(candidates, annotations)
        verdicts = reverifier.classify_batch(template, test, candidates)

        line, machine = assignments[position]
        board = Board(
            stem=pair.stem,
            split=split,
            lot_id=f"LOT-{2608000 + position // 25}",
            line_id=line,
            machine_id=machine,
            shift=SHIFTS[(position // 8) % len(SHIFTS)],
            inspected_at=START + BOARD_INTERVAL * position,
        )
        session.add(board)
        counts["boards"] += 1

        for index, (labelled, verdict) in enumerate(
            zip(result.labelled, verdicts, strict=True)
        ):
            record = CandidateRecord(
                board=board,
                index_on_board=index,
                x1=int(labelled.candidate.x1),
                y1=int(labelled.candidate.y1),
                x2=int(labelled.candidate.x2),
                y2=int(labelled.candidate.y2),
                area=int(labelled.candidate.area),
                predicted_class=verdict.predicted_class,
                confidence=verdict.confidence,
                false_call_probability=verdict.false_call_probability,
                ground_truth=labelled.label,
            )
            session.add(record)
            session.add(
                ReviewDecision(
                    candidate=record,
                    verdict=verdict.predicted_class,
                    source="model",
                    rationale=f"confidence {verdict.confidence:.3f}",
                    **provenance.columns(),
                )
            )
            counts["candidates"] += 1
            counts["decisions"] += 1

        if progress_every and (position + 1) % progress_every == 0:
            session.commit()
            print(f"  {position + 1}/{len(pairs)} boards, {counts['candidates']} candidates")

    # After the boards, because the write guard reads the machines off them.
    counts["events"] = plant_events(session)
    session.commit()
    return counts
