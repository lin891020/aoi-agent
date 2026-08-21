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
from datetime import datetime, timedelta

from aoi_agent.aoi.matching import match
from aoi_agent.aoi.simulator import DetectorConfig, detect
from aoi_agent.data.deeppcb import load_split
from aoi_agent.store.models import Board, CandidateRecord, ReviewDecision
from aoi_agent.vision.inference import ReVerifier

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


def _plan_assignments(all_annotations, rng: random.Random) -> list[tuple[str, str]]:
    """Decide every board's machine up front, so the ranking is global."""
    ranked = sorted(
        range(len(all_annotations)),
        key=lambda i: _open_share(all_annotations[i]),
        reverse=True,
    )
    suspect_count = int(len(ranked) * SUSPECT_TOP_SHARE)
    suspect = set(ranked[:suspect_count])

    others = _other_machines()
    assignments: list[tuple[str, str]] = []
    for index in range(len(all_annotations)):
        assignments.append(
            SUSPECT_MACHINE if index in suspect else rng.choice(others)
        )
    return assignments


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
    config = DetectorConfig()
    reverifier = reverifier or ReVerifier()

    pairs = load_split(split)
    if limit:
        pairs = pairs[:limit]

    counts = {"boards": 0, "candidates": 0, "decisions": 0}

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
            inspected_at=START + timedelta(hours=position * 0.4),
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
                )
            )
            counts["candidates"] += 1
            counts["decisions"] += 1

        if progress_every and (position + 1) % progress_every == 0:
            session.commit()
            print(f"  {position + 1}/{len(pairs)} boards, {counts['candidates']} candidates")

    session.commit()
    return counts
