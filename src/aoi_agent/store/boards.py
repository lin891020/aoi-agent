"""Read access to boards and candidates.

Thin functions returning plain dicts rather than ORM objects, so the MCP tools
serialise cleanly and never leak a live session.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sqlalchemy import select

from aoi_agent.data.deeppcb import load_split
from aoi_agent.store.models import Board, CandidateRecord, make_session_factory

_session_factory = None


def session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = make_session_factory()
    return _session_factory


@lru_cache(maxsize=1)
def _pairs_by_stem() -> dict:
    pairs = {}
    for split in ("trainval", "test"):
        try:
            for pair in load_split(split):
                pairs[pair.stem] = pair
        except FileNotFoundError:
            continue
    return pairs


def load_board_images(stem: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(template, test)`` for one board."""
    pair = _pairs_by_stem().get(stem)
    if pair is None:
        raise KeyError(f"board {stem!r} is not in the dataset")
    return pair.load_template(), pair.load_test()


def _as_dict(record: CandidateRecord, board: Board) -> dict:
    return {
        "reference": f"{board.stem}#{record.index_on_board}",
        "board_stem": board.stem,
        "index": record.index_on_board,
        "x1": record.x1, "y1": record.y1, "x2": record.x2, "y2": record.y2,
        "area": record.area,
        "predicted_class": record.predicted_class,
        "confidence": record.confidence,
        "false_call_probability": record.false_call_probability,
        "lot_id": board.lot_id,
        "line_id": board.line_id,
        "machine_id": board.machine_id,
        "shift": board.shift,
        "inspected_at": board.inspected_at.isoformat(),
    }


def resolve_candidate(reference: str) -> dict | None:
    """Look up a candidate by its ``<board>#<index>`` reference."""
    if "#" not in reference:
        return None
    stem, _, index = reference.partition("#")
    if not index.isdigit():
        return None

    with session_factory()() as session:
        row = session.execute(
            select(CandidateRecord, Board)
            .join(Board)
            .where(Board.stem == stem, CandidateRecord.index_on_board == int(index))
        ).first()
        return _as_dict(*row) if row else None


def candidates_for_board(stem: str) -> list[dict]:
    with session_factory()() as session:
        rows = session.execute(
            select(CandidateRecord, Board)
            .join(Board)
            .where(Board.stem == stem)
            .order_by(CandidateRecord.index_on_board)
        ).all()
        return [_as_dict(*row) for row in rows]


def sample_board_stems(limit: int = 10) -> list[str]:
    with session_factory()() as session:
        return list(
            session.execute(select(Board.stem).order_by(Board.id).limit(limit))
            .scalars()
            .all()
        )


def record_decision(
    reference: str,
    verdict: str,
    source: str,
    reviewer: str | None = None,
    rationale: str | None = None,
) -> bool:
    """Append a verdict to a candidate's decision history.

    Decisions accumulate rather than overwrite. A model call followed by an
    operator's correction is two rows, and the pair is exactly what the next
    training round needs -- an overwritten decision is a correction that never
    happened.
    """
    from aoi_agent.store.models import ReviewDecision

    if "#" not in reference:
        return False
    stem, _, index = reference.partition("#")
    if not index.isdigit():
        return False

    with session_factory()() as session:
        record = session.execute(
            select(CandidateRecord)
            .join(Board)
            .where(Board.stem == stem, CandidateRecord.index_on_board == int(index))
        ).scalar()
        if record is None:
            return False
        session.add(
            ReviewDecision(
                candidate_id=record.id,
                verdict=verdict,
                source=source,
                reviewer=reviewer,
                rationale=rationale,
            )
        )
        session.commit()
    return True


def corrections(limit: int = 100) -> list[dict]:
    """Candidates where a human overruled the model.

    These are the training set for the next revision.
    """
    from aoi_agent.store.models import ReviewDecision

    with session_factory()() as session:
        rows = session.execute(
            select(ReviewDecision, CandidateRecord, Board)
            .join(CandidateRecord, ReviewDecision.candidate_id == CandidateRecord.id)
            .join(Board, CandidateRecord.board_id == Board.id)
            .where(ReviewDecision.source == "human")
            .order_by(ReviewDecision.decided_at.desc())
            .limit(limit)
        ).all()

        return [
            {
                "reference": f"{board.stem}#{candidate.index_on_board}",
                "model_said": candidate.predicted_class,
                "model_confidence": candidate.confidence,
                "human_said": decision.verdict,
                "overruled": decision.verdict != candidate.predicted_class,
                "reviewer": decision.reviewer,
                "decided_at": decision.decided_at.isoformat() if decision.decided_at else None,
            }
            for decision, candidate, board in rows
        ]


def correction_summary(limit: int = 1000) -> dict:
    """Where the model gets corrected, aggregated.

    The list of corrections says what happened; this says what keeps happening.
    A class the operators overturn again and again is a training-set problem or
    a threshold problem, and it is invisible in a chronological list once there
    are more than a screenful.

    Counts human decisions only, so a class the model was never asked about
    does not appear as a perfect score.
    """
    rows = corrections(limit)
    pairs: dict[tuple[str, str], int] = {}
    by_model_class: dict[str, dict[str, int]] = {}

    for row in rows:
        key = (row["model_said"], row["human_said"])
        pairs[key] = pairs.get(key, 0) + 1
        bucket = by_model_class.setdefault(
            row["model_said"], {"total": 0, "overruled": 0}
        )
        bucket["total"] += 1
        bucket["overruled"] += int(row["overruled"])

    classes = sorted(
        by_model_class,
        key=lambda name: (-by_model_class[name]["overruled"], name),
    )
    return {
        "total": len(rows),
        "overruled": sum(1 for row in rows if row["overruled"]),
        "pairs": pairs,
        "by_model_class": [
            {
                "model_said": name,
                "total": by_model_class[name]["total"],
                "overruled": by_model_class[name]["overruled"],
                "overruled_share": (
                    by_model_class[name]["overruled"] / by_model_class[name]["total"]
                ),
                "corrected_to": sorted(
                    (
                        {"verdict": human, "count": count}
                        for (model, human), count in pairs.items()
                        if model == name and human != name
                    ),
                    key=lambda entry: -entry["count"],
                ),
            }
            for name in classes
        ],
    }
