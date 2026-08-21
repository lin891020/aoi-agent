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
