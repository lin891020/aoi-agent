"""MCP server exposing production context.

Deliberately **not** a text-to-SQL tool. The model fills typed parameters on a
fixed set of queries instead of writing SQL.

The reason is the failure mode, not the difficulty. A generated query that is
syntactically valid but semantically wrong returns a plausible-looking number
with no error, and in a quality-disposition context a plausible wrong number is
worse than a crash: it gets acted on. Parameter filling exercises the same
tool-calling ability while keeping the query set reviewable.
"""

from __future__ import annotations

from datetime import timedelta

from mcp.server.mcpserver import MCPServer
from sqlalchemy import func, select

from aoi_agent.store.boards import session_factory
from aoi_agent.store.models import Board, CandidateRecord

mcp = MCPServer("aoi-production")

DEFECT_CLASSES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]


@mcp.tool()
def query_defect_history(
    lot_id: str | None = None,
    line_id: str | None = None,
    machine_id: str | None = None,
    defect_type: str | None = None,
    days: int = 7,
) -> dict:
    """Count defects by class over a slice of recent production.

    Use this to judge whether a defect on one board is isolated or part of a
    pattern. Leave a filter unset to include everything.

    Args:
        lot_id: Restrict to one lot, for example ``LOT-2608003``.
        line_id: Restrict to one line: ``L1``, ``L2`` or ``L3``.
        machine_id: Restrict to one machine, for example ``M22``.
        defect_type: Restrict to one class: open, short, mousebite, spur,
            copper or pin-hole.
        days: How far back to look from the most recent inspection.
    """
    if defect_type and defect_type not in DEFECT_CLASSES:
        return {"error": f"unknown defect_type {defect_type!r}; expected one of {DEFECT_CLASSES}"}

    with session_factory()() as session:
        latest = session.execute(select(func.max(Board.inspected_at))).scalar()
        if latest is None:
            return {"error": "the store is empty; run scripts/seed_store.py"}
        since = latest - timedelta(days=days)

        query = (
            select(CandidateRecord.predicted_class, func.count())
            .join(Board)
            .where(Board.inspected_at >= since)
            .where(CandidateRecord.predicted_class != "false_call")
            .group_by(CandidateRecord.predicted_class)
        )
        boards_query = select(func.count(func.distinct(Board.id))).where(
            Board.inspected_at >= since
        )

        for column, value in (
            (Board.lot_id, lot_id),
            (Board.line_id, line_id),
            (Board.machine_id, machine_id),
        ):
            if value:
                query = query.where(column == value)
                boards_query = boards_query.where(column == value)
        if defect_type:
            query = query.where(CandidateRecord.predicted_class == defect_type)

        counts = dict(session.execute(query).all())
        boards = session.execute(boards_query.join(CandidateRecord).distinct()).scalar() or 0

    total = sum(counts.values())
    return {
        "filters": {
            "lot_id": lot_id, "line_id": line_id, "machine_id": machine_id,
            "defect_type": defect_type, "days": days,
        },
        "window_end": latest.isoformat(),
        "boards_inspected": boards,
        "defects_total": total,
        "defects_per_board": round(total / boards, 2) if boards else 0.0,
        "by_class": counts,
    }


@mcp.tool()
def query_machine_stats(defect_type: str, days: int = 14) -> dict:
    """Compare every machine's rate for one defect class.

    Use this when a defect might be traceable to a station. Returns each
    machine's rate for the class, ranked worst first.

    Two rates are reported and they answer different questions.
    ``per_board`` is the raw count and moves with anything that raises total
    defects, so a machine running harder material looks worse on every class at
    once. ``share_of_defects`` is the class as a fraction of that machine's own
    defects, which stays flat unless the machine has a problem specific to this
    class. Prefer ``share_of_defects`` when deciding whether a station is the
    cause; use ``per_board`` to judge how much it costs.

    Args:
        defect_type: One of open, short, mousebite, spur, copper, pin-hole.
        days: How far back to look from the most recent inspection.
    """
    if defect_type not in DEFECT_CLASSES:
        return {"error": f"unknown defect_type {defect_type!r}; expected one of {DEFECT_CLASSES}"}

    with session_factory()() as session:
        latest = session.execute(select(func.max(Board.inspected_at))).scalar()
        if latest is None:
            return {"error": "the store is empty; run scripts/seed_store.py"}
        since = latest - timedelta(days=days)

        boards = dict(
            session.execute(
                select(Board.line_id + "-" + Board.machine_id, func.count(func.distinct(Board.id)))
                .where(Board.inspected_at >= since)
                .group_by(Board.line_id, Board.machine_id)
            ).all()
        )
        defects = dict(
            session.execute(
                select(Board.line_id + "-" + Board.machine_id, func.count())
                .join(CandidateRecord)
                .where(Board.inspected_at >= since)
                .where(CandidateRecord.predicted_class == defect_type)
                .group_by(Board.line_id, Board.machine_id)
            ).all()
        )
        all_defects = dict(
            session.execute(
                select(Board.line_id + "-" + Board.machine_id, func.count())
                .join(CandidateRecord)
                .where(Board.inspected_at >= since)
                .where(CandidateRecord.predicted_class != "false_call")
                .group_by(Board.line_id, Board.machine_id)
            ).all()
        )

    rows = []
    for machine, count in boards.items():
        this_class = defects.get(machine, 0)
        every_class = all_defects.get(machine, 0)
        rows.append(
            {
                "machine": machine,
                "boards": count,
                "defects": this_class,
                "per_board": round(this_class / count, 3) if count else 0.0,
                "share_of_defects": round(this_class / every_class, 3) if every_class else 0.0,
            }
        )
    rows.sort(key=lambda r: r["share_of_defects"], reverse=True)

    total_this_class = sum(defects.values())
    total_all = sum(all_defects.values())
    return {
        "defect_type": defect_type,
        "days": days,
        "fleet_average_per_board": round(
            sum(r["per_board"] for r in rows) / len(rows), 3
        ) if rows else 0.0,
        "fleet_share_of_defects": round(total_this_class / total_all, 3) if total_all else 0.0,
        "machines": rows,
    }


@mcp.tool()
def query_board_context(board: str) -> dict:
    """Where and when one board was made, and what else came off that lot.

    Args:
        board: Board identifier, for example ``12000001``.
    """
    with session_factory()() as session:
        record = session.execute(select(Board).where(Board.stem == board)).scalar()
        if record is None:
            return {"error": f"no board {board!r} in the store"}

        lot_boards = session.execute(
            select(func.count()).select_from(Board).where(Board.lot_id == record.lot_id)
        ).scalar()
        lot_defects = session.execute(
            select(func.count())
            .select_from(CandidateRecord)
            .join(Board)
            .where(Board.lot_id == record.lot_id)
            .where(CandidateRecord.predicted_class != "false_call")
        ).scalar()

        return {
            "board": board,
            "lot_id": record.lot_id,
            "line_id": record.line_id,
            "machine_id": record.machine_id,
            "shift": record.shift,
            "inspected_at": record.inspected_at.isoformat(),
            "lot_boards": lot_boards,
            "lot_defects": lot_defects,
            "lot_defects_per_board": round(lot_defects / lot_boards, 2) if lot_boards else 0.0,
        }


if __name__ == "__main__":
    mcp.run()
