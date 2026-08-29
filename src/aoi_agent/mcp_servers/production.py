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

from datetime import date, datetime, timedelta

from mcp.server.mcpserver import MCPServer
from sqlalchemy import func, select

from aoi_agent.store import events as machine_events
from aoi_agent.store.boards import session_factory
from aoi_agent.store.models import Board, CandidateRecord

mcp = MCPServer("aoi-production")

DEFECT_CLASSES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]

#: What a date argument has to look like. Stated in the error rather than
#: guessed at: ``30/07/2026`` and ``2026/7/30`` are both dates a person would
#: type, and reading either as "no date" would return the whole span under a
#: window the caller believes is one day.
DATE_FORMAT = "YYYY-MM-DD"


def _parse_date(name: str, value: str | None) -> tuple[date | None, str | None]:
    """An ISO date, or the reason it is not one."""
    if value is None:
        return None, None
    try:
        return date.fromisoformat(str(value)), None
    except ValueError:
        return None, f"{name}={value!r} is not a date; write it as {DATE_FORMAT}"


def _dated_window(
    date_from: str | None, date_to: str | None, earliest: datetime, latest: datetime
) -> tuple[datetime, datetime] | dict:
    """The window two calendar dates describe, inclusive at both ends.

    A day is a day: ``date_to`` runs to the last microsecond of that date, so
    ``date_from == date_to`` is one whole day and not the instant it began.
    An end left open runs to the edge of the data on that side.
    """
    start, error = _parse_date("date_from", date_from)
    if error:
        return {"error": error}
    end, error = _parse_date("date_to", date_to)
    if error:
        return {"error": error}
    since = datetime.combine(start, datetime.min.time()) if start else earliest
    until = datetime.combine(end, datetime.max.time()) if end else latest
    if since > until:
        return {"error": f"date_from={date_from!r} is after date_to={date_to!r}; the window is empty"}
    return since, until


#: The two sides of a machine event. A window is *before* an event or *after*
#: it; there is no "around", because a window that straddles the anchor is
#: the question with the answer averaged out of it.
SIDES = ("before", "after")


@mcp.tool()
def query_defect_history(
    lot_id: str | None = None,
    line_id: str | None = None,
    machine_id: str | None = None,
    defect_type: str | None = None,
    days: int = 7,
    relative_to: str | None = None,
    side: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict:
    """Count defects by class over a slice of recent production.

    Use this to judge whether a defect on one board is isolated or part of a
    pattern. Leave a filter unset to include everything. To compare a machine
    before and after something happened to it, call this twice with the same
    ``machine_id`` and ``relative_to`` -- once with ``side="before"`` and once
    with ``side="after"`` -- and read the two ``open_share`` intervals.

    Args:
        lot_id: Restrict to one lot, for example ``LOT-2608003``.
        line_id: Restrict to one line: ``L1``, ``L2`` or ``L3``.
        machine_id: Restrict to one machine, for example ``M22``.
        defect_type: Restrict to one class: open, short, mousebite, spur,
            copper or pin-hole.
        days: How far back to look from the most recent inspection. Ignored
            when ``relative_to`` is set: the window is then bounded by the
            event instead.
        relative_to: The kind of machine event to anchor on, for example
            ``parameter_change``; needs ``machine_id`` and ``side``. The
            anchor is that machine's newest event of the kind.
        side: ``before`` (strictly) or ``after`` (from the event's instant on)
            the anchor. The two windows partition the machine's boards.
        date_from: First calendar day of the window, ``YYYY-MM-DD``, inclusive.
            Use this with ``date_to`` for a specific day or span of days --
            ``date_from="2026-08-05", date_to="2026-08-05"`` is that one day.
            Replaces ``days``; cannot be combined with ``relative_to``.
        date_to: Last calendar day of the window, inclusive. Either end may
            be left unset to run to the edge of the data on that side.
    """
    if defect_type and defect_type not in DEFECT_CLASSES:
        return {"error": f"unknown defect_type {defect_type!r}; expected one of {DEFECT_CLASSES}"}

    dated = date_from is not None or date_to is not None
    if dated and (relative_to is not None or side is not None):
        # Two ways of bounding one window. Composing them -- the part of the
        # dated span before the event -- is a question nobody has asked, and
        # answering it by accident is worse than refusing.
        return {"error": "date_from/date_to and relative_to/side are two different "
                         "windows; use one or the other"}

    # An event belongs to one machine; there is no fleet-wide "after". And a
    # side without an anchor, or an anchor without a side, is half a question.
    if (relative_to is None) != (side is None):
        return {"error": "relative_to and side go together: name the event kind and which side of it"}
    if relative_to is not None and not machine_id:
        return {"error": "relative_to needs machine_id: an event happened to one machine"}
    if side is not None and side not in SIDES:
        return {"error": f"unknown side {side!r}; expected one of {SIDES}"}

    anchor = None
    if relative_to is not None:
        anchor = machine_events.anchor_for(machine_id, relative_to)
        if anchor is None:
            return {
                "error": f"no {relative_to!r} event is recorded for {machine_id}; "
                         f"query_machine_events(machine_id={machine_id!r}) lists what is",
            }

    with session_factory()() as session:
        latest = session.execute(select(func.max(Board.inspected_at))).scalar()
        if latest is None:
            return {"error": "the store is empty; run scripts/seed_store.py"}
        earliest = session.execute(select(func.min(Board.inspected_at))).scalar()

        # "Before" is strictly before: a board inspected at the event's own
        # instant is the first board *after* it, so the two windows partition
        # the machine's boards and no board is counted on both sides.
        if dated:
            window = _dated_window(date_from, date_to, earliest, latest)
            if isinstance(window, dict):
                return window
            since, until = window
            in_window = (Board.inspected_at >= since, Board.inspected_at <= until)
        elif anchor is None:
            since, until = latest - timedelta(days=days), latest
            in_window = (Board.inspected_at >= since, Board.inspected_at <= until)
        elif side == "before":
            since, until = earliest, anchor
            in_window = (Board.inspected_at >= since, Board.inspected_at < until)
        else:
            since, until = anchor, latest
            in_window = (Board.inspected_at >= since, Board.inspected_at <= until)

        query = (
            select(CandidateRecord.predicted_class, func.count())
            .join(Board)
            .where(*in_window)
            .where(CandidateRecord.predicted_class != "false_call")
            .group_by(CandidateRecord.predicted_class)
        )
        boards_query = select(func.count(func.distinct(Board.id))).where(*in_window)

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

        # The share of the window's *defects* the model called `open`, with an
        # interval. The denominator is defects, not every flagged region: the
        # first draft divided by all candidates and the planted effect on the
        # real store came back with overlapping intervals -- not because it was
        # absent but because sixteen candidates a board, nine of them false
        # calls, diluted it by the false-call rate, which is a property of the
        # AOI and not of the parameter that was changed. "Did the change reduce
        # opens" is a question about the defects that were found. The flagged
        # count is still returned, for context. Only computed for a windowed
        # query: an unanchored slice already answers a different question.
        flagged = 0
        if anchor is not None:
            flagged_query = select(func.count(CandidateRecord.id)).join(Board).where(*in_window)
            for column, value in (
                (Board.lot_id, lot_id), (Board.line_id, line_id), (Board.machine_id, machine_id),
            ):
                if value:
                    flagged_query = flagged_query.where(column == value)
            flagged = int(session.execute(flagged_query).scalar() or 0)

    total = sum(counts.values())
    out = {
        "filters": {
            "lot_id": lot_id, "line_id": line_id, "machine_id": machine_id,
            "defect_type": defect_type,
            # `days` is reported as None when it did not bound the window,
            # so a reader of the record cannot take the default for a choice.
            "days": None if (dated or anchor is not None) else days,
            "relative_to": relative_to, "side": side,
            "date_from": date_from, "date_to": date_to,
        },
        "window_start": since.isoformat(),
        "window_end": until.isoformat(),
        "boards_inspected": boards,
        "defects_total": total,
        "defects_per_board": round(total / boards, 2) if boards else 0.0,
        "by_class": counts,
    }
    if anchor is not None:
        from aoi_agent.stats import wilson

        opens = int(counts.get("open", 0))
        low, high = wilson(opens, total)
        out["event_at"] = anchor.isoformat()
        out["flagged_regions"] = flagged
        if defect_type:
            # A window filtered to one class has a share of opens that is 1 or
            # 0 by construction, and a planner that splits the question into
            # "opens before" and "all defects before" would otherwise get a
            # chart of two full bars. The number is not reported; the reason
            # is, so the prose can say why and the chart draws only the
            # unfiltered windows.
            out["open_share"] = {
                "value": None, "interval_95": None,
                "basis": (
                    f"not reported: this window is filtered to {defect_type!r}, "
                    "so the share of opens in it is 1 or 0 by construction; "
                    "the share is on the unfiltered window for the same side"
                ),
            }
        else:
            out["open_share"] = {
                "value": round(opens / total, 4) if total else None,
                "interval_95": [round(low, 4), round(high, 4)],
                "basis": (
                    "share of the defects the re-verifier confirmed in this window "
                    "that it classified as open (false calls are not in the "
                    "denominator); the interval is a Wilson interval on that "
                    "count, and two windows whose intervals overlap have not been "
                    "shown to differ"
                ),
            }
    return out


@mcp.tool()
def query_machine_events(machine_id: str | None = None, kind: str | None = None) -> dict:
    """List what has happened to a machine: parameter changes, maintenance.

    Use this before comparing a machine's output before and after something
    was done to it, to find out what was done and when. Newest first.

    Args:
        machine_id: Restrict to one machine, for example ``M32``.
        kind: Restrict to one kind of event, for example ``parameter_change``.
    """
    rows = machine_events.events_for(machine_id, kind)
    return {
        "filters": {"machine_id": machine_id, "kind": kind},
        "events": rows,
        "count": len(rows),
        "basis": (
            "events are recorded by a person or planted by the seed and say "
            "what was done and when; whether the output changed afterwards is "
            "a separate lookup, query_defect_history with relative_to and side"
        ),
    }


@mcp.tool()
def query_machine_stats(
    defect_type: str | None = None,
    days: int = 14,
    date_from: str | None = None,
    date_to: str | None = None,
    top_n: int | None = None,
) -> dict:
    """Rank every machine by one defect class, or by all defects per board.

    Use this when a defect might be traceable to a station, or when the
    question is "which machines had the most defects". With ``defect_type``
    set, each machine's rate for that class is returned, ranked worst first
    by ``share_of_defects``. With it unset, every class counts and machines
    are ranked by ``per_board`` -- the field ``ranked_by`` says which.

    Two rates are reported and they answer different questions.
    ``per_board`` is the raw count and moves with anything that raises total
    defects, so a machine running harder material looks worse on every class at
    once. ``share_of_defects`` is the class as a fraction of that machine's own
    defects, which stays flat unless the machine has a problem specific to this
    class. Prefer ``share_of_defects`` when deciding whether a station is the
    cause; use ``per_board`` to judge how much it costs. When no class is
    given, ``share_of_defects`` is not reported: all defects as a share of all
    defects is 1 on every machine.

    Args:
        defect_type: One of open, short, mousebite, spur, copper, pin-hole;
            or unset to count every class together.
        days: How far back to look from the most recent inspection. Ignored
            when ``date_from`` or ``date_to`` is set.
        date_from: First calendar day of the window, ``YYYY-MM-DD``, inclusive.
            ``date_from="2026-08-05", date_to="2026-08-05"`` is that one day.
        date_to: Last calendar day of the window, inclusive.
        top_n: Return only the first N machines of the ranking; the count
            before the cut is reported as ``machines_total``.
    """
    if defect_type is not None and defect_type not in DEFECT_CLASSES:
        return {"error": f"unknown defect_type {defect_type!r}; expected one of {DEFECT_CLASSES}"}
    if top_n is not None and (not isinstance(top_n, int) or top_n < 1):
        return {"error": f"top_n={top_n!r} must be a whole number of at least 1"}

    dated = date_from is not None or date_to is not None
    with session_factory()() as session:
        earliest, latest = session.execute(
            select(func.min(Board.inspected_at), func.max(Board.inspected_at))
        ).first()
        if latest is None:
            return {"error": "the store is empty; run scripts/seed_store.py"}
        if dated:
            window = _dated_window(date_from, date_to, earliest, latest)
            if isinstance(window, dict):
                return window
            since, until = window
        else:
            since, until = latest - timedelta(days=days), latest
        in_window = (Board.inspected_at >= since, Board.inspected_at <= until)
        key = Board.line_id + "-" + Board.machine_id

        boards = dict(
            session.execute(
                select(key, func.count(func.distinct(Board.id)))
                .where(*in_window)
                .group_by(Board.line_id, Board.machine_id)
            ).all()
        )
        all_defects = dict(
            session.execute(
                select(key, func.count())
                .join(CandidateRecord)
                .where(*in_window)
                .where(CandidateRecord.predicted_class != "false_call")
                .group_by(Board.line_id, Board.machine_id)
            ).all()
        )
        if defect_type is None:
            defects = all_defects
        else:
            defects = dict(
                session.execute(
                    select(key, func.count())
                    .join(CandidateRecord)
                    .where(*in_window)
                    .where(CandidateRecord.predicted_class == defect_type)
                    .group_by(Board.line_id, Board.machine_id)
                ).all()
            )

    ranked_by = "per_board" if defect_type is None else "share_of_defects"
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
                "share_of_defects": (
                    None if defect_type is None
                    else round(this_class / every_class, 3) if every_class else 0.0
                ),
            }
        )
    rows.sort(key=lambda r: (-(r[ranked_by] or 0.0), r["machine"]))
    machines_total = len(rows)
    if top_n is not None:
        rows = rows[:top_n]

    total_this_class = sum(defects.values())
    total_all = sum(all_defects.values())
    return {
        "filters": {
            "defect_type": defect_type, "days": None if dated else days,
            "date_from": date_from, "date_to": date_to, "top_n": top_n,
        },
        "defect_type": defect_type,
        "days": None if dated else days,
        "window_start": since.isoformat(),
        "window_end": until.isoformat(),
        "ranked_by": ranked_by,
        "fleet_average_per_board": round(
            sum(all_defects.get(m, 0) if defect_type is None else defects.get(m, 0)
                for m in boards) / sum(boards.values()), 3
        ) if boards else 0.0,
        "fleet_share_of_defects": (
            None if defect_type is None
            else round(total_this_class / total_all, 3) if total_all else 0.0
        ),
        "machines_total": machines_total,
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


#: What `query_false_call_rate` may group over. The tool owns this vocabulary
#: -- it is a fact about what the tool can do, not about what the store holds --
#: and the plan validator reads it from here so the two cannot drift.
GROUP_BY = ("machine", "line", "shift")


@mcp.tool()
def query_false_call_rate(group_by: str = "machine", days: int = 7) -> dict:
    """How much of what the AOI flags this system dismisses, grouped.

    Use this when the question is about false calls in aggregate -- which
    machine over-flags, whether a line's flags are worth an operator's time --
    rather than about real defects. `query_defect_history` excludes false calls
    by design; this is the tool that counts them.

    **What the rate is, and is not.** Every figure here is the re-verifier's
    own judgement: `dismissed` counts regions this system classified as
    `false_call`, not regions known to be false calls. There is no ground truth
    in this store's production records and none is reported. Two consequences,
    both worth stating to whoever reads the number: a machine with a high rate
    is a machine whose flags *this model* dismisses, which usually means the
    machine over-flags but can also mean the model is confidently wrong about
    that machine's images; and an escape -- a real defect wrongly dismissed --
    is counted on the dismissed side, so this rate cannot see the one error
    that matters most. The escape rate lives in docs/benchmarks.md, measured
    against annotations, and this tool is not a substitute for it.

    Args:
        group_by: One of ``machine``, ``line`` or ``shift``.
        days: How far back to look from the most recent inspection. If the
            store holds less than asked, the window actually covered is
            reported beside the request rather than silently substituted.
    """
    if group_by not in GROUP_BY:
        return {"error": f"unknown group_by {group_by!r}; expected one of {list(GROUP_BY)}"}

    key = {
        "machine": Board.line_id + "-" + Board.machine_id,
        "line": Board.line_id,
        "shift": Board.shift,
    }[group_by]

    with session_factory()() as session:
        earliest, latest = session.execute(
            select(func.min(Board.inspected_at), func.max(Board.inspected_at))
        ).first()
        if latest is None:
            return {"error": "the store is empty; run scripts/seed_store.py"}
        since = latest - timedelta(days=days)

        flagged = dict(
            session.execute(
                select(key, func.count())
                .join(CandidateRecord)
                .where(Board.inspected_at >= since)
                .group_by(key)
            ).all()
        )
        dismissed = dict(
            session.execute(
                select(key, func.count())
                .join(CandidateRecord)
                .where(Board.inspected_at >= since)
                .where(CandidateRecord.predicted_class == "false_call")
                .group_by(key)
            ).all()
        )

    groups = [
        {
            "group": group,
            "flagged": count,
            "dismissed_as_false_call": dismissed.get(group, 0),
            "false_call_rate": round(dismissed.get(group, 0) / count, 4),
        }
        for group, count in flagged.items()
    ]
    groups.sort(key=lambda row: (-row["false_call_rate"], row["group"]))

    total_flagged = sum(flagged.values())
    total_dismissed = sum(dismissed.values())
    # The days=14-over-a-9-day-store lesson, applied at birth instead of found
    # later: the caller asked for a window, the store holds what it holds, and
    # the payload states both instead of labelling one with the other's number.
    covered = min(days, max(0, (latest - earliest).days) if earliest else 0)
    return {
        "filters": {"group_by": group_by, "days": days},
        "window_end": latest.isoformat(),
        "window_days_requested": days,
        "window_days_covered": covered,
        "basis": (
            "rates are the re-verifier's own classifications, not ground truth; "
            "a wrongly dismissed real defect (an escape) is invisible here"
        ),
        "flagged_total": total_flagged,
        "dismissed_total": total_dismissed,
        "false_call_rate": round(total_dismissed / total_flagged, 4) if total_flagged else 0.0,
        "by_group": groups,
    }


if __name__ == "__main__":
    mcp.run()
