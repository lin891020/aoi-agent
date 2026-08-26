"""Machine events: what happened to a station, and when.

Read the module docstring on ``models.MachineEvent`` for why this table
exists. This module is the only writer, because the one thing the table can
get quietly wrong is naming a machine that does not exist -- ``line_id="L4"``
one level up -- and a guard that lives in every caller lives in none of them.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select

from aoi_agent.store.boards import session_factory
from aoi_agent.store.models import Board, MachineEvent

#: The kinds the seeder plants. The *validator* does not read this tuple: it
#: reads the kinds the table actually holds, the same way it reads machines
#: and lines off ``boards``, so a kind recorded by hand one day is plannable
#: without a code change and a kind listed here but never recorded is not.
KINDS = ("parameter_change", "maintenance", "lamp_replaced", "nozzle_cleaned")

SEEDED = "seeded"


def known_machines(session) -> set[str]:
    return set(session.execute(select(Board.machine_id).distinct()).scalars())


def record(
    machine_id: str,
    kind: str,
    happened_at: datetime,
    note: str | None = None,
    recorded_by: str = SEEDED,
    session=None,
) -> MachineEvent:
    """Write one event, refusing a machine the store has never inspected on.

    Raises rather than returning an error dict: this is the store's own write
    path, not a tool the planner calls, and a silent row about ``M99`` is the
    failure the whole no-free-text-SQL invariant exists to prevent.
    """
    if not kind or not kind.strip():
        raise ValueError("an event needs a kind")

    def _write(s):
        machines = known_machines(s)
        if machine_id not in machines:
            raise ValueError(
                f"no board in this store was inspected on {machine_id!r}; "
                f"known machines: {sorted(machines)}"
            )
        row = MachineEvent(
            machine_id=machine_id, kind=kind, happened_at=happened_at,
            note=note, recorded_by=recorded_by,
        )
        s.add(row)
        s.flush()
        return row

    if session is not None:
        return _write(session)
    with session_factory()() as s:
        row = _write(s)
        s.commit()
        s.refresh(row)
        s.expunge(row)
        return row


def _as_dict(row: MachineEvent) -> dict:
    return {
        "machine_id": row.machine_id,
        "kind": row.kind,
        "happened_at": row.happened_at.isoformat(),
        "note": row.note,
        "recorded_by": row.recorded_by,
    }


def events_for(machine_id: str | None = None, kind: str | None = None) -> list[dict]:
    """Every recorded event, newest first, optionally narrowed."""
    with session_factory()() as session:
        query = select(MachineEvent)
        if machine_id:
            query = query.where(MachineEvent.machine_id == machine_id)
        if kind:
            query = query.where(MachineEvent.kind == kind)
        rows = session.execute(
            query.order_by(MachineEvent.happened_at.desc(), MachineEvent.id.desc())
        ).scalars().all()
        return [_as_dict(r) for r in rows]


def anchor_for(machine_id: str, kind: str) -> datetime | None:
    """The newest event of ``kind`` on ``machine_id``, or None.

    One rule, here, for "which event does *after* mean": the newest. A tool
    that resolved it differently from another tool would give two answers to
    "did the change help", and the two would disagree exactly on the machines
    with more than one event -- the ones somebody is asking about.
    """
    with session_factory()() as session:
        return session.execute(
            select(func.max(MachineEvent.happened_at)).where(
                MachineEvent.machine_id == machine_id, MachineEvent.kind == kind
            )
        ).scalar()


def kinds_present() -> set[str]:
    """The kinds the table actually holds -- the validator's domain."""
    with session_factory()() as session:
        return set(session.execute(select(MachineEvent.kind).distinct()).scalars())


def count() -> int:
    with session_factory()() as session:
        return int(session.execute(select(func.count(MachineEvent.id))).scalar())
