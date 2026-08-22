"""Reading and writing analysis runs.

JSON columns rather than tables of rows: nothing queries inside a plan or a
result set, they are read back whole to render or to score. Normalising them
would buy nothing and cost a migration every time a tool's return shape moves.
"""

from __future__ import annotations

import json

from sqlalchemy import select

from aoi_agent.store.boards import session_factory
from aoi_agent.store.models import AnalysisRun


def _as_dict(row: AnalysisRun) -> dict:
    return {
        "id": row.id,
        "question": row.question,
        "plan": json.loads(row.plan_json) if row.plan_json else None,
        "results": json.loads(row.results_json),
        "chart": json.loads(row.chart_json) if row.chart_json else None,
        "answer": row.answer,
        "timings": json.loads(row.timings_json),
        "refused": row.refused,
        "asked_by": row.asked_by,
        "asked_at": row.asked_at.isoformat() if row.asked_at else None,
    }


def save_run(
    question: str,
    plan: dict | None,
    results: list[dict],
    chart: dict | None,
    answer: str,
    timings: dict,
    refused: bool,
    asked_by: str | None,
) -> int:
    with session_factory()() as session:
        row = AnalysisRun(
            question=question,
            plan_json=json.dumps(plan, ensure_ascii=False) if plan else None,
            results_json=json.dumps(results, ensure_ascii=False),
            chart_json=json.dumps(chart, ensure_ascii=False) if chart else None,
            answer=answer,
            timings_json=json.dumps(timings),
            refused=refused,
            asked_by=asked_by,
        )
        session.add(row)
        session.commit()
        return row.id


def get_run(run_id: int) -> dict | None:
    with session_factory()() as session:
        row = session.get(AnalysisRun, run_id)
        return _as_dict(row) if row else None


def recent_runs(limit: int = 20) -> list[dict]:
    with session_factory()() as session:
        rows = session.execute(
            select(AnalysisRun).order_by(AnalysisRun.id.desc()).limit(limit)
        ).scalars().all()
        return [_as_dict(row) for row in rows]
