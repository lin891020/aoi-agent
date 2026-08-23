"""Reading and writing analysis runs.

JSON columns rather than tables of rows: nothing queries inside a plan or a
result set, they are read back whole to render or to score. Normalising them
would buy nothing and cost a migration every time a tool's return shape moves.
"""

from __future__ import annotations

import json

from sqlalchemy import select

from aoi_agent.i18n import DEFAULT_LOCALE
from aoi_agent.provenance import UNRECORDED
from aoi_agent.store.boards import session_factory
from aoi_agent.store.models import AnalysisRun


def _answers_of(row: AnalysisRun) -> dict[str, str]:
    """Every language this run has an answer in.

    A row written before `answers_json` existed has one answer and no record of
    its language, so it comes back under the same word the migration stamped:
    the reader can see there is a version and that nobody knows which language
    it is in. Inventing a key for it would claim the opposite.
    """
    if row.answers_json:
        return json.loads(row.answers_json)
    return {row.asked_lang or UNRECORDED: row.answer} if row.answer else {}


def _as_dict(row: AnalysisRun) -> dict:
    return {
        "id": row.id,
        "question": row.question,
        "plan": json.loads(row.plan_json) if row.plan_json else None,
        "results": json.loads(row.results_json),
        "chart": json.loads(row.chart_json) if row.chart_json else None,
        "answer": row.answer,
        "asked_lang": row.asked_lang,
        "answers": _answers_of(row),
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
    asked_lang: str | None = None,
) -> int:
    with session_factory()() as session:
        row = AnalysisRun(
            question=question,
            plan_json=json.dumps(plan, ensure_ascii=False) if plan else None,
            results_json=json.dumps(results, ensure_ascii=False),
            chart_json=json.dumps(chart, ensure_ascii=False) if chart else None,
            answer=answer,
            asked_lang=asked_lang or DEFAULT_LOCALE,
            # Written to both from the start. `answer` is what every reader
            # that predates this column expects; `answers_json` is what a
            # second language is added to.
            answers_json=json.dumps(
                {asked_lang or DEFAULT_LOCALE: answer}, ensure_ascii=False),
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


def add_answer(run_id: int, lang: str, answer: str) -> dict | None:
    """Keep one more language's answer for a run that already exists.

    Only ever adds. A language already present is left alone and the model is
    not called for it again -- the stored answer is the one whose figures were
    checked, and rewriting it would replace a measured artefact with an
    unmeasured one that reads the same.
    """
    with session_factory()() as session:
        row = session.get(AnalysisRun, run_id)
        if row is None:
            return None
        answers = json.loads(row.answers_json) if row.answers_json else {}
        answers.setdefault(lang, answer)
        row.answers_json = json.dumps(answers, ensure_ascii=False)
        session.commit()
        return _as_dict(row)
