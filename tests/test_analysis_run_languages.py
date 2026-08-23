"""One run, and the languages its answer exists in.

The spine of the bilingual design lives here: what the planning call wrote is a
record and keeps its language; the synthesised answer is the one thing that can
be produced again, from the stored results, in a language the run does not hold
yet. This module is about the store's half of that -- the writing, the reading,
and the two absences the migration has to keep apart.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import text

from aoi_agent.provenance import UNRECORDED
from aoi_agent.store import analysis as store
from aoi_agent.store.models import (
    ADDED_COLUMNS,
    BACKFILL_ON_ADD,
    create_all,
    make_engine,
    make_session_factory,
)


@pytest.fixture
def run_store(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'a.db'}"
    create_all(url)
    monkeypatch.setattr("aoi_agent.store.boards._session_factory",
                        make_session_factory(url))
    return url


def a_run(**kwargs) -> int:
    return store.save_run(**{
        "question": "比較三條線", "plan": {"interpretation": "i", "calls": []},
        "results": [], "chart": None, "answer": "答案", "timings": {},
        "refused": False, "asked_by": "mike", **kwargs,
    })


def test_a_run_records_the_language_it_was_asked_in(run_store):
    run = store.get_run(a_run(asked_lang="zh-TW"))

    assert run["asked_lang"] == "zh-TW"
    assert run["answers"] == {"zh-TW": "答案"}


def test_the_original_answer_column_still_holds_what_it_always_held(run_store):
    """Every reader written before `answers_json` keeps working."""
    run = store.get_run(a_run(asked_lang="en", answer="the answer"))

    assert run["answer"] == "the answer"


def test_a_second_language_is_added_without_disturbing_the_first(run_store):
    run_id = a_run(asked_lang="zh-TW")
    store.add_answer(run_id, "en", "the answer")
    run = store.get_run(run_id)

    assert run["answers"] == {"zh-TW": "答案", "en": "the answer"}
    assert run["answer"] == "答案", "the original column is not rewritten"


def test_a_language_already_present_is_never_overwritten(run_store):
    """The stored answer is the one whose figures were checked. Writing a
    fresh one over it swaps a measured artefact for an unmeasured one that
    reads exactly the same."""
    run_id = a_run(asked_lang="zh-TW")
    store.add_answer(run_id, "zh-TW", "一個不同的答案")

    assert store.get_run(run_id)["answers"]["zh-TW"] == "答案"


# ---------------------------------------------------------------------------
# The migration, and the two absences
# ---------------------------------------------------------------------------

def test_a_row_written_before_the_columns_says_so_rather_than_being_null(tmp_path):
    """`NULL` would have to mean both "predates the column" and "the language
    was not captured", and only the first is true of these rows."""
    url = f"sqlite:///{tmp_path / 'old.db'}"
    create_all(url)
    engine = make_engine(url)

    # A store as it stood before the columns existed.
    with engine.begin() as connection:
        for column in ADDED_COLUMNS["analysis_runs"]:
            connection.execute(text(f"ALTER TABLE analysis_runs DROP COLUMN {column}"))
        connection.execute(text(
            "INSERT INTO analysis_runs (question, results_json, answer, "
            "timings_json, refused) VALUES ('q', '[]', 'old answer', '{}', 0)"))

    create_all(url)

    with engine.begin() as connection:
        rows = connection.execute(
            text("SELECT asked_lang, answers_json FROM analysis_runs")).all()

    assert rows == [(UNRECORDED, None)]


def test_the_old_answer_is_not_filed_under_a_guessed_language(tmp_path, monkeypatch):
    """`answers_json` is deliberately not backfilled. Copying `answer` into it
    under a key would assert the language the stamp beside it says was never
    recorded -- and would make a re-synthesis in the other language look like
    it had already been done."""
    assert "answers_json" not in BACKFILL_ON_ADD["analysis_runs"]

    url = f"sqlite:///{tmp_path / 'old.db'}"
    create_all(url)
    engine = make_engine(url)
    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO analysis_runs (question, results_json, answer, "
            "timings_json, refused, asked_lang) "
            "VALUES ('q', '[]', 'old answer', '{}', 0, 'unrecorded')"))
    monkeypatch.setattr("aoi_agent.store.boards._session_factory",
                        make_session_factory(url))

    run = store.recent_runs(1)[0]

    assert run["answers"] == {UNRECORDED: "old answer"}, (
        "the answer is readable, and its language is readable as unknown"
    )


def test_nothing_this_module_writes_leaves_a_null_language(run_store):
    a_run()
    engine = make_engine(run_store)
    with engine.begin() as connection:
        nulls = connection.execute(
            text("SELECT count(*) FROM analysis_runs WHERE asked_lang IS NULL")
        ).scalar()

    assert nulls == 0
