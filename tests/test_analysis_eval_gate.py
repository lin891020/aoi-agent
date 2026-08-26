"""The planner eval refuses to publish a run the machine ruined.

On 2026-08-26 the independent set was run while a hung job from another
session held Ollama. Thirty-four of the first thirty-nine questions timed
out; the last thirty-one, after the job was killed, scored 25/31. The script
did the right thing with the timeouts -- scored them as misses, not
refusals, so the damage read as a collapse and not as calibration -- and then
appended the collapse to docs/benchmarks.md as if it were a measurement,
because nothing between the score and the file asked whether the run was
worth publishing.

Two gates now sit there, both pure so they can be held here without a model:
the share of questions with no plan at all, and what the machine looked like
when the run started. A contended machine is reported into the section
rather than left to a hand-written note.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analysis_eval  # noqa: E402


def test_a_run_with_a_tenth_of_its_questions_unplanned_is_not_published():
    assert analysis_eval.publishable(no_plan=0, total=70)
    assert analysis_eval.publishable(no_plan=7, total=70)
    assert not analysis_eval.publishable(no_plan=8, total=70)
    assert not analysis_eval.publishable(no_plan=24, total=70), "the 2026-08-26 run"


def test_an_empty_run_is_not_publishable_either():
    assert not analysis_eval.publishable(no_plan=0, total=0)


def test_the_limit_is_the_one_the_docstring_names():
    assert analysis_eval.PUBLISH_FAILURE_LIMIT == pytest.approx(0.10)


def test_the_models_own_residency_is_not_contention(monkeypatch):
    """The eval keeps its own model resident; that is the run, not a rival."""
    monkeypatch.setattr(analysis_eval, "_resident_models", lambda: ["gpt-oss:20b"])
    monkeypatch.setattr(analysis_eval, "_competing_processes", lambda: [])
    assert analysis_eval.machine_state("gpt-oss:20b") == []


def test_another_resident_model_or_a_torch_job_is_reported(monkeypatch):
    monkeypatch.setattr(analysis_eval, "_resident_models", lambda: ["gpt-oss:20b", "llama3:8b"])
    monkeypatch.setattr(analysis_eval, "_competing_processes", lambda: ["python train.py (pid 4242)"])
    state = analysis_eval.machine_state("gpt-oss:20b")
    assert "llama3:8b" in state
    assert any("train.py" in s for s in state)


def test_the_machine_line_is_derived_into_the_section():
    quiet = analysis_eval.machine_line([])
    busy = analysis_eval.machine_line(["llama3:8b"])
    assert quiet.startswith("Machine at start:") and "quiet" in quiet
    assert busy.startswith("Machine at start:") and "llama3:8b" in busy and "busy" in busy
