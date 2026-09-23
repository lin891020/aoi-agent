"""The session probe compares plans, not prose, and its verdict was written first.

Four ways it could report something other than what it claims:

* it compares the whole response, so a reworded `interpretation` reads as a
  different plan -- which is exactly what the five-minute version did;
* the interposed workload is drawn from the fixture, so it warms the prefixes
  being measured;
* a partial run appends, and the entry's table has empty rows for conditions
  nobody measured;
* the verdict is read after the fact, on a threshold chosen to fit.

Nothing here calls a model: the plans are hand-written strings.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The shape `build_planning_messages` reads; the probe composes a real prompt
#: even in these tests, because a probe that cannot build one is broken.
DOMAINS = {"line_id": ["L1"], "machine_id": ["M11"], "defect_type": ["open"],
           "relative_to": [], "max_days": 9, "date_span": ["2026-08-01", "2026-08-09"]}


def _script(name: str):
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    pytest.importorskip("langgraph", reason="the probe imports the analysis graph's prompts")
    return _script("planner_session_probe")


# ---- what counts as a difference ------------------------------------------------


def test_only_the_calls_are_compared(probe):
    """Two plans that call the same tools with the same arguments are one plan."""
    baseline = {"S01": "query_defect_history({\"days\": 7})"}
    same = {"S01": "query_defect_history({\"days\": 7})"}
    other = {"S01": "query_defect_history({\"days\": 3})"}
    assert probe.differences(baseline, same) == []
    assert probe.differences(baseline, other) == ["S01"]


def test_a_question_missing_from_a_condition_counts_as_moved(probe):
    assert probe.differences({"S01": "a", "S02": "b"}, {"S01": "a"}) == ["S02"]


def test_a_failed_call_is_a_cell_and_not_a_crash(probe):
    class Refuses:
        def chat(self, *_a, **_k):
            raise TimeoutError("deadline")

    text = probe.plan_calls(Refuses(), "問題", DOMAINS)
    assert text.startswith("(no answer: TimeoutError")


def test_a_prompt_this_probe_cannot_build_raises_rather_than_reading_as_a_timeout(probe):
    class NeverCalled:
        def chat(self, *_a, **_k):                     # pragma: no cover - must not be reached
            raise AssertionError("the prompt should have raised first")

    with pytest.raises(KeyError):
        probe.plan_calls(NeverCalled(), "問題", {})


def test_an_unparseable_answer_is_its_own_cell(probe):
    class Garbles:
        def chat(self, *_a, **_k):
            return type("R", (), {"text": "not json"})()

    assert probe.plan_calls(Garbles(), "問題", DOMAINS) == "(unparsed)"


# ---- the interposition cannot drift ---------------------------------------------


def test_the_interposed_questions_are_fixed_and_not_the_fixtures(probe):
    fixture = {q["question"] for q in json.loads(probe.QUESTIONS.read_text())}
    assert len(probe.INTERPOSED) == 10 and probe.INTERPOSED_ROUNDS == 20
    assert not (set(probe.INTERPOSED) & fixture), (
        "interposing a measured question warms the prefix the measurement reads")


# ---- the verdict was written before the run -------------------------------------


@pytest.mark.parametrize(("moves", "label"), [
    ({"C2": 0, "C3": 5, "C4": 6}, "the_sequence"),
    ({"C2": 4, "C3": 7, "C4": 4}, "the_sequence"),    # the floor is C2, not zero
    ({"C2": 4, "C3": 5, "C4": 4}, "neither"),         # one past the floor is noise
    ({"C2": 0, "C3": 0, "C4": 4}, "the_reload"),
    ({"C2": 2, "C3": 3, "C4": 6}, "the_reload"),
    ({"C2": 0, "C3": 0, "C4": 0}, "neither"),
    ({"C2": 9, "C3": 9, "C4": 9}, "neither"),         # a noisy repeat is a wide floor
    ({"C2": 0, "C3": 4}, "incomplete"),
])
def test_the_verdict_is_the_pre_registered_ladder(probe, moves, label):
    assert probe.verdict(moves)[0] == label


def test_the_threshold_is_the_one_the_drift_set(probe):
    """Three questions past C2 is the bar, because three readings spanned 23-28 of 42."""
    assert probe.MOVED == 3
    benchmarks = (ROOT / "docs" / "benchmarks.md").read_text()
    assert "25 / 42 every time" in benchmarks


def test_only_the_full_four_condition_run_may_append(probe):
    assert probe.publishable(["C1", "C2", "C3", "C4"], 70)
    assert not probe.publishable(["C1", "C2"], 70)
    assert not probe.publishable(["C1", "C2", "C3", "C4"], 6)
