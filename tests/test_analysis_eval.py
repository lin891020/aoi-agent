"""Scoring a plan against what it should have done.

No model is called here. The eval's own tests have to be deterministic and fast,
or a contended Ollama turns a green suite red for reasons that have nothing to do
with the code.
"""

from __future__ import annotations

import re
from pathlib import Path

from aoi_agent.analysis.plan import PLANNABLE_TOOLS
from aoi_agent.analysis.prompts import FEW_SHOT
from aoi_agent.store import seed

from analysis_eval import load_questions, score_plan

FIXTURE = Path(__file__).parent / "fixtures" / "analysis_questions.json"

#: `L2`, `M22`, `L4` -- not the digits of a board stem.
NAMED_ENTITY = re.compile(r"(?<![A-Za-z0-9])([LM]\d{1,2})(?![0-9])")


def plan(*tools, assumptions=None):
    return {
        "interpretation": "i",
        "assumptions": assumptions if assumptions is not None else [],
        "calls": [{"tool": t, "args": {}, "why": "w"} for t in tools],
    }


def known_values() -> set[str]:
    return set(seed.LINES) | {m for ms in seed.LINES.values() for m in ms}


def test_the_question_set_loads_and_covers_both_outcomes():
    questions = load_questions(FIXTURE)

    assert len(questions) >= 20
    assert any(q["expect_refusal"] for q in questions)
    assert any(not q["expect_refusal"] for q in questions)


def test_calling_the_expected_tools_scores_a_hit():
    result = score_plan(plan("query_machine_stats"),
                        {"expect_refusal": False,
                         "expect_tools": ["query_machine_stats"]})
    assert result["ok"] is True


def test_a_missing_expected_tool_is_a_miss():
    result = score_plan(plan("query_defect_history"),
                        {"expect_refusal": False,
                         "expect_tools": ["query_machine_stats"]})
    assert result["ok"] is False
    assert "query_machine_stats" in result["reason"]


def test_extra_tools_are_allowed():
    """Gathering more context than the minimum is not an error. Gathering less
    is, because the answer then rests on data that was never fetched."""
    result = score_plan(plan("query_machine_stats", "search_standards"),
                        {"expect_refusal": False,
                         "expect_tools": ["query_machine_stats"]})
    assert result["ok"] is True


def test_refusing_when_a_refusal_was_expected_scores_a_hit():
    result = score_plan(plan(), {"expect_refusal": True})
    assert result["ok"] is True


def test_answering_a_question_that_should_be_refused_is_a_miss():
    """The dangerous direction. A system that answers everything is worse on a
    factory floor than one that says it cannot."""
    result = score_plan(plan("query_defect_history"), {"expect_refusal": True})
    assert result["ok"] is False
    assert "refus" in result["reason"].lower()


def test_a_causal_question_must_disclaim_cause_in_its_assumptions():
    without = score_plan(plan("query_defect_history"),
                         {"expect_refusal": False,
                          "expect_assumption_about_cause": True})
    with_note = score_plan(
        plan("query_defect_history",
             assumptions=["This shows association, not cause."]),
        {"expect_refusal": False, "expect_assumption_about_cause": True},
    )

    assert without["ok"] is False
    assert with_note["ok"] is True


def test_a_comparison_with_no_stated_baseline_is_a_miss():
    result = score_plan(plan("query_defect_history"),
                        {"expect_refusal": False, "expect_assumptions": True})
    assert result["ok"] is False


def test_every_expected_tool_is_one_the_planner_can_actually_call():
    """A fixture naming a tool that does not exist scores every plan a miss and
    reads as a planner failure. This plan has already shipped one fabricated
    payload; the fixture is checked against the registry, not against prose."""
    for question in load_questions(FIXTURE):
        for tool in question.get("expect_tools") or []:
            assert tool in PLANNABLE_TOOLS, f"{question['question']}: {tool}"


def test_an_answerable_question_only_names_values_the_seed_creates():
    for question in load_questions(FIXTURE):
        if question["expect_refusal"]:
            continue
        named = set(NAMED_ENTITY.findall(question["question"]))
        assert not named - known_values(), question["question"]


def test_naming_something_the_seed_does_not_create_expects_a_refusal():
    """`L4` is the point of that question. If the seed ever grows a fourth line
    the expectation is wrong, and this says so instead of the run scoring a
    correct plan as a miss."""
    unknown_named = [
        q for q in load_questions(FIXTURE)
        if set(NAMED_ENTITY.findall(q["question"])) - known_values()
    ]
    assert unknown_named, "no question probes a value that does not exist"
    for question in unknown_named:
        assert question["expect_refusal"], question["question"]


def test_questions_already_shown_to_the_model_are_flagged_as_such():
    """Three of these questions are few-shot examples verbatim. Scoring them
    measures recall of the prompt, not planning, so the fixture marks them and
    the report scores the held-out subset separately."""
    shown = {example["question"] for example in FEW_SHOT}
    for question in load_questions(FIXTURE):
        if question["question"] in shown:
            assert question.get("in_prompt") == "verbatim", question["question"]
        if question.get("in_prompt") == "verbatim":
            assert question["question"] in shown, question["question"]


def test_the_held_out_questions_still_cover_both_outcomes():
    held_out = [q for q in load_questions(FIXTURE) if not q.get("in_prompt")]

    assert len(held_out) >= 12
    assert any(q["expect_refusal"] for q in held_out)
    assert any(not q["expect_refusal"] for q in held_out)
