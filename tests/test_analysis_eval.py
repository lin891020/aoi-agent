"""Scoring a plan against what it should have done.

No model is called here. The eval's own tests have to be deterministic and fast,
or a contended Ollama turns a green suite red for reasons that have nothing to do
with the code.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from aoi_agent.analysis.plan import PLANNABLE_TOOLS
from aoi_agent.analysis.prompts import FEW_SHOT
from aoi_agent.store import seed

from analysis_eval import (
    CLEAN_SWEEP,
    HELD_OUT_CAVEAT,
    SCORED_ARGS,
    load_questions,
    render_plan,
    score_plan,
)

FIXTURE = Path(__file__).parent / "fixtures" / "analysis_questions.json"

#: `L2`, `M22`, `L4` -- not the digits of a board stem.
NAMED_ENTITY = re.compile(r"(?<![A-Za-z0-9])([LM]\d{1,2})(?![0-9])")


def plan(*tools, assumptions=None):
    """Tools as bare names, or as ``(name, args)`` when the arguments matter."""
    calls = []
    for tool in tools:
        name, args = tool if isinstance(tool, tuple) else (tool, {})
        calls.append({"tool": name, "args": args, "why": "w"})
    return {
        "interpretation": "i",
        "assumptions": assumptions if assumptions is not None else [],
        "calls": calls,
    }


def expectations(question: dict) -> list[dict]:
    """The question, or each accepted plan merged onto it."""
    alternatives = question.get("expect_any_of")
    if not alternatives:
        return [question]
    return [{**question, **alternative} for alternative in alternatives]


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
        for expected in expectations(question):
            for tool in expected.get("expect_tools") or []:
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


def test_a_valid_but_wrong_argument_value_is_a_miss():
    """The failure the no-SQL invariant exists to prevent, one level up. A plan
    that asks about `short` when the question asked about `open` runs cleanly,
    returns real numbers, and answers a question nobody put."""
    result = score_plan(
        plan(("query_machine_stats", {"defect_type": "short", "days": 7})),
        {"expect_refusal": False, "expect_tools": ["query_machine_stats"],
         "expect_args": {"defect_type": ["open"]}},
    )
    assert result["ok"] is False
    assert "defect_type" in result["reason"]


def test_the_right_argument_value_scores_a_hit():
    result = score_plan(
        plan(("query_machine_stats", {"defect_type": "open", "days": 7})),
        {"expect_refusal": False, "expect_tools": ["query_machine_stats"],
         "expect_args": {"defect_type": ["open"]}},
    )
    assert result["ok"] is True


def test_a_required_argument_may_be_carried_by_any_call():
    """Which call carries it is the planner's business; that the plan asks about
    the thing the question named is not."""
    result = score_plan(
        plan(("search_standards", {"query": "open circuit"}),
             ("query_machine_stats", {"defect_type": "open"})),
        {"expect_refusal": False, "expect_args": {"defect_type": ["open"]}},
    )
    assert result["ok"] is True


def test_every_required_value_of_an_argument_must_appear():
    """Comparing three lines means querying three lines. Two is a different
    answer, not a partial one."""
    result = score_plan(
        plan(("query_defect_history", {"line_id": "L1"}),
             ("query_defect_history", {"line_id": "L2"})),
        {"expect_refusal": False,
         "expect_args": {"line_id": ["L1", "L2", "L3"]}},
    )
    assert result["ok"] is False
    assert "L3" in result["reason"]


def test_arguments_the_scorer_does_not_read_are_not_scored():
    """`days` and `top_k` are deliberately unscored: no question pins a window,
    and `validate_plan` already bounds `days` against the store."""
    result = score_plan(
        plan(("query_machine_stats", {"defect_type": "open", "days": 3}),
             ("search_standards", {"query": "anything", "top_k": 9})),
        {"expect_refusal": False, "expect_args": {"defect_type": ["open"]}},
    )
    assert result["ok"] is True


def test_the_fixture_only_pins_arguments_the_scorer_reads():
    for question in load_questions(FIXTURE):
        for expected in expectations(question):
            for key in expected.get("expect_args") or {}:
                assert key in SCORED_ARGS, f"{question['question']}: {key}"


def test_the_fixture_only_pins_argument_values_that_exist():
    defects = {"open", "short", "mousebite", "spur", "copper", "pin-hole"}
    for question in load_questions(FIXTURE):
        for expected in expectations(question):
            args = expected.get("expect_args") or {}
            for value in args.get("line_id", []) + args.get("machine_id", []):
                assert value in known_values(), f"{question['question']}: {value}"
            for value in args.get("defect_type", []):
                # `null` pins the *absence* of the argument -- the unclassed
                # ranking -- and is not a class the seed has to create.
                if value is None:
                    continue
                assert value in defects, f"{question['question']}: {value}"


def test_an_answerable_question_naming_a_defect_class_pins_it():
    """Guards against the regression this fix exists for: a question about one
    defect class whose expectation would accept a plan about another."""
    pinned = [
        q for q in load_questions(FIXTURE)
        if (q.get("expect_args") or {}).get("defect_type")
    ]
    assert len(pinned) >= 3


def test_the_overall_machine_rate_question_accepts_the_three_equivalent_plans():
    """Three plans fetch the overall per-machine rate and the fixture takes any.

    `query_defect_history` per machine returns `defects_per_board` directly.
    `query_machine_stats` per defect class returns `per_board` for one class, and
    the six classes are exactly the non-`false_call` set, so summing them is the
    same rate. Since 2026-08-29 `query_machine_stats` with no class ranks every
    machine by `defects_per_board` in one call, and that is the third. A single
    *classed* `query_machine_stats` call is one class, not every class, and
    must still miss -- which is what `defect_type: [null]` on the third
    alternative holds."""
    machines = {m for ms in seed.LINES.values() for m in ms}
    question = next(q for q in load_questions(FIXTURE)
                    if q["question"].startswith("哪一台機器"))
    alternatives = question["expect_any_of"]
    assert len(alternatives) == 3
    history = next(a for a in alternatives if a["expect_tools"] == ["query_defect_history"])
    assert history["expect_args"]["machine_id"] == sorted(machines)
    classed, unclassed = [a for a in alternatives if a["expect_tools"] == ["query_machine_stats"]]
    assert set(classed["expect_args"]["defect_type"]) == {
        "open", "short", "mousebite", "spur", "copper", "pin-hole"
    }
    assert unclassed["expect_args"]["defect_type"] == [None]
    assert score_plan(plan(("query_machine_stats", {"top_n": 1})), question)["ok"] is True


def test_a_per_machine_history_fan_out_scores_a_hit():
    question = next(q for q in load_questions(FIXTURE)
                    if q["question"].startswith("哪一台機器"))
    machines = sorted({m for ms in seed.LINES.values() for m in ms})

    result = score_plan(
        plan(*[("query_defect_history", {"machine_id": m}) for m in machines]),
        question,
    )
    assert result["ok"] is True


def test_a_per_class_machine_stats_fan_out_scores_a_hit():
    """The plan the model actually emits. Six calls, one per defect class,
    each ranking every machine -- summing them is the overall rate."""
    question = next(q for q in load_questions(FIXTURE)
                    if q["question"].startswith("哪一台機器"))

    result = score_plan(
        plan(*[("query_machine_stats", {"defect_type": c})
               for c in ("open", "short", "mousebite", "spur", "copper", "pin-hole")]),
        question,
    )
    assert result["ok"] is True


def test_one_machine_stats_call_still_misses_the_overall_rate_question():
    """The boundary the two accepted plans share: one class is not every class."""
    question = next(q for q in load_questions(FIXTURE)
                    if q["question"].startswith("哪一台機器"))

    result = score_plan(plan(("query_machine_stats", {"defect_type": "open"})),
                        question)
    assert result["ok"] is False


def test_a_hit_records_which_accepted_plan_matched():
    """Otherwise the run record cannot say which of two equivalent plans ran."""
    result = score_plan(
        plan(("query_defect_history", {"machine_id": "M11"})),
        {"expect_refusal": False,
         "expect_any_of": [
             {"why": "history per machine",
              "expect_tools": ["query_defect_history"],
              "expect_args": {"machine_id": ["M11"]}},
             {"why": "stats per class",
              "expect_tools": ["query_machine_stats"]},
         ]},
    )
    assert result["ok"] is True
    assert result["matched"] == "history per machine"


def test_a_plan_matching_no_accepted_alternative_names_them_all():
    result = score_plan(
        plan("search_standards"),
        {"expect_refusal": False,
         "expect_any_of": [
             {"why": "history per machine", "expect_tools": ["query_defect_history"]},
             {"why": "stats per class", "expect_tools": ["query_machine_stats"]},
         ]},
    )
    assert result["ok"] is False
    assert "query_defect_history" in result["reason"]
    assert "query_machine_stats" in result["reason"]


def test_every_accepted_alternative_says_why_it_is_equivalent():
    """A fixture that judges two plans the same owes the reader the reason."""
    for question in load_questions(FIXTURE):
        for alternative in question.get("expect_any_of") or []:
            assert alternative.get("why"), question["question"]


def test_a_call_with_null_args_is_a_miss_not_a_crash():
    """`validate_plan` guards this as `call.get("args") or {}`; so does this."""
    result = score_plan(
        {"interpretation": "i", "assumptions": [],
         "calls": [{"tool": "query_machine_stats", "args": None, "why": "w"}]},
        {"expect_refusal": False, "expect_args": {"defect_type": ["open"]}},
    )
    assert result["ok"] is False
    assert "defect_type" in result["reason"]


def test_the_rendered_plan_shows_the_arguments_that_were_scored():
    """The run record has to distinguish one `query_machine_stats` call from a
    six-call fan-out. Tool names alone cannot."""
    rendered = render_plan(
        plan(("query_machine_stats", {"defect_type": "open", "days": 7}),
             ("query_machine_stats", {"defect_type": "short", "days": 7}))
    )

    assert "defect_type='open'" in rendered
    assert "defect_type='short'" in rendered
    assert "days" not in rendered


def test_a_refusal_renders_as_a_refusal_rather_than_an_empty_string():
    assert render_plan(plan()) == "(refused)"
    assert render_plan(None) == "(no plan)"


# ---------------------------------------------------------------------------
# The report the script writes, and the file it was written into.
# ---------------------------------------------------------------------------

BENCHMARKS = Path(__file__).resolve().parents[1] / "docs" / "benchmarks.md"


def test_the_published_section_carries_the_script_s_own_wording():
    """docs/benchmarks.md is appended to by this script, so the file and the
    generator have to say the same thing. Editing one and not the other leaves
    a document nobody can reproduce, which is worse than an unedited one."""
    published = BENCHMARKS.read_text()

    assert CLEAN_SWEEP in published
    assert HELD_OUT_CAVEAT in published


def test_the_limit_is_printed_above_the_paragraphs_it_limits():
    """A table of three 100%s whose caveat sits four paragraphs below it reads
    as a result with a footnote. It is a question set with a score, and this
    project reports the limit with the number rather than after it.
    """
    published = BENCHMARKS.read_text()
    table_row = "| determinism | 20 |"

    assert published.index(table_row) < published.index(CLEAN_SWEEP)
    assert published.index(CLEAN_SWEEP) < published.index("**Held out from the prompt.**")
    assert published.index(CLEAN_SWEEP) < published.index("\nMisses:")


def test_the_held_out_caveat_names_what_the_number_does_not_bound():
    """It used to read "That is the number to read" -- a sentence that restates
    a 100% while sounding like a correction. A caveat that does not name a
    limit is not one."""
    assert "does not bound" in HELD_OUT_CAVEAT
    assert "number to read" not in HELD_OUT_CAVEAT


def test_plan_only_scores_the_same_and_runs_neither_the_tools_nor_the_prose(
    monkeypatch, capsys
):
    """Everything this script scores is decided by the plan node.

    The tools and the synthesis call therefore produce output nobody reads --
    about half the wall time of a 35-minute run. `--plan-only` stops after the
    planner. This asserts the two paths agree on the score and that the short
    one really does skip the work: the stub tool is never entered, and the
    model is asked once per repeat rather than twice.
    """
    import analysis_eval as ae
    from aoi_agent.llm.ollama import ChatResult, Timing

    ran_tools, chats = [], []

    def stub_tool(**kwargs):
        ran_tools.append(kwargs)
        return {"machines": []}

    for name in PLANNABLE_TOOLS:
        monkeypatch.setitem(PLANNABLE_TOOLS, name, stub_tool)

    plan = {"interpretation": "i", "assumptions": [],
            "calls": [{"tool": "query_machine_stats",
                       "args": {"defect_type": "open"}, "why": "w"}]}

    class StubClient:
        def __init__(self, *a, **kw):
            pass

        def chat(self, messages, **kwargs):
            chats.append(kwargs.get("response_format") is not None)
            text = json.dumps(plan) if kwargs.get("response_format") else "prose"
            return ChatResult(text=text, tool_calls=[], thinking="",
                              timing=Timing(1.0, 0.0, 1.0, 1.0, 10, 10))

    monkeypatch.setattr(ae, "OllamaClient", StubClient)
    monkeypatch.setattr(ae, "store_domains", lambda: {
        "line_id": {"L1", "L2", "L3"},
        "machine_id": {"M11", "M12", "M21", "M22", "M31", "M32"},
        "defect_type": {"open"}, "max_days": 9})

    def run(*extra):
        chats.clear()
        ran_tools.clear()
        monkeypatch.setattr(
            sys, "argv",
            ["analysis_eval.py", "--repeats", "1", "--dry-run", *extra],
        )
        assert ae.main() == 0
        return capsys.readouterr().out

    full = run()
    assert ran_tools, "the default path runs the tools"
    assert False in chats, "the default path writes the prose"

    short = run("--plan-only")
    assert not ran_tools, "--plan-only must not run a tool"
    assert all(chats), "--plan-only must not make the synthesis call"

    def table(output: str) -> list[str]:
        return [line for line in output.splitlines() if line.startswith("| should")]

    assert table(short) == table(full), "the flag must not change a score"
