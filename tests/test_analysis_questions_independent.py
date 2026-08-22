"""The independent question set, checked against the system it grades.

Seventy questions written by three authors who never saw the planner's prompt,
its few-shot examples, or `analysis_questions.json`: thirty-five by an author
told nothing about the tools and asked to write what a shift supervisor would
type, thirty-five by an author given only the five tool signatures and asked to
probe the boundary, and a verdict on all seventy from a third author who read
the tools and the store's source but not the prompt. That is the point of the
set and the whole of its value: the existing fixture's author also wrote the
prompt, so its 100% bounds nothing.

Blind authorship is exactly what makes the graded file untrustworthy as a
fixture. An author who has not read `PLANNABLE_TOOLS` can pin an argument onto
a tool that has no such parameter, and the eval would then score a planner
failure that is really a grading error. So the fixture is checked here against
the live registry, the live value domains and the live validator, and the
grader's mistakes are named rather than repaired -- `fixture_defect` marks the
questions no plan can pass, and the eval reports them apart from the misses.

No model is called. These are file-and-signature checks.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from aoi_agent.analysis.plan import PLANNABLE_TOOLS, store_domains, validate_plan
from aoi_agent.store import seed

from analysis_eval import (
    BLIND_TO_THE_PROMPT,
    DAYS_DEFAULT_DEFECT,
    NO_FALSE_CALL_METRIC,
    QUESTIONS,
    SCORED_ARGS,
    load_questions,
)

FIXTURE = Path(__file__).parent / "fixtures" / "analysis_questions_independent.json"

SEVERITIES = {"core", "boundary", "stretch"}

DEFECT_CLASSES = {"open", "short", "mousebite", "spur", "copper", "pin-hole"}

#: Values for the arguments a tool requires but the fixture does not pin. They
#: only have to be legal -- nothing here runs a query.
FILLER = {"defect_type": "open", "board": "20085294", "query": "acceptance criteria"}

#: Every argument the graded file pinned onto a tool that cannot take it, as
#: `(id, branch index, argument)`. Both shapes are the same author error: the
#: grader recorded the entity the *question* named rather than the argument the
#: *call* would carry.
#:
#: Seven of them are `defect_type` on a `search_standards`-only plan.
#: `search_standards` takes `query`; there is no `defect_type` parameter to
#: carry the class, so `validate_plan` would throw out any plan that tried, and
#: those seven questions score a miss however well the planner does. They are
#: marked `fixture_defect` in the fixture and reported apart from the misses.
#:
#: A10's two are milder: its primary branch pins `machine_id` and `line_id` onto
#: `query_machine_stats`, which takes only `defect_type` and `days`, but its
#: alternative branch is `query_defect_history`, which takes all three. The
#: question is still scorable, so it is not marked.
#:
#: This is a ledger, not a waiver. It is asserted to be exact, so a tenth
#: appearing -- or one of these being fixed upstream -- fails the suite.
STRANDED_ARGUMENTS = {
    ("S05", 0, "defect_type"),
    ("S18", 0, "defect_type"),
    ("S28", 0, "defect_type"),
    ("S33", 0, "defect_type"),
    ("A10", 0, "machine_id"),
    ("A10", 0, "line_id"),
    ("A11", 0, "defect_type"),
    ("A20", 0, "defect_type"),
    ("A32", 0, "defect_type"),
}

#: The six questions that ask for a false-call count or rate at an aggregate
#: level. No tool returns one: `query_defect_history` excludes
#: `predicted_class='false_call'` and `query_machine_stats` accepts only the six
#: real classes. In a system whose subject is false calls, that is the sharpest
#: thing this set found, and a refusal is the correct behaviour for the system
#: as built -- so it is pinned here rather than left to a reader of the JSON.
FALSE_CALL_QUESTIONS = {"S02", "S06", "S12", "S16", "S22", "S31"}


def questions() -> list[dict]:
    return load_questions(FIXTURE)


def parameters(tool: str) -> set[str]:
    return set(inspect.signature(PLANNABLE_TOOLS[tool]).parameters)


def branches(question: dict) -> list[tuple[int, dict]]:
    """Each accepted plan, merged onto the question, with its index."""
    alternatives = question.get("expect_any_of")
    if not alternatives:
        return [(0, question)]
    return [(index, {**question, **alternative})
            for index, alternative in enumerate(alternatives)]


def answer_branches(question: dict) -> list[tuple[int, dict]]:
    return [(index, branch) for index, branch in branches(question)
            if not branch.get("expect_refusal")]


def stranded(branch: dict) -> list[str]:
    """Arguments this branch pins that none of its tools can carry."""
    tools = [t for t in (branch.get("expect_tools") or []) if t in PLANNABLE_TOOLS]
    return [key for key in (branch.get("expect_args") or {})
            if not any(key in parameters(tool) for tool in tools)]


def materialise(branch: dict) -> dict:
    """The smallest plan that would score this branch a hit.

    One call per tool, fanned out where the branch pins several values for an
    argument -- three `line_id`s is three calls, and it has to be, because
    `defects_per_board` comes back per call. Required arguments the fixture does
    not pin are filled with something legal, since the point is to run the
    expectation past `validate_plan`, not past the store.
    """
    calls = []
    for tool in dict.fromkeys(branch.get("expect_tools") or []):
        if tool not in PLANNABLE_TOOLS:
            continue
        takes = parameters(tool)
        pinned = {key: values for key, values in (branch.get("expect_args") or {}).items()
                  if key in takes}
        width = max((len(values) for values in pinned.values()), default=1)
        for position in range(width):
            args = {key: values[min(position, len(values) - 1)]
                    for key, values in pinned.items()}
            for name, parameter in inspect.signature(PLANNABLE_TOOLS[tool]).parameters.items():
                if parameter.default is inspect.Parameter.empty and name not in args:
                    args[name] = FILLER[name]
            calls.append({"tool": tool, "args": args, "why": "the expected plan"})
    return {"interpretation": "i", "assumptions": [], "calls": calls}


def live_domains() -> dict:
    """The store's domains, or the seeder's if no store has been built.

    `store_domains()` degrades to empty sets rather than raising when the
    database is missing, and falling back to empty would make the value checks
    below pass vacuously on a fresh checkout. `seed.LINES` is what the seeder
    creates, so it is the same domain by construction.
    """
    domains = store_domains()
    if domains["line_id"]:
        return domains
    return {"line_id": set(seed.LINES),
            "machine_id": {m for ms in seed.LINES.values() for m in ms},
            "defect_type": DEFECT_CLASSES,
            "max_days": domains["max_days"]}


# ---------------------------------------------------------------------------
# The set is what it claims to be.
# ---------------------------------------------------------------------------


def test_the_whole_graded_set_is_carried_over():
    rows = questions()
    ids = [q["id"] for q in rows]

    assert len(rows) == 70
    assert len(set(ids)) == 70
    assert sum(1 for i in ids if i.startswith("S")) == 35
    assert sum(1 for i in ids if i.startswith("A")) == 35


def test_every_question_carries_the_severity_and_the_reason_it_was_graded_on():
    """A `core` failure says the system is not fit for the floor and a
    `boundary` failure is a judgement call. A score that cannot tell them apart
    is one number covering two different findings."""
    for question in questions():
        assert question["severity"] in SEVERITIES, question["id"]
        assert question["reason"].strip(), question["id"]


def test_the_set_covers_both_outcomes_at_both_severities():
    rows = questions()

    for severity in ("core", "boundary"):
        at = [q for q in rows if q["severity"] == severity]
        assert any(q["expect_refusal"] for q in at), severity
        assert any(not q["expect_refusal"] for q in at), severity


def test_every_accepted_alternative_says_why_it_is_equivalent():
    for question in questions():
        for alternative in question.get("expect_any_of") or []:
            assert alternative.get("why"), question["id"]


def test_an_alternative_never_inherits_an_argument_its_tools_cannot_carry():
    """`score_plan` merges each alternative onto the question, so a branch with
    no `expect_args` of its own would silently inherit the primary's. Where the
    branch changes tools that inheritance is wrong -- a `query_machine_stats`
    fan-out cannot carry `machine_id` -- so every branch states its arguments."""
    for question in questions():
        for alternative in question.get("expect_any_of") or []:
            assert "expect_args" in alternative, question["id"]


# ---------------------------------------------------------------------------
# The set is checked against the system, not against prose.
# ---------------------------------------------------------------------------


def test_every_expected_tool_is_one_the_planner_can_actually_call():
    """The authors were blind to the prompt, and two of the three were blind to
    the code. A fixture naming a tool that does not exist scores every plan a
    miss and reads as a planner failure."""
    for question in questions():
        for _, branch in branches(question):
            for tool in branch.get("expect_tools") or []:
                assert tool in PLANNABLE_TOOLS, f"{question['id']}: {tool}"


def test_the_fixture_only_pins_arguments_the_scorer_reads():
    for question in questions():
        for _, branch in branches(question):
            for key in branch.get("expect_args") or {}:
                assert key in SCORED_ARGS, f"{question['id']}: {key}"


def test_every_pinned_value_exists_in_the_store():
    """`line_id='L4'` raises nothing and returns nothing. An expectation naming
    a value the store does not hold would demand a plan the validator refuses."""
    domains = live_domains()
    for question in questions():
        for _, branch in branches(question):
            for key, values in (branch.get("expect_args") or {}).items():
                if key not in domains:
                    continue
                for value in values:
                    assert value in domains[key], f"{question['id']}: {key}={value}"


def test_every_expected_plan_passes_the_validator_that_gates_the_real_ones():
    """The check that catches what the other two miss. A branch can name real
    tools and real values and still describe a plan `validate_plan` throws out
    -- a required argument omitted, or an argument on a tool that has no such
    parameter."""
    domains = live_domains()
    for question in questions():
        for index, branch in answer_branches(question):
            if stranded(branch):
                continue  # recorded in STRANDED_ARGUMENTS, asserted below
            errors = validate_plan(materialise(branch), domains)
            assert not errors, f"{question['id']} branch {index}: {errors}"


def test_the_arguments_the_graders_stranded_are_exactly_the_ones_on_record():
    """The grader errors this fixture is known to carry, named rather than
    repaired. Asserted exact in both directions: a new one fails here instead of
    showing up in `docs/benchmarks.md` as a planner that cannot retrieve a work
    instruction."""
    found = {
        (question["id"], index, key)
        for question in questions()
        for index, branch in answer_branches(question)
        for key in stranded(branch)
    }
    assert found == STRANDED_ARGUMENTS


def test_a_question_no_plan_can_pass_is_marked_as_such():
    """Marked, not fixed. The expectation stands as graded; the eval reports
    these apart from the misses so the score is not read as a planner that
    cannot answer 'what does the WI say about open'."""
    unpassable = {
        question["id"] for question in questions()
        if answer_branches(question)
        and all(stranded(branch) for _, branch in answer_branches(question))
    }
    marked = {q["id"] for q in questions() if q.get("fixture_defect")}

    assert marked == unpassable
    assert marked == {"S05", "S18", "S28", "S33", "A11", "A20", "A32"}
    for question in questions():
        if question.get("fixture_defect"):
            assert "search_standards" in question["fixture_defect"], question["id"]


def test_a_refusal_expects_no_tools_and_pins_no_arguments():
    for question in questions():
        if not question["expect_refusal"]:
            continue
        assert not question["expect_tools"], question["id"]
        assert not question["expect_args"], question["id"]


# ---------------------------------------------------------------------------
# The gap the set was built to find.
# ---------------------------------------------------------------------------


def test_the_false_call_questions_expect_a_refusal():
    """Six questions ask for a false-call count or rate at an aggregate level,
    and no tool returns one: `query_defect_history` excludes
    `predicted_class='false_call'`, `query_machine_stats` takes only the six
    real classes. A refusal is right for the system as built. Answering with
    confirmed-defect counts under a false-call label is the failure to catch,
    and it is the one this set was most likely to find, because the supervisor
    author did not know which numbers the tools hold."""
    rows = {q["id"]: q for q in questions()}

    for question_id in FALSE_CALL_QUESTIONS:
        assert rows[question_id]["expect_refusal"], question_id
    asked = {q["id"] for q in rows.values()
             if "false call" in q["reason"].lower()
             or "false-call" in q["reason"].lower()
             or "false_call" in q["reason"].lower()}
    assert FALSE_CALL_QUESTIONS <= asked


def test_no_tool_returns_a_false_call_aggregate():
    """The other half of the claim above, checked against the code rather than
    against the fixture, so the day a tool starts returning one the expectations
    fail here instead of silently becoming wrong."""
    from aoi_agent.mcp_servers.production import DEFECT_CLASSES

    assert "false_call" not in DEFECT_CLASSES

    for tool in ("query_defect_history", "query_machine_stats"):
        source = inspect.getsource(PLANNABLE_TOOLS[tool])
        # The only mention either tool makes of the class is to exclude it from
        # a denominator. Neither ever counts it.
        assert 'predicted_class != "false_call"' in source, tool
        assert 'predicted_class == "false_call"' not in source, tool


def test_the_published_twenty_are_still_what_the_script_runs_by_default():
    """Two sets, two sections, and the older one's figures have to stay
    reproducible. `--questions` is what makes the seventy runnable; making them
    the default would silently reinterpret every number already published."""
    assert QUESTIONS.name == "analysis_questions.json"
    assert len(load_questions(QUESTIONS)) == 20
    assert QUESTIONS != FIXTURE


BENCHMARKS = Path(__file__).resolve().parents[1] / "docs" / "benchmarks.md"


def test_the_published_section_carries_the_script_s_own_wording():
    """The section in docs/benchmarks.md is appended by the script, so the file
    and the generator have to say the same thing. Editing one and not the other
    leaves a document nobody can reproduce.

    All three of these are claims about the run rather than numbers from it: who
    wrote the questions, and the two defects the set exposes that no score can
    show."""
    published = BENCHMARKS.read_text()

    assert BLIND_TO_THE_PROMPT in published
    assert DAYS_DEFAULT_DEFECT in published
    assert NO_FALSE_CALL_METRIC in published


def test_the_published_section_did_not_displace_the_original_one():
    """Two question sets, two sections, and the older one's figures stay put.
    They are not comparable and the new section is not a correction of the old."""
    published = BENCHMARKS.read_text()

    assert published.count("### Analysis planner") == 2
    assert published.index("### Analysis planner —") < published.index(
        "### Analysis planner, asked by someone else"
    )
