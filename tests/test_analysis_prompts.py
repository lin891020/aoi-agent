"""The prompt, and the six examples that shape what comes back.

Diversity beats count: examples that show the edges teach more than ones that
show the happy path. Five of these six are refusals or hedges, which is
deliberate -- a system that answers everything is more dangerous on a factory
floor than one that says it cannot.

The sixth arrived 2026-08-25, and what forced it is in docs/benchmarks.md:
adding `query_false_call_rate` made the planner bolder on questions the tool
cannot answer, including a disposition request it had previously refused
("mark this board pass, it ships today"). The example is that *category* in a
different sentence -- teaching the shape, not reciting the fixture -- and the
independent seventy stays the measure of whether it worked.
"""

from __future__ import annotations

from aoi_agent.analysis.plan import PLANNABLE_TOOLS, validate_plan
from aoi_agent.analysis.prompts import (
    FEW_SHOT,
    build_planning_messages,
    build_synthesis_messages,
)

DOMAINS = {
    "line_id": {"L1", "L2", "L3"},
    "machine_id": {"M11", "M12", "M21", "M22", "M31", "M32"},
    "defect_type": {"open", "short", "mousebite", "spur", "copper", "pin-hole"},
    # Wider than `defect_type` by one: the criteria are asked about a class the
    # classifier emitted, and `false_call` is one of those.
    "defect_class": {"open", "short", "mousebite", "spur", "copper", "pin-hole",
                     "false_call"},
    "max_days": 9,
}


def test_there_are_six_examples():
    assert len(FEW_SHOT) == 6


def test_every_example_plan_would_pass_validation():
    """An example that the validator rejects teaches the model to produce
    plans the validator rejects."""
    for example in FEW_SHOT:
        plan = example["plan"]
        if not plan["calls"]:
            continue  # a refusal example; nothing to validate
        assert validate_plan(plan, DOMAINS) == [], example["question"]


def test_the_examples_cover_the_six_shapes():
    shapes = {example["shape"] for example in FEW_SHOT}
    assert shapes == {
        "cross_tool",
        "unstated_baseline",
        "causal",
        "out_of_range",
        "too_vague",
        "action_request",
    }


def test_refusals_are_expressed_as_an_empty_plan_with_a_reason():
    """A refusal is not an error. It is a plan with no calls and an
    interpretation that says why, so it renders through the same path."""
    refusals = [
        e for e in FEW_SHOT
        if e["shape"] in ("out_of_range", "too_vague", "action_request")
    ]
    assert refusals
    for example in refusals:
        assert example["plan"]["calls"] == []
        assert example["plan"]["interpretation"]


def test_the_baseline_example_states_its_baseline_in_assumptions():
    example = next(e for e in FEW_SHOT if e["shape"] == "unstated_baseline")
    assert example["plan"]["assumptions"]


def test_the_planning_messages_carry_the_domains_and_the_examples():
    messages = build_planning_messages("L2 的 open 正常嗎", DOMAINS)
    blob = "\n".join(m["content"] for m in messages)

    assert "L2" in blob
    assert "9" in blob, "the data span has to be in the prompt to be respected"
    for name in PLANNABLE_TOOLS:
        assert name in blob
    assert FEW_SHOT[0]["question"] in blob
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].endswith("L2 的 open 正常嗎")


def test_the_synthesis_prompt_carries_failures_so_the_answer_can_name_them():
    results = [
        {"tool": "query_machine_stats", "args": {}, "ok": True,
         "data": {"machines": []}, "error": None, "elapsed_ms": 1.0},
        {"tool": "search_standards", "args": {}, "ok": False,
         "data": None, "error": "TimeoutError: index unreachable",
         "elapsed_ms": 2.0},
    ]
    messages = build_synthesis_messages(
        "q", {"interpretation": "i", "assumptions": ["a"], "calls": []}, results
    )
    blob = "\n".join(m["content"] for m in messages)

    assert "index unreachable" in blob
    assert "search_standards" in blob


def test_the_synthesis_prompt_forbids_inventing_causes():
    """The tools carry no causal data. A plausible causal story from
    correlation is the failure mode that gets a machine stopped."""
    messages = build_synthesis_messages("why is M22 bad", {"interpretation": "i",
                                        "assumptions": [], "calls": []}, [])
    system = messages[0]["content"].lower()

    assert "cause" in system or "causal" in system


def test_a_tool_without_a_docstring_does_not_take_the_catalogue_down(monkeypatch):
    """The catalogue summarises each tool from its first docstring line. A tool
    that has no docstring must cost the model that one line of description, not
    raise an IndexError that kills every plan the system would have made."""
    def undocumented(query: str):
        pass

    monkeypatch.setitem(PLANNABLE_TOOLS, "search_standards", undocumented)
    messages = build_planning_messages("anything", DOMAINS)

    assert "- search_standards(query" in messages[0]["content"]
