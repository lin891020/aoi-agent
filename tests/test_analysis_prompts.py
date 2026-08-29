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
    "group_by": {"machine", "line", "defect_type"},
    "relative_to": {"parameter_change", "maintenance", "lamp_replaced",
                    "nozzle_cleaned"},
    "side": {"before", "after"},
    "date_span": ("2026-08-01", "2026-08-09"),
}


def test_there_are_eight_examples():
    assert len(FEW_SHOT) == 8


def test_every_example_plan_would_pass_validation():
    """An example that the validator rejects teaches the model to produce
    plans the validator rejects."""
    for example in FEW_SHOT:
        plan = example["plan"]
        if not plan["calls"]:
            continue  # a refusal example; nothing to validate
        assert validate_plan(plan, DOMAINS) == [], example["question"]


def test_the_examples_cover_the_seven_shapes():
    shapes = {example["shape"] for example in FEW_SHOT}
    assert shapes == {
        "cross_tool",
        "unstated_baseline",
        "causal",
        "out_of_range",
        "too_vague",
        "action_request",
        "event_window",
        "dated_top_n",
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


def test_the_catalogue_tells_the_model_what_relative_to_and_side_mean():
    """The first before/after question asked at the station was refused with
    "neither tool allows filtering by a time boundary relative to an event".
    The tool did; the catalogue showed one sentence per tool and the two
    parameters reached the model as bare names. The docstring's own paragraph
    -- call it twice, once per side -- has to be in the prompt, not only in
    the source."""
    messages = build_planning_messages("M32 參數變更前後有差嗎", DOMAINS)
    blob = "\n".join(m["content"] for m in messages)

    assert 'side="before"' in blob and 'side="after"' in blob
    assert "relative_to: The kind of machine event" in blob


def test_the_rules_do_not_name_before_and_after_as_a_missing_dimension():
    """The rule listing dimensions no tool expresses used "a before/after
    boundary" as its example, written before the event tool existed. Left in
    place beside a tool that expresses exactly that, it read as an instruction
    to refuse the questions the tool was added to answer."""
    messages = build_planning_messages("M32 參數變更前後有差嗎", DOMAINS)
    system = messages[0]["content"]

    assert "before/after boundary" not in system
    assert "relative_to" in system and "one call per side" in system


def test_the_event_window_example_is_the_two_call_shape():
    example = next(e for e in FEW_SHOT if e["shape"] == "event_window")
    calls = example["plan"]["calls"]
    sides = [c["args"].get("side") for c in calls if c["tool"] == "query_defect_history"]

    assert sorted(sides) == ["after", "before"]
    assert {c["args"].get("relative_to") for c in calls
            if c["tool"] == "query_defect_history"} == {"parameter_change"}
    assert any(c["tool"] == "query_machine_events" for c in calls)
    assert validate_plan(example["plan"], DOMAINS) == []


def test_the_prompt_lists_the_event_kinds_the_store_holds():
    """Asked about a lamp replacement on M31, the planner anchored on
    `parameter_change` -- the one kind its example names -- because nothing
    told it `lamp_replaced` existed. The kinds are a value domain like the
    machines are, and the model has to be shown the domain to stay inside it."""
    messages = build_planning_messages("M31 換燈前後有差嗎", DOMAINS)
    blob = "\n".join(m["content"] for m in messages)

    assert "lamp_replaced" in blob
    assert "nozzle_cleaned" in blob


def test_the_domain_note_names_the_dates_held_so_a_date_can_be_planned():
    """The planner is shown which days exist, the same way it is shown which
    machines exist. Without it «2026-07-30» is a date it has no way to know is
    outside the data, and «2026-08-05» one it has no way to know is inside."""
    system = build_planning_messages("q", DOMAINS)[0]["content"]
    assert "2026-08-01" in system and "2026-08-09" in system
    assert "date_from" in system


def test_the_dated_top_n_example_uses_a_date_the_domain_holds():
    example = next(e for e in FEW_SHOT if e["shape"] == "dated_top_n")
    call = example["plan"]["calls"][0]
    assert call["tool"] == "query_machine_stats"
    assert call["args"]["top_n"] == 5
    assert DOMAINS["date_span"][0] <= call["args"]["date_from"] <= DOMAINS["date_span"][1]
