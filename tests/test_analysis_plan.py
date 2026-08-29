"""Validating a plan before anything runs.

The layer that matters is the third. A tool name that does not exist raises;
an argument name that does not exist raises; but `line_id="L4"` raises nothing
at all -- it returns an empty result, the chart comes back with one fewer line,
and nobody notices. That is the failure this whole file exists to prevent, and
it is the same argument as the project's no-SQL invariant.
"""

from __future__ import annotations

import pytest

from aoi_agent.analysis.plan import (
    DOMAIN_OF,
    PLANNABLE_TOOLS,
    Domains,
    store_domains,
    validate_plan,
)

DOMAINS = {
    "line_id": {"L1", "L2", "L3"},
    "machine_id": {"M11", "M12", "M21", "M22", "M31", "M32"},
    "defect_type": {"open", "short", "mousebite", "spur", "copper", "pin-hole"},
    # Wider than `defect_type` by one: the criteria are asked about a class the
    # classifier emitted, and `false_call` is one of those.
    "defect_class": {"open", "short", "mousebite", "spur", "copper", "pin-hole",
                     "false_call"},
    # The tool's own axis vocabulary, not a store fact -- see plan.DOMAIN_OF.
    "group_by": {"machine", "line", "shift"},
    "relative_to": {"parameter_change"},
    "side": {"before", "after"},
    "max_days": 9,
    # The span the store covers, as dates. `date_from`/`date_to` are checked
    # against it the way `days` is checked against `max_days`: a date outside
    # it returns nothing and would read as a finding.
    "date_span": ("2026-08-01", "2026-08-09"),
}


def plan(*calls) -> dict:
    return {
        "interpretation": "how the question was read",
        "assumptions": ["compared against the fleet average"],
        "calls": [
            {"tool": t, "args": a, "why": "because"} for t, a in calls
        ],
    }


def test_a_well_formed_plan_validates():
    errors = validate_plan(
        plan(("query_machine_stats", {"defect_type": "open", "days": 7})), DOMAINS
    )
    assert errors == []


def test_an_unknown_tool_is_rejected():
    errors = validate_plan(plan(("drop_tables", {})), DOMAINS)
    assert len(errors) == 1
    assert "drop_tables" in errors[0]


def test_classify_defect_is_not_plannable():
    """It loads a torch model onto MPS. Ten of them in one fan-out is ten
    contentions, so it is kept out of the registry rather than rate-limited."""
    assert "classify_defect" not in PLANNABLE_TOOLS
    errors = validate_plan(plan(("classify_defect", {"candidate_ref": "1#0"})), DOMAINS)
    assert errors


def test_an_unknown_argument_name_is_rejected():
    errors = validate_plan(
        plan(("query_machine_stats", {"defect_type": "open", "weeks": 3})), DOMAINS
    )
    assert any("weeks" in e for e in errors)


def test_a_missing_required_argument_is_rejected():
    # `query_board_context` is the tool with a required argument now that
    # `query_machine_stats` ranks every class when no class is named.
    errors = validate_plan(plan(("query_board_context", {})), DOMAINS)
    assert any("board" in e for e in errors)


def test_an_optional_argument_may_be_omitted():
    errors = validate_plan(plan(("query_defect_history", {"days": 7})), DOMAINS)
    assert errors == []


@pytest.mark.parametrize(
    "args,bad",
    [
        ({"defect_type": "open", "days": 7, "line_id": "L4"}, "L4"),
        ({"defect_type": "open", "days": 7, "machine_id": "M99"}, "M99"),
        ({"defect_type": "scratch", "days": 7}, "scratch"),
    ],
)
def test_a_legal_looking_value_outside_its_domain_is_rejected(args, bad):
    """The quiet failure. `line_id="L4"` is a valid string and a valid
    parameter; it simply matches nothing, and a chart with a missing series
    reads as a finding rather than as a bug."""
    errors = validate_plan(plan(("query_defect_history", args)), DOMAINS)
    assert any(bad in e for e in errors)


def test_a_window_longer_than_the_data_is_rejected():
    """Asking for 30 days of an 8-day store does not error, it silently
    returns the same 8 days -- so a month-on-month comparison would report the
    two windows as identical."""
    errors = validate_plan(
        plan(("query_defect_history", {"days": 30})), DOMAINS
    )
    assert any("30" in e for e in errors)


def test_every_error_is_reported_not_just_the_first():
    """The plan is shown to the user. Fixing one error at a time across
    several model round trips is worse than seeing them all at once."""
    errors = validate_plan(
        plan(
            ("nope", {}),
            ("query_defect_history", {"line_id": "L9", "days": 999}),
        ),
        DOMAINS,
    )
    assert len(errors) >= 3


def test_an_empty_plan_is_rejected():
    errors = validate_plan(plan(), DOMAINS)
    assert any("no calls" in e.lower() for e in errors)


@pytest.mark.dataset
def test_store_domains_reads_the_real_store():
    domains = store_domains()
    assert domains["line_id"] == {"L1", "L2", "L3"}
    assert len(domains["machine_id"]) == 6
    assert 1 <= domains["max_days"] <= 400


def test_this_files_domains_carry_every_key_the_validator_looks_up():
    """The fixture is the validator's whole world here. A domain added to
    `Domains` and not to this dict does not fail a test on the way in -- it
    raises `KeyError` out of the validator the first time a plan names that
    argument, which on `/ask` is a page that does not render."""
    assert set(DOMAINS) == set(Domains.__annotations__)
    for argument, domain in DOMAIN_OF.items():
        assert domain in DOMAINS, argument


@pytest.mark.parametrize("scope", ["open", "false_call"])
def test_a_standards_scope_the_classifier_can_emit_is_accepted(scope):
    errors = validate_plan(
        plan(("search_standards", {"query": "x", "defect_class": scope})), DOMAINS
    )
    assert errors == []


def test_a_standards_scope_nobody_classifies_is_rejected_before_it_runs():
    """`defect_class="pinhole"` is a legal string in a real parameter. The tool
    would refuse it, but the third validation layer exists so that a plan is
    shown to the person whole rather than one failed branch at a time."""
    errors = validate_plan(
        plan(("search_standards", {"query": "x", "defect_class": "pinhole"})), DOMAINS
    )
    assert any("pinhole" in e for e in errors)


def test_a_tool_taking_kwargs_accepts_arguments_rather_than_requiring_one(monkeypatch):
    """`**kwargs` is not an argument named "kwargs" that every plan is missing,
    and it is not a wall that rejects every argument passed through it. Read the
    other way round -- which is how this read before -- a tool with a catch-all
    signature is unusable: every plan naming it is rejected twice over, for an
    argument it does accept and for one that does not exist."""
    def catch_all(**kwargs):
        return {}

    monkeypatch.setitem(PLANNABLE_TOOLS, "search_standards", catch_all)

    errors = validate_plan(plan(("search_standards", {"query": "open"})), DOMAINS)

    assert errors == []


# --- dated windows and top-N -------------------------------------------------

def test_a_date_inside_the_span_validates():
    errors = validate_plan(
        plan(("query_machine_stats", {"date_from": "2026-08-05", "date_to": "2026-08-05", "top_n": 5})),
        DOMAINS,
    )
    assert errors == []


def test_a_date_outside_the_span_is_refused_and_the_span_is_named():
    """«2026-07-30 前 5 名» is the question that motivated the parameter, and
    the store does not hold that day. The tool would return nothing for it;
    the validator refuses it and says what days exist."""
    errors = validate_plan(
        plan(("query_defect_history", {"date_from": "2026-07-30", "date_to": "2026-07-30"})),
        DOMAINS,
    )
    assert len(errors) == 2
    assert all("2026-08-01" in e and "2026-08-09" in e for e in errors)


def test_a_date_that_does_not_parse_is_refused():
    errors = validate_plan(plan(("query_defect_history", {"date_from": "7/30"})), DOMAINS)
    assert errors and "date_from" in errors[0]


def test_a_top_n_below_one_is_refused_before_it_runs():
    errors = validate_plan(plan(("query_machine_stats", {"top_n": 0})), DOMAINS)
    assert errors and "top_n" in errors[0]
