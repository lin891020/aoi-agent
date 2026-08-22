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
    PLANNABLE_TOOLS,
    store_domains,
    validate_plan,
)

DOMAINS = {
    "line_id": {"L1", "L2", "L3"},
    "machine_id": {"M11", "M12", "M21", "M22", "M31", "M32"},
    "defect_type": {"open", "short", "mousebite", "spur", "copper", "pin-hole"},
    "max_days": 9,
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
    errors = validate_plan(plan(("query_machine_stats", {"days": 7})), DOMAINS)
    assert any("defect_type" in e for e in errors)


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
