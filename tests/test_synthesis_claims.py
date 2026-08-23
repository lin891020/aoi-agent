"""Does the claim checker catch a summary that is wrong on purpose?

No model is called here, and none should be. The checker's whole value is that
it is deterministic: every verdict it reaches is a comparison against a stored
payload, so its own tests can be a fixture and an assertion rather than a run.

Two halves, and both are load-bearing. The corrupted summary proves the
taxonomy can fail something -- a checker written by the author of the thing it
checks can be lenient without anyone noticing, and "we measured it and it was
fine" reads identically to "nothing here can fail". The faithful summary proves
it is not merely loud: a correct answer over the same results must produce
nothing at all, or every published count is noise.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from aoi_agent.analysis.claims import (
    CHECKED_KINDS,
    KINDS,
    Grounding,
    check,
    gaps,
    normalise,
    numbers_in,
    perturbations,
    sentences,
)

FIXTURE = Path(__file__).parent / "fixtures" / "synthesis_wrong_summary.json"


@pytest.fixture(scope="module")
def case() -> dict:
    return json.loads(FIXTURE.read_text())


def kinds_of(findings) -> set[str]:
    return {finding.kind for finding in findings}


def test_the_corrupted_summary_fails_on_every_kind(case):
    findings, _waved = check(case["corrupted"], case["plan"], case["results"])
    assert kinds_of(findings) == set(KINDS), (
        "a kind that never fires is a kind that measures nothing: "
        f"missing {set(KINDS) - kinds_of(findings)}"
    )


def test_every_kind_the_fixture_promises_is_named_in_the_taxonomy(case):
    assert set(case["expected_kinds"]) == set(KINDS)


def test_the_faithful_summary_produces_nothing(case):
    findings, _waved = check(case["faithful"], case["plan"], case["results"])
    assert findings == [], [f"{f.kind}: {f.claim} — {f.sentence}" for f in findings]


def test_the_fabricated_figure_is_the_one_that_is_not_there(case):
    findings, _waved = check(case["corrupted"], case["plan"], case["results"])
    fabricated = [f.claim for f in findings if f.kind == "fabricated_figure"]
    assert "1412" in fabricated


def test_the_misattribution_names_the_line_the_figure_belongs_to(case):
    findings, _waved = check(case["corrupted"], case["plan"], case["results"])
    swapped = [f for f in findings if f.kind == "misattributed_figure"]
    assert swapped, "L2's open count printed under L1 went unnoticed"
    assert any("L2" in f.evidence for f in swapped)


def test_a_figure_the_plan_stated_is_not_called_a_fabrication(case):
    """`SYNTHESIS_PROMPT` orders the assumptions repeated. Scoring the synthesis
    node for the planner's numbers would measure the wrong node."""
    prose = "Working to a 7-day window, L1 recorded 966 defects."
    _findings, waved = check(prose, case["plan"], case["results"])
    assert waved == 0  # 7 is in the payload's own `days`
    plan = {"interpretation": "", "assumptions": ["A 24-hour window was used."],
            "calls": []}
    findings, waved = check("A 24-hour window was used.", plan, case["results"])
    assert waved == 1
    assert not [f for f in findings if f.kind == "fabricated_figure"]


def test_a_figure_quoted_out_of_a_passage_is_not_a_misattribution(case):
    """A sentence repeating "Class 3 product" beside a machine name is quoting a
    document, not attributing a figure to that machine."""
    grounding = Grounding(case["results"])
    assert Decimal("1.1291") in grounding.everything
    prose = "For open, the criteria state that continuity is binary."
    findings, _waved = check(prose, case["plan"], case["results"])
    assert not [f for f in findings if f.kind == "misattributed_figure"]


def test_the_checked_kinds_need_no_judgement(case):
    """The split the report leads with. If a judged kind creeps into
    `CHECKED_KINDS` the published "checkable" fraction silently inflates."""
    assert set(CHECKED_KINDS) == {"fabricated_figure", "misattributed_figure"}
    assert set(CHECKED_KINDS) < set(KINDS)


def test_a_failed_tool_is_a_gap_whether_or_not_the_prose_admits_it(case):
    assert any("query_machine_stats" in gap for gap in gaps(case["results"]))
    findings, _waved = check(case["faithful"], case["plan"], case["results"])
    assert not [f for f in findings if f.kind == "unhedged_gap"]


def test_the_models_typographic_hyphens_do_not_hide_a_figure():
    """`gpt-oss:20b` writes U+2011. A checker that cannot read its output is a
    checker that finds nothing and reports a clean sweep."""
    raw = "7‑day window ending 2026‑08‑09"
    assert [value for value, *_ in numbers_in(normalise(raw))] == [
        Decimal(7), Decimal(2026), Decimal(8), Decimal(9)
    ]


def test_a_thousands_separator_is_one_figure():
    assert [value for value, *_ in numbers_in("1,049 defects")] == [Decimal(1049)]


def test_bullets_split_as_hard_as_full_stops():
    assert len(sentences("- M11 is clean\n- M22 is not")) == 2


def test_perturbing_a_grounded_rate_stops_it_grounding(case):
    """The leniency test, on the fixture rather than in an argument. A checker
    that accepts a rate moved 30% is not checking rates."""
    grounding = Grounding(case["results"])
    _accepted, _tried, accepted_decimal, tried_decimal = perturbations(
        case["faithful"], grounding
    )
    assert tried_decimal >= 4
    assert accepted_decimal / tried_decimal < 0.25


def test_a_spelled_out_count_is_not_read_as_a_figure(case):
    """"The worse of the two lines" is a count of lines, and reading the `two`
    as a measurement produced a swap finding against a sentence whose only
    fault was a wrong characterisation the checker cannot see anyway."""
    prose = "Line L1 is the worse of the two lines, at 7.05 defects per board."
    findings, _waved = check(prose, case["plan"], case["results"])
    assert not [f for f in findings if f.kind in CHECKED_KINDS]


def test_a_wrong_characterisation_over_right_figures_is_outside_the_taxonomy(case):
    """The boundary the published section states, asserted rather than claimed.

    L2 is the worse line at 7.39 against L1's 7.05. Every figure in this
    sentence is real and attached to the right line; the reading of them is
    wrong, and no kind here covers a reading. If this ever starts failing the
    report's stated limit has moved and the section has to say so.
    """
    prose = "Line L1 is the worse line, at 7.05 defects per board."
    findings, _waved = check(prose, case["plan"], case["results"])
    assert not [f for f in findings if f.kind in CHECKED_KINDS]
