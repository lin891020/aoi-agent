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
    findings, _waved, _derived = check(case["corrupted"], case["plan"], case["results"])
    assert kinds_of(findings) == set(KINDS), (
        "a kind that never fires is a kind that measures nothing: "
        f"missing {set(KINDS) - kinds_of(findings)}"
    )


def test_every_kind_the_fixture_promises_is_named_in_the_taxonomy(case):
    assert set(case["expected_kinds"]) == set(KINDS)


def test_the_faithful_summary_produces_nothing(case):
    findings, _waved, _derived = check(case["faithful"], case["plan"], case["results"])
    assert findings == [], [f"{f.kind}: {f.claim} — {f.sentence}" for f in findings]


def test_the_fabricated_figure_is_the_one_that_is_not_there(case):
    findings, _waved, _derived = check(case["corrupted"], case["plan"], case["results"])
    fabricated = [f.claim for f in findings if f.kind == "fabricated_figure"]
    assert "1412" in fabricated


def test_the_misattribution_names_the_line_the_figure_belongs_to(case):
    findings, _waved, _derived = check(case["corrupted"], case["plan"], case["results"])
    swapped = [f for f in findings if f.kind == "misattributed_figure"]
    assert swapped, "L2's open count printed under L1 went unnoticed"
    assert any("L2" in f.evidence for f in swapped)


def test_a_figure_the_plan_stated_is_not_called_a_fabrication(case):
    """`SYNTHESIS_PROMPT` orders the assumptions repeated. Scoring the synthesis
    node for the planner's numbers would measure the wrong node."""
    prose = "Working to a 7-day window, L1 recorded 966 defects."
    _findings, waved, _derived = check(prose, case["plan"], case["results"])
    assert waved == 0  # 7 is in the payload's own `days`
    plan = {"interpretation": "", "assumptions": ["A 24-hour window was used."],
            "calls": []}
    findings, waved, _derived = check("A 24-hour window was used.", plan, case["results"])
    assert waved == 1
    assert not [f for f in findings if f.kind == "fabricated_figure"]


def test_a_figure_quoted_out_of_a_passage_is_not_a_misattribution(case):
    """A sentence repeating "Class 3 product" beside a machine name is quoting a
    document, not attributing a figure to that machine."""
    grounding = Grounding(case["results"])
    assert Decimal("1.1291") in grounding.everything
    prose = "For open, the criteria state that continuity is binary."
    findings, _waved, _derived = check(prose, case["plan"], case["results"])
    assert not [f for f in findings if f.kind == "misattributed_figure"]


def test_the_checked_kinds_need_no_judgement(case):
    """The split the report leads with. If a judged kind creeps into
    `CHECKED_KINDS` the published "checkable" fraction silently inflates."""
    assert set(CHECKED_KINDS) == {"fabricated_figure", "misattributed_figure"}
    assert set(CHECKED_KINDS) < set(KINDS)


def test_a_failed_tool_is_a_gap_whether_or_not_the_prose_admits_it(case):
    assert any("query_machine_stats" in gap for gap in gaps(case["results"]))
    findings, _waved, _derived = check(case["faithful"], case["plan"], case["results"])
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
    findings, _waved, _derived = check(prose, case["plan"], case["results"])
    assert not [f for f in findings if f.kind in CHECKED_KINDS]


def test_a_wrong_characterisation_over_right_figures_is_outside_the_taxonomy(case):
    """The boundary the published section states, asserted rather than claimed.

    L2 is the worse line at 7.39 against L1's 7.05. Every figure in this
    sentence is real and attached to the right line; the reading of them is
    wrong, and no kind here covers a reading. If this ever starts failing the
    report's stated limit has moved and the section has to say so.
    """
    prose = "Line L1 is the worse line, at 7.05 defects per board."
    findings, _waved, _derived = check(prose, case["plan"], case["results"])
    assert not [f for f in findings if f.kind in CHECKED_KINDS]


def test_a_count_before_its_noun_belongs_to_the_noun_after_it(case):
    """`19 copper, 22 mousebite` and `copper 19, mousebite 22` say the same
    thing. Reading only backwards shifted a whole list by one and reported
    every entry as a swap -- four findings on one correct sentence."""
    for prose in (
        "Line L1: 138 copper, 188 mousebite, 175 open.",
        "Line L1: copper 138, mousebite 188, open 175.",
    ):
        findings, _waved, _derived = check(prose, case["plan"], case["results"])
        assert not [f for f in findings if f.kind in CHECKED_KINDS], prose


def test_the_swap_is_caught_in_either_word_order(case):
    """The other half of the same fix: making the checker quiet on correct
    prose must not make it quiet on L2's counts printed under L1."""
    for prose in (
        "Line L1 recorded 302 open defects.",
        "Line L1 recorded open at 302.",
    ):
        findings, _waved, _derived = check(prose, case["plan"], case["results"])
        assert [f for f in findings if f.kind == "misattributed_figure"], prose


def test_a_machine_name_is_not_a_figure(case):
    """`M12` put a bare 12 into the figure list, and the entity check found it
    `belonging to` M12. Twenty-two of the first run's thirty-seven attribution
    findings were this and nothing else."""
    prose = "M11, M12, M21, M22, M31 and M32 were all compared."
    findings, _waved, _derived = check(prose, case["plan"], case["results"])
    assert not [f for f in findings if f.kind in CHECKED_KINDS]


def test_a_share_the_model_divided_out_is_arithmetic_not_invention(case):
    """175 of 966 is 18.1%. The payload does not store it and the model is not
    forbidden to divide -- but the division is checked, not assumed."""
    findings, _waved, derived = check(
        "Open accounts for 18.1% of L1's defects.", case["plan"], case["results"]
    )
    assert not [f for f in findings if f.kind == "fabricated_figure"]
    assert derived
    findings, _waved, derived = check(
        "Open accounts for 41.3% of L1's defects.", case["plan"], case["results"]
    )
    assert [f for f in findings if f.kind == "fabricated_figure"]


def test_a_fleet_figure_beside_a_machine_name_is_not_a_swap(case):
    """"M11 is below the fleet average of 0.95" names a machine and then
    quotes a figure that is deliberately no machine's."""
    grounding = Grounding(case["results"])
    assert grounding.family_free.get("machine")


def test_a_ranked_list_with_one_entry_swapped_is_still_caught(case):
    """The strongest guard against tuning until nothing fails.

    Five corrections were made after adjudicating a real run, each traced to a
    checker defect rather than to a number somebody wanted lower. Every one of
    them made the checker quieter, so every one of them needs a case proving it
    did not make it blind: this is the machine-ranking shape the corrections
    were made *on*, correct except that L2's open count sits in L1's row.
    """
    right = "L1 recorded 966 defects; L2 recorded 1,049; open on L1 was 175."
    findings, _waved, _derived = check(right, case["plan"], case["results"])
    assert not [f for f in findings if f.kind in CHECKED_KINDS], right

    swapped = "L1 recorded 966 defects; L2 recorded 1,049; open on L1 was 302."
    findings, _waved, _derived = check(swapped, case["plan"], case["results"])
    assert [f for f in findings if f.kind == "misattributed_figure"], swapped


def test_a_figure_that_is_simply_not_there_is_still_caught(case):
    """The ratio excuse must not become a licence. 1412 is not a quotient of
    any two figures in this payload, and it must stay a fabrication."""
    findings, _waved, _derived = check(
        "L1 recorded 1,412 defects.", case["plan"], case["results"]
    )
    assert [f for f in findings if f.kind == "fabricated_figure"]


def test_limited_to_one_class_is_not_a_rule_about_that_class(case):
    """`limit` inside `limited to copper` raised a criterion finding against a
    sentence that states no rule at all. It is the one false positive the
    published run carries, and it is fixed here rather than re-run away."""
    findings, _waved, _derived = check(
        "No other defect class comparisons are provided, so the analysis is "
        "limited to copper.",
        {"calls": []},
        [{"tool": "query_defect_history", "args": {}, "ok": True,
          "data": {"by_class": {"copper": 3}}, "error": None}],
    )
    assert not [f for f in findings if f.kind == "misquoted_criterion"]


def test_a_real_rule_about_an_unretrieved_class_still_fires(case):
    """The other side of it: the incident this kind exists for."""
    findings, _waved, _derived = check(
        "A short of this size must be rejected under the criteria.",
        case["plan"], case["results"],
    )
    assert [f for f in findings if f.kind == "misquoted_criterion"]


# ---------------------------------------------------------------------------
# A list of names against a list of figures
# ---------------------------------------------------------------------------

MACHINE_STATS = {
    "tool": "query_machine_stats", "args": {}, "ok": True, "error": None,
    "elapsed_ms": 1.0,
    "data": {
        "defect_type": "mousebite", "days": 2,
        "fleet_share_of_defects": 0.189, "fleet_average_per_board": 1.326,
        "machines": [
            {"machine": "L1-M12", "defects": 29, "per_board": 1.45,
             "share_of_defects": 0.199},
            {"machine": "L2-M21", "defects": 21, "per_board": 1.5,
             "share_of_defects": 0.198},
            {"machine": "L2-M22", "defects": 37, "per_board": 1.276,
             "share_of_defects": 0.178},
        ],
    },
}

DISTRIBUTED = (
    "在各機台中，L1-M12、L2-M21、L2-M22 分別產生 29、21、37 個，"
    "平均每板 1.45、1.5、1.276 個，佔缺陷的 19.9%、19.8%、17.8%。"
)


def misattributions(prose: str) -> list:
    findings, _waved, _derived = check(prose, {"assumptions": []}, [MACHINE_STATS])
    return [f for f in findings if f.kind == "misattributed_figure"]


def test_a_list_of_names_against_a_list_of_figures_pairs_by_position():
    """`A、B、C 分別產生 X、Y、Z` is one sentence in which every figure's
    nearest preceding name is the last one.

    The checker gave all nine figures here to `L2-M22` and flagged every entry
    after the first -- eleven findings on one real sentence, none of them the
    model's fault, with the payload matching the prose six machines deep.
    """
    assert misattributions(DISTRIBUTED) == []


def test_a_transposed_list_is_still_caught_in_exactly_that_shape():
    """The other side of the change, and the reason it is not just a mute.

    Before it, a faithful list and a swapped one both produced roughly eleven
    findings -- so the flags carried no information about which was which. The
    pairing is what makes a real swap legible.
    """
    swapped = DISTRIBUTED.replace("分別產生 29、21、37", "分別產生 21、29、37")
    found = misattributions(swapped)

    assert len(found) == 2
    claims = sorted(f.claim for f in found)
    assert claims == ["21 attributed to L1/M12", "29 attributed to L2/M21"]


def test_the_english_shape_of_the_same_facts_was_never_affected():
    """Why one language saw this and the other did not: English pairs each name
    with its own figure, so the nearest-preceding rule was always right there.
    This is the construction that stayed clean, kept as the control."""
    english = (
        "L1-M12 produced 29 mousebites at 1.45 per board, "
        "L2-M21 produced 21 at 1.5 per board, "
        "and L2-M22 produced 37 at 1.276 per board."
    )

    assert misattributions(english) == []


def test_two_names_with_no_separator_between_them_are_not_a_list():
    """The join has to be punctuation. `L1-M12 produced 29 and L2-M21 21` is a
    sentence, not a parallel list, and reading it as one would pair figures
    across a verb."""
    from aoi_agent.analysis.claims import _LIST_JOIN

    assert _LIST_JOIN.match("、")
    assert _LIST_JOIN.match("%、")
    assert _LIST_JOIN.match("個、")
    assert not _LIST_JOIN.match(" ")
    assert not _LIST_JOIN.match(" 產生 ")
    assert not _LIST_JOIN.match(" 個，平均每板 ")


BY_CLASS = {
    "tool": "query_defect_history", "args": {}, "ok": True, "error": None,
    "elapsed_ms": 1.0,
    "data": {
        "filters": {"line_id": None, "days": 7},
        "boards_inspected": 421,
        "by_class": {"mousebite": 537, "spur": 438},
    },
}


def by_class_misattributions(prose: str) -> list:
    findings, _w, _d = check(prose, {"assumptions": []}, [BY_CLASS])
    return [f for f in findings if f.kind == "misattributed_figure"]


def test_a_measure_word_between_a_count_and_its_noun_does_not_break_the_pairing():
    """Chinese counts through a measure word -- `438 個 spur` -- where English
    writes `438 spur` and needs nothing.

    Requiring bare whitespace between a figure and the noun that follows it
    therefore rejected every Chinese count-plus-noun and handed each one to
    whatever was named before it. The English half of this very sentence was
    always right, which is why one payload written up twice is what found it.
    """
    assert by_class_misattributions(
        "發現 537 個 mousebite 缺陷，以及 438 個 spur 缺陷。"
    ) == []
    assert by_class_misattributions(
        "There were 537 mousebite defects and 438 spur defects."
    ) == []


def test_the_counts_swapped_across_the_measure_word_are_still_caught():
    found = by_class_misattributions(
        "發現 438 個 mousebite 缺陷，以及 537 個 spur 缺陷。"
    )

    assert {f.claim for f in found} == {
        "438 attributed to mousebite", "537 attributed to spur",
    }


def test_punctuation_between_a_figure_and_a_name_still_ends_the_pairing():
    """The guard this widened is against `M21(1.933/板, 25%)` -- a share after
    one name, a bracket in between, read as the next machine's and shifting a
    whole ranked list. A measure word is not punctuation; a bracket is."""
    from aoi_agent.analysis.claims import _MEASURE_WORD

    assert _MEASURE_WORD.match(" ")
    assert _MEASURE_WORD.match(" 個 ")
    assert not _MEASURE_WORD.match("), ")
    assert not _MEASURE_WORD.match(" 的 ")
    assert not _MEASURE_WORD.match(" 缺陷，以及 ")
