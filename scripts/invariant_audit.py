"""Which invariants in CLAUDE.md would actually fail a test if broken?

CLAUDE.md's "Invariants" section is the load-bearing document in this
repository: it is what a reader is asked to trust and what an agent is asked
not to undo. Trusting it means something only if breaking one of those rules
breaks the build. Audited by hand on 2026-08-22, seven of twelve did not --
including the one naming ``InMemorySaver`` as the thing that must never be the
checkpointer, which was verified by a suite that used nothing else. That audit
was a scratch file, it was stale within a day, and nobody would ever have run
it again.

So it is a script now, and the script is honest about what it can and cannot
decide.

**What it cannot do.** "Is this invariant enforced?" is not decidable by
reading source. Nothing in a test's name, its assertions or its imports tells
you which English sentence it holds up, and a test can assert loudly about the
wrong thing.

**What it does instead.** Every invariant gets a declared entry below, naming
the test or tests that enforce it and saying what those tests do *not* cover.
Writing the entry is a human judgement, made once, ideally with a mutation to
back it (see docs/benchmarks.md, "The invariant audit"). The script then holds
that judgement to the code:

- an invariant in CLAUDE.md with no entry here          -> drift
- an entry here matching no invariant in CLAUDE.md      -> drift
- a named test whose file is gone                       -> drift
- a named test that has been renamed or deleted         -> drift
- a named test wearing ``@pytest.mark.skip``            -> drift
- an ``enforced`` entry every one of whose tests is
  ``@pytest.mark.dataset``, so CI never runs it         -> drift
- ``--collect``: a named test pytest does not collect   -> drift

That is the whole of the mechanism. It cannot tell you an entry's *claim* is
true; it can tell you the claim has stopped being checkable, which is the way
this document actually rots.

Statuses are declared, not inferred:

``enforced``       breaking it fails a named test that runs in CI.
``partial``        some of it is held; the entry says which part is not.
``unenforceable``  prose discipline, not behaviour. Names no test, and must
                   say why, so that "no test" reads as a decision rather than
                   an oversight.

Parsing is static (``ast``, plus the Markdown), the same as
``check_skill_freshness.py``: running this imports no package and touches no
GPU. ``--collect`` is the one exception and shells out to pytest.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAUDE_MD = ROOT / "CLAUDE.md"
INVARIANTS_HEADING = "## Invariants"

ENFORCED = "enforced"
PARTIAL = "partial"
UNENFORCEABLE = "unenforceable"
STATUSES = (ENFORCED, PARTIAL, UNENFORCEABLE)


@dataclass(frozen=True)
class Entry:
    """One invariant, and the tests claimed to enforce it."""

    #: A phrase that must appear in exactly one invariant bullet. Matching on a
    #: phrase rather than the whole bullet means the prose can be edited without
    #: breaking the audit, while rewriting the *rule* still does.
    match: str
    status: str
    #: ``tests/file.py::test_name``. The tests that fail when it is broken.
    tests: tuple[str, ...] = ()
    #: What those tests hold. For ``partial``, what they do not.
    note: str = ""
    #: For ``partial`` and ``unenforceable``: what is still unguarded, in the
    #: terms someone would have to write a test in.
    gap: str = ""
    #: A mutation that was applied to the tree and the tests that noticed, when
    #: the entry was written. Evidence, not a check -- the script does not
    #: re-run it.
    proved_by: str = ""


#: The registry. Adding an invariant to CLAUDE.md without adding a row here is
#: a failure of this script, which is the point: the audit cannot fall behind
#: the document silently.
REGISTRY: tuple[Entry, ...] = (
    Entry(
        match="Report an operating-point curve",
        status=ENFORCED,
        tests=(
            "tests/test_operating_point.py::test_escape_budget_is_respected",
            "tests/test_operating_point.py::test_review_reduction_rises_as_the_threshold_falls",
            "tests/test_operating_point.py::test_unreachable_budget_returns_none",
            "tests/test_operating_point.py::test_system_escape_rate_compounds_both_stages",
            "tests/test_report_curve.py::test_the_report_sweeps_more_than_one_escape_budget",
            "tests/test_report_curve.py::test_every_budget_is_published_with_what_it_costs_and_what_it_buys",
            "tests/test_report_curve.py::test_the_published_points_are_a_trade_off_and_not_one_point_repeated",
            "tests/test_report_curve.py::test_accuracy_is_not_the_headline",
        ),
        note=(
            "Both the arithmetic and the publication. The sweep, the escape "
            "budget and the two-stage compounding fail when the arithmetic "
            "moves; the report itself is now generated in the suite over a "
            "synthetic split with a real trade-off in it, and held to a row per "
            "budget carrying the escape rate achieved and the review removed, "
            "with accuracy after the curve and unemphasised. Values are compared "
            "numerically against `sweep` and columns read by header, so "
            "rewording the report is free."
        ),
        proved_by=(
            "BUDGETS = [] in scripts/report.py: 691 passed before "
            "tests/test_report_curve.py existed, 3 failed after. BUDGETS = "
            "[0.005]: 3 failed. The accuracy line moved above the table and "
            "bolded: 1 failed."
        ),
    ),
    Entry(
        match="The LLM explains; it does not decide",
        status=ENFORCED,
        tests=(
            "tests/test_graph.py::test_the_classifier_class_stands_when_the_llm_disagrees",
            "tests/test_graph.py::test_the_llm_saying_it_is_confident_does_not_rescue_a_weak_classification",
            "tests/test_graph.py::test_the_agent_branch_cannot_dismiss",
            "tests/test_analysis_plan.py::test_a_legal_looking_value_outside_its_domain_is_rejected",
            "tests/test_analysis_charts.py::test_a_defect_breakdown_becomes_bars",
        ),
        note=(
            "Both ends of the rule. On the disposition path the verdict and the "
            "hand-off are taken off the model; on the analysis path the plan is "
            "validated before it runs and the chart comes from the result shape."
        ),
        proved_by=(
            "decide_node reading agent_verdict: 1 failed. "
            "route_after_reason reading agent_confident: 5 failed."
        ),
    ),
    Entry(
        match="every failure escalates to a human",
        status=ENFORCED,
        tests=(
            "tests/test_graph.py::test_an_uncertain_case_gathers_evidence_and_goes_to_a_person",
            "tests/test_graph.py::test_an_unconfident_agent_interrupts_for_a_person",
            "tests/test_graph.py::test_an_unparseable_verdict_reaches_a_person_at_a_confidence_of_0_55",
            "tests/test_graph.py::test_an_unparseable_verdict_at_0_93_is_decided_anyway",
            "tests/test_graph.py::test_an_unreachable_model_no_longer_forces_an_escalation",
            "tests/test_graph.py::test_an_unreachable_model_at_0_55_reaches_a_person_rather_than_crashing",
            "tests/test_graph.py::test_a_confident_open_still_gathers_evidence",
        ),
        note=(
            "Both directions, which matters here more than usual: the rule has a "
            "carve-out, so tests that only prove escalation would pass a flow "
            "that escalated everything."
        ),
        proved_by="route_after_reason returning 'decide' unconditionally: 23 failed",
    ),
    Entry(
        match="an escalation must outlive the process",
        status=ENFORCED,
        tests=(
            "tests/test_checkpoint_durability.py::test_an_escalation_survives_the_process_that_raised_it",
            "tests/test_checkpoint_durability.py::test_the_default_checkpointer_is_a_file_and_not_a_dictionary",
        ),
        note=(
            "This is the invariant the 2026-08-22 audit found hollow: every test "
            "in the suite passed an InMemorySaver, so nothing could fail when it "
            "became the default. The first test now raises the escalation in an "
            "interpreter that exits before anything answers it."
        ),
        proved_by=(
            "make_checkpointer returning InMemorySaver: 3 failed, including the "
            "cross-process one"
        ),
    ),
    Entry(
        match="never shows `ground_truth`",
        status=ENFORCED,
        tests=(
            "tests/test_station.py::test_the_ground_truth_never_leaves_the_store",
            "tests/test_station.py::test_the_corrections_page_does_not_serve_the_ground_truth",
            "tests/test_analysis_page.py::test_no_payload_shape_puts_the_answer_key_on_the_page",
            "tests/test_analysis_page.py::test_the_synthesis_prompt_is_filtered_by_the_same_rule_as_the_table",
            "tests/test_analysis_page.py::test_every_probed_position_is_one_the_page_accounts_for",
        ),
        note=(
            "Held at the dict boundary on both routes out, and the analysis side "
            "enumerates the positions a value can occupy rather than trusting a "
            "top-level key check."
        ),
        proved_by="ground_truth added to store.boards._as_dict: 1 failed",
    ),
    Entry(
        match="No free-form text-to-SQL",
        status=ENFORCED,
        tests=(
            "tests/test_analysis_plan.py::test_an_unknown_tool_is_rejected",
            "tests/test_analysis_plan.py::test_an_unknown_argument_name_is_rejected",
            "tests/test_analysis_plan.py::test_a_legal_looking_value_outside_its_domain_is_rejected",
            "tests/test_analysis_plan.py::test_classify_defect_is_not_plannable",
            "tests/test_analysis_plan.py::test_this_files_domains_carry_every_key_the_validator_looks_up",
            "tests/test_analysis_registry.py::test_the_registry_as_it_stands_accounts_for_every_parameter_it_exposes",
            "tests/test_analysis_registry.py::test_a_tool_taking_free_text_query_language_cannot_be_registered",
            "tests/test_analysis_registry.py::test_calling_the_query_language_an_identifier_does_not_get_it_registered",
            "tests/test_analysis_registry.py::test_free_text_declared_on_a_module_that_can_reach_the_store_is_refused",
            "tests/test_analysis_registry.py::test_search_standards_still_registers_with_its_free_text_intact",
        ),
        note=(
            "Both the call and the surface. The validator is held hard against "
            "the registry that exists -- unknown tool, unknown argument, "
            "out-of-domain value all refused before anything runs -- and the "
            "registry itself now has to account for every parameter it exposes, "
            "at import, or the module does not load. A signature cannot tell a "
            "retrieval query from a query language, so the account is declared "
            "per parameter and backed by what can be checked: the corpus has to "
            "exist, the tool's module and the corpus module are checked "
            "statically for a route to the store, and the tool's body for SQL "
            "built out of a string. The last test is the control -- "
            "search_standards takes arbitrary prose and must go on registering."
        ),
        proved_by=(
            "run_query(sql: str) added to production.py and to the registry, "
            "three ways: undeclared, declared an identifier, and declared a "
            "retrieval query over the standards corpus. 691 passed before this "
            "gate existed; each of the three now stops the module importing, on "
            "the unaccounted parameter, on text(), and on the module's route to "
            "the store respectively."
        ),
    ),
    Entry(
        match="The fan-out is the shape of the work",
        status=PARTIAL,
        tests=(
            "tests/test_analysis_graph.py::test_the_wall_time_covers_the_scheduling_and_not_only_the_branches",
            "tests/test_analysis_graph.py::test_timings_are_recorded_per_tool_and_for_the_phases",
            "tests/test_analysis_service.py::test_the_run_records_what_the_fan_out_saved",
            "tests/test_analysis_page.py::test_the_page_shows_what_the_fan_out_cost_and_saved",
        ),
        note=(
            "The comparison the claim rests on is measured, stored per run and "
            "rendered, and `tools_wall` is held to cover the scheduling rather "
            "than just the slowest branch -- which is what would let a saving be "
            "overstated."
        ),
        gap=(
            "The prohibition is on prose, and no test reads prose. Rewriting the "
            "graph module's own docstring to call the fan-out a four-times "
            "speed-up leaves the suite green, and so would the same edit to "
            "README.md or the page."
        ),
        proved_by=(
            "analysis/graph.py docstring changed to 'is a latency optimisation': "
            "691 passed"
        ),
    ),
    Entry(
        match="come from that class's document",
        status=ENFORCED,
        tests=(
            "tests/test_standards_retrieval.py::test_no_phrasing_returns_another_classs_criteria",
            "tests/test_standards_retrieval.py::test_a_class_scopes_to_its_own_document_and_the_policy_documents",
            "tests/test_standards_retrieval.py::test_every_document_declares_the_class_it_governs",
            "tests/test_standards_retrieval.py::test_the_measurement_still_detects_the_fault_it_was_written_for",
            "tests/test_graph.py::test_the_criteria_are_retrieved_scoped_to_the_class_in_hand",
        ),
        note=(
            "Both halves: the retrieval cannot cross a class, and the disposition "
            "path is held to passing the scope. The fourth test re-runs the "
            "original fault against an unscoped search, so a guard that stops "
            "guarding anything fails rather than passing vacuously."
        ),
        proved_by="defect_class dropped from the flow's search_standards call: 2 failed",
    ),
    Entry(
        match="Thresholds come from the sweep or the work instructions",
        status=ENFORCED,
        tests=(
            "tests/test_threshold_citations.py::test_no_threshold_reaches_the_code_without_a_row",
            "tests/test_threshold_citations.py::test_every_row_states_the_value_the_code_carries",
            "tests/test_threshold_citations.py::test_a_cited_work_instruction_states_the_number",
            "tests/test_threshold_citations.py::test_the_cost_gate_never_drops_below_the_escalation_threshold",
            "tests/test_response_budget.py::test_the_budget_matches_what_the_work_instruction_says",
        ),
        note=(
            "A value that drifts from its source fails, and so does a new "
            "threshold constant that reaches the code with no row in the table."
        ),
        proved_by="ESCALATE_BELOW set to 0.80: 3 failed",
    ),
    Entry(
        match="Use the official DeepPCB split",
        status=PARTIAL,
        tests=(
            "tests/test_deeppcb.py::test_unknown_split_is_rejected",
            "tests/test_deeppcb.py::test_official_splits_have_the_documented_sizes",
        ),
        note=(
            "A split name the dataset does not have is refused, and the two "
            "official sizes are asserted."
        ),
        gap=(
            "The only test that would catch a re-split is dataset-marked, so CI "
            "-- which runs `-m \"not dataset\"` -- never runs it, and the size "
            "check only catches a re-split that changes a count. A "
            "size-preserving reshuffle of the same 1500 boards passes both."
        ),
        proved_by=(
            "load_split reading trainval.txt for both splits: 3 failed, all three "
            "dataset-marked"
        ),
    ),
    Entry(
        match="never by patch",
        status=ENFORCED,
        tests=(
            "tests/test_train_split.py::test_no_image_appears_on_both_sides_of_the_split",
            "tests/test_train_split.py::test_a_patch_level_split_is_what_this_test_would_catch",
            "tests/test_train_split.py::test_every_patch_lands_on_exactly_one_side",
            "tests/test_train_split.py::test_the_holdout_is_about_the_size_that_was_asked_for",
        ),
        note=(
            "Had no test at all until 2026-08-23, on a project where every "
            "published number is read off this split. The second test replays the "
            "leaking split against the same assertion, so the guard cannot pass "
            "by being vacuous."
        ),
        proved_by=(
            "split_by_image shuffling patch indices instead of images: 691 passed "
            "before this file existed, 5 failed after"
        ),
    ),
    Entry(
        match="An automated disposition names what produced it",
        status=ENFORCED,
        tests=(
            "tests/test_provenance.py::test_an_automated_decision_cannot_be_written_without_provenance",
            "tests/test_provenance.py::test_a_statement_of_absence_is_not_provenance",
            "tests/test_provenance.py::test_an_automated_decision_carries_the_weights_the_thresholds_and_the_code",
            "tests/test_provenance.py::test_a_row_written_before_the_columns_is_not_a_row_with_no_value",
        ),
        note=(
            "The write path is guarded, not just the schema: an automated row "
            "with no attributable provenance raises. The migration's stamp is "
            "held too, so `unrecorded` cannot silently become `NULL`."
        ),
        gap=(
            "Nothing checks that the digest on a row is the digest of the "
            "checkpoint that actually ran -- a stubbed tool can hand out any "
            "string, and the suite does exactly that."
        ),
        proved_by=(
            "Removed the guard in record_decision: 5 tests failed. Removed the "
            "migration backfill: 2 failed."
        ),
    ),
    Entry(
        match="Language is a rendering, not a record",
        status=ENFORCED,
        tests=(
            "tests/test_i18n.py::test_the_question_is_never_rewritten_by_the_switch",
            "tests/test_i18n.py::test_the_plan_sections_keep_the_language_they_were_written_in",
            "tests/test_i18n.py::test_a_section_the_switch_does_not_touch_says_so",
            "tests/test_analysis_run_languages.py::test_a_language_already_present_is_never_overwritten",
            "tests/test_analysis_run_languages.py::test_the_old_answer_is_not_filed_under_a_guessed_language",
        ),
        note=(
            "Both halves are held. What the planning call wrote -- the "
            "question, the interpretation, the assumptions -- is asserted "
            "unchanged when the page is read in the other language, and the "
            "badge that says so is asserted present only when the languages "
            "differ. On the store side, a language already answered is never "
            "rewritten and the migration does not file an old answer under a "
            "guessed key."
        ),
        gap=(
            "Nothing asserts that a re-synthesis quotes the same figures as "
            "the first pass: that comparison needs a real model and lives in "
            "`scripts/synthesis_eval.py --lang both`, which the suite does not "
            "run. What the suite holds is that the second language is written "
            "from the stored results rather than translated -- not that the "
            "two accounts agree."
        ),
        proved_by=(
            "Let the switch re-render the plan's prose: 2 tests failed. Let "
            "`add_answer` overwrite an existing language: 1 failed. Backfilled "
            "`answers_json` under a guessed key: 2 failed."
        ),
    ),
    Entry(
        match="Say what the prevalence is, and what it costs",
        status=ENFORCED,
        tests=(
            "tests/test_prevalence.py::test_the_escape_rate_is_exactly_invariant_under_reweighting",
            "tests/test_prevalence.py::test_review_reduction_rises_as_the_line_gets_cleaner",
            "tests/test_prevalence.py::test_the_reweighting_agrees_with_the_measurement_at_its_own_prevalence",
            "tests/test_prevalence.py::test_the_projects_own_escape_figure_does_not_exclude_exceeding_its_budget",
        ),
        note=(
            "The three claims are held separately: that the escape rate cannot "
            "move with prevalence, that review reduction must, and that the "
            "published escape figure's interval crosses its own budget. The "
            "re-weighting is also checked against the swept figure at the "
            "dataset's own prevalence, which is what stops it drifting into a "
            "different quantity."
        ),
        gap=(
            "Everything here assumes label shift -- that the score "
            "distributions within each group are what a line would produce. "
            "The dataset is binarised, pre-registered and partly augmented, "
            "which is precisely the reason to doubt that, and no test can "
            "check it without a second dataset. What is enforced is the "
            "separation, not the transfer."
        ),
        proved_by=(
            "Divided the escape rate by the candidate total instead of the "
            "defect total: 1 test failed. Made the re-weighting ignore its "
            "prevalence argument: 1 failed -- and did not, until the ordering "
            "assertion was tightened from `== sorted(...)`, which a constant "
            "list satisfies."
        ),
    ),
    Entry(
        match="A human disposition names who made it",
        status=ENFORCED,
        tests=(
            "tests/test_attribution.py::test_a_human_decision_cannot_be_written_without_an_identity",
            "tests/test_attribution.py::test_a_name_with_nothing_behind_it_is_not_an_identity",
            "tests/test_attribution.py::test_an_unattributable_answer_does_not_consume_the_escalation",
            "tests/test_attribution.py::test_the_recorded_reviewer_is_the_session_and_not_the_form",
            "tests/test_attribution.py::test_no_page_answers_a_visitor_who_has_not_signed_in",
            "tests/test_attribution.py::test_a_verdict_from_nobody_is_refused_and_writes_nothing",
            "tests/test_attribution.py::test_a_row_written_before_the_column_is_not_a_row_that_names_nobody",
        ),
        note=(
            "Three mechanisms, each held: the store refuses a `human` row that "
            "is not attributable, the station takes the name off the session "
            "and ignores one posted with the form, and no page answers a "
            "request that carries no session. The migration's stamp is held "
            "too, so a decision written today saying `automated` cannot be "
            "confused with one of the 9,140 that predate the column. What the "
            "tests hold is that a name is *established by a mechanism* rather "
            "than typed -- not that the name is true of the person at the "
            "keyboard, which no scheme short of a badge reader gives you and "
            "which src/aoi_agent/station/auth.py says out loud."
        ),
        proved_by=(
            "Removed the guard in record_decision and defaulted the identity to "
            "a typed name: 7 tests failed. Put the `reviewer` form field back "
            "and read the name off it: 4 failed. Let the middleware pass an "
            "anonymous request through as `operator`: 13 failed. Removed the "
            "migration's `reviewer_auth` backfill: 3 failed."
        ),
    ),
    Entry(
        match="Say what is simulated",
        status=UNENFORCEABLE,
        note=(
            "Prose discipline. The rule is that a reader is told which figures "
            "come from generated metadata, and no assertion distinguishes an "
            "honest sentence from a missing one."
        ),
        gap=(
            "The nearest testable shadow of it is the real/simulated table in "
            "docs/architecture.md -- a test could hold that table against the "
            "modules it names. That would check the table exists, not that the "
            "prose around it is candid, so it is not claimed here as enforcement."
        ),
    ),
)


# ---------------------------------------------------------------------------
# CLAUDE.md
# ---------------------------------------------------------------------------


def invariant_bullets(text: str | None = None) -> list[str]:
    """The top-level bullets under CLAUDE.md's Invariants heading.

    Continuation lines are folded in, so a `match` phrase may sit anywhere in
    the bullet rather than only on its first line.
    """
    text = CLAUDE_MD.read_text() if text is None else text
    lines = text.splitlines()

    start = next(
        (i for i, line in enumerate(lines) if line.startswith(INVARIANTS_HEADING)), None
    )
    if start is None:
        raise SystemExit(
            f"{CLAUDE_MD.name}: no '{INVARIANTS_HEADING}' heading. "
            "The audit reads that section; renaming it silently empties this script."
        )
    end = next(
        (i for i, line in enumerate(lines[start + 1 :], start + 1) if line.startswith("## ")),
        len(lines),
    )

    bullets: list[str] = []
    for line in lines[start + 1 : end]:
        if line.startswith("- "):
            bullets.append(line[2:].strip())
        elif bullets and line.startswith("  ") and line.strip():
            bullets[-1] += " " + line.strip()
    return bullets


def title_of(bullet: str) -> str:
    """A short name for a bullet: its bold lead, or its first clause."""
    bold = re.match(r"\*\*(.+?)\*\*", bullet)
    if bold:
        return bold.group(1).rstrip(".")
    plain = re.split(r" -- |\. ", bullet, maxsplit=1)[0]
    return re.sub(r"\*\*", "", plain).rstrip(".")


# ---------------------------------------------------------------------------
# Tests, read statically
# ---------------------------------------------------------------------------


@dataclass
class TestFacts:
    exists: bool = False
    skipped: bool = False
    dataset: bool = False
    reason: str = ""


def _marks(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        parts: list[str] = []
        while isinstance(target, ast.Attribute):
            parts.append(target.attr)
            target = target.value
        if isinstance(target, ast.Name):
            parts.append(target.id)
        dotted = ".".join(reversed(parts))
        if dotted.startswith("pytest.mark."):
            names.add(dotted.split(".", 2)[2])
    return names


def inspect_test(node_id: str) -> TestFacts:
    """What the source says about one ``tests/file.py::test_name``."""
    relpath, _, name = node_id.partition("::")
    path = ROOT / relpath
    if not path.exists():
        return TestFacts(reason=f"{relpath} does not exist")

    tree = ast.parse(path.read_text(), filename=relpath)
    node = next(
        (
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        ),
        None,
    )
    if node is None:
        return TestFacts(reason=f"{relpath} has no {name}")

    marks = _marks(node)
    return TestFacts(
        exists=True,
        skipped=bool(marks & {"skip", "skipif"}),
        dataset="dataset" in marks,
        reason="wears @pytest.mark.skip" if marks & {"skip", "skipif"} else "",
    )


def collected_node_ids() -> set[str]:
    """What pytest actually collects. Shells out; only used by ``--collect``."""
    finished = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if finished.returncode != 0:
        raise SystemExit(
            "pytest could not collect the suite, so --collect can decide nothing:\n"
            + finished.stdout[-2000:]
        )
    # Parametrised ids carry a `[...]` suffix; the registry names base functions.
    return {line.split("[", 1)[0].strip() for line in finished.stdout.splitlines() if "::" in line}


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------


@dataclass
class Row:
    entry: Entry
    title: str
    problems: list[str] = field(default_factory=list)
    dataset_only: bool = False


def audit(
    collect: bool = False,
    registry: tuple[Entry, ...] = REGISTRY,
    bullets: list[str] | None = None,
) -> tuple[list[Row], list[str]]:
    """Returns one row per invariant, plus drift that belongs to no single row.

    ``registry`` and ``bullets`` are injectable so the suite can hand this a
    document and a registry that disagree, and check that it says so. A checker
    that has never been shown a failure is not known to have one.
    """
    bullets = invariant_bullets() if bullets is None else bullets
    loose: list[str] = []
    rows: list[Row] = []
    claimed: dict[int, Entry] = {}

    for entry in registry:
        if entry.status not in STATUSES:
            loose.append(f"{entry.match!r}: unknown status {entry.status!r}")

        hits = [i for i, bullet in enumerate(bullets) if entry.match in bullet]
        if not hits:
            loose.append(
                f"{entry.match!r} matches no invariant in CLAUDE.md -- the rule was "
                "removed or reworded, and this entry has to follow it"
            )
            continue
        if len(hits) > 1:
            loose.append(
                f"{entry.match!r} matches {len(hits)} invariants; the phrase has to "
                "pick out exactly one"
            )
        index = hits[0]
        if index in claimed:
            loose.append(
                f"two entries claim the same invariant: {claimed[index].match!r} "
                f"and {entry.match!r}"
            )
        claimed[index] = entry

        row = Row(entry=entry, title=title_of(bullets[index]))
        rows.append(row)

        if entry.status == UNENFORCEABLE:
            if entry.tests:
                row.problems.append(
                    "declared unenforceable but names tests; pick one or the other"
                )
            if not entry.gap and not entry.note:
                row.problems.append("declared unenforceable with no reason given")
            continue

        if not entry.tests:
            row.problems.append(
                "names no test. An invariant is either enforced by something or "
                "declared unenforceable with a reason"
            )
        if entry.status == PARTIAL and not entry.gap:
            row.problems.append("declared partial without saying what is unguarded")

        facts = {node_id: inspect_test(node_id) for node_id in entry.tests}
        for node_id, fact in facts.items():
            if not fact.exists:
                row.problems.append(f"{node_id}: {fact.reason}")
            elif fact.skipped:
                row.problems.append(f"{node_id}: {fact.reason}, so it enforces nothing")

        live = [f for f in facts.values() if f.exists and not f.skipped]
        row.dataset_only = bool(live) and all(f.dataset for f in live)
        if row.dataset_only and entry.status == ENFORCED:
            row.problems.append(
                "every test named here is @pytest.mark.dataset, so CI "
                '(`-m "not dataset"`) never runs one of them. That is `partial`, '
                "not `enforced`"
            )

    for index, bullet in enumerate(bullets):
        if index not in claimed:
            loose.append(
                f"CLAUDE.md invariant {title_of(bullet)!r} has no entry in the "
                "registry. Add one, even if the entry is 'unenforceable'"
            )

    if collect:
        collected = collected_node_ids()
        for row in rows:
            for node_id in row.entry.tests:
                if node_id not in collected:
                    row.problems.append(f"{node_id}: pytest does not collect it")

    return rows, loose


MARK = {ENFORCED: "ok    ", PARTIAL: "PART  ", UNENFORCEABLE: "n/a   "}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collect",
        action="store_true",
        help="also check pytest collects each named test (runs pytest --collect-only)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when anything is less than fully enforced, not only on drift",
    )
    args = parser.parse_args()

    rows, loose = audit(collect=args.collect)

    for row in rows:
        flag = "DRIFT " if row.problems else MARK[row.entry.status]
        print(f"{flag} {row.title}")
        for node_id in row.entry.tests:
            print(f"           {node_id}")
        if row.entry.gap:
            print(f"       gap: {row.entry.gap}")
        for problem in row.problems:
            print(f"       !! {problem}")

    counts = {status: sum(1 for r in rows if r.entry.status == status) for status in STATUSES}
    print(
        f"\n{len(rows)} invariants: {counts[ENFORCED]} enforced, "
        f"{counts[PARTIAL]} partly enforced, {counts[UNENFORCEABLE]} unenforceable"
    )

    drift = sum(len(row.problems) for row in rows) + len(loose)
    for line in loose:
        print(f"DRIFT  {line}")
    if drift:
        print(
            f"\n{drift} drift(s). A named test moved, or CLAUDE.md and this "
            "registry stopped agreeing."
        )
        return 1
    if args.strict and counts[PARTIAL]:
        print(f"\n--strict: {counts[PARTIAL]} invariant(s) are not fully enforced.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
