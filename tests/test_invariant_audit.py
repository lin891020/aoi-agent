"""CLAUDE.md's invariants, and whether each still has the guard it claims.

An invariant is a promise that something cannot happen. The promise is worth
what the test behind it is worth, and on 2026-08-22 an outside audit found that
seven of the twelve promises in CLAUDE.md had nothing behind them at all -- the
worst of them being the one that names ``InMemorySaver`` as the thing the
checkpointer must never be, verified at the time by a suite in which every
single test passed an ``InMemorySaver``.

That audit was a scratch file. It was stale inside a day. This is the same
audit as something the repository runs: ``scripts/invariant_audit.py`` holds a
registry of invariant -> tests, and these tests hold the registry to the
document and to the suite.

What that catches is drift, not correctness. Whether
``test_the_ground_truth_never_leaves_the_store`` really enforces the rule it is
listed under is a judgement a person made once, with a mutation to back it (see
docs/benchmarks.md, "The invariant audit"). What is caught here is the way that
judgement rots: the test gets renamed, or skipped, or the invariant is reworded,
or a fourteenth is added with nothing behind it -- and the document goes on
promising.

Same shape as test_skill_freshness.py: make the document and the code disagree
loudly, at the moment the code changes, rather than quietly forever.
"""

from __future__ import annotations

import pytest

from invariant_audit import (
    ENFORCED,
    PARTIAL,
    REGISTRY,
    UNENFORCEABLE,
    Entry,
    audit,
    inspect_test,
    invariant_bullets,
    title_of,
)


def test_every_invariant_still_has_the_enforcement_the_registry_claims():
    """The headline. Fails when a named test is renamed, deleted or skipped,
    when an invariant is added to CLAUDE.md with no entry, or when an entry
    outlives the rule it was written for."""
    rows, loose = audit()
    problems = [f"{row.title}: {p}" for row in rows for p in row.problems] + loose
    assert not problems, "\n".join(
        ["the invariant audit and CLAUDE.md have drifted apart:", *problems]
    )


def test_pytest_collects_every_test_the_registry_names():
    """Existing in a file is not the same as running. A test whose module stops
    importing, or whose name stops matching `python_functions`, is still there
    in the source and enforces nothing."""
    rows, loose = audit(collect=True)
    uncollected = [f"{row.title}: {p}" for row in rows for p in row.problems]
    assert not uncollected and not loose, "\n".join(
        ["tests the registry names that pytest does not collect:", *uncollected, *loose]
    )


def test_every_invariant_in_the_document_is_accounted_for():
    """The count is published in docs/benchmarks.md, so it is worth having a
    test say out loud when it moves."""
    bullets = invariant_bullets()
    assert len(bullets) == len(REGISTRY) == 16, (
        f"CLAUDE.md states {len(bullets)} invariants and the registry has "
        f"{len(REGISTRY)} entries; docs/benchmarks.md and both READMEs publish 16"
    )


def test_the_published_counts_are_what_the_registry_says():
    """13 enforced, 2 partly enforced, 1 unenforceable -- the numbers in
    docs/benchmarks.md and in both READMEs. Changing one means republishing the
    others."""
    counts = {status: 0 for status in (ENFORCED, PARTIAL, UNENFORCEABLE)}
    for entry in REGISTRY:
        counts[entry.status] += 1
    assert counts == {ENFORCED: 13, PARTIAL: 2, UNENFORCEABLE: 1}, (
        f"{counts} -- if this is an improvement, say so in docs/benchmarks.md "
        "and move the number here"
    )


def test_an_unenforceable_invariant_says_why_and_names_no_test():
    """The escape hatch has to cost something to use, or every awkward
    invariant becomes prose discipline."""
    for entry in REGISTRY:
        if entry.status != UNENFORCEABLE:
            continue
        assert not entry.tests, f"{entry.match!r}: unenforceable but names tests"
        assert entry.gap or entry.note, f"{entry.match!r}: unenforceable with no reason"


def test_a_partly_enforced_invariant_says_what_is_unguarded():
    """`partial` without a stated gap is `enforced` with a hedge."""
    for entry in REGISTRY:
        if entry.status == PARTIAL:
            assert entry.gap, f"{entry.match!r}: partial, but nothing said about the gap"


def test_every_enforced_invariant_is_backed_by_a_recorded_mutation():
    """The registry's claims were each checked by breaking the invariant and
    watching the suite. Keeping that evidence beside the claim is what stops a
    future entry being added on a hunch."""
    for entry in REGISTRY:
        if entry.status in (ENFORCED, PARTIAL):
            assert entry.proved_by, f"{entry.match!r}: no mutation recorded"


# ---------------------------------------------------------------------------
# The checker, shown failing. An audit that cannot fail grants a false
# all-clear to twelve invariants at once -- which is the exact failure this
# whole file exists because of.
# ---------------------------------------------------------------------------


REAL = REGISTRY[0].match


def test_a_renamed_test_is_reported_as_drift():
    rows, _ = audit(
        registry=(
            Entry(
                match=REAL,
                status=ENFORCED,
                tests=("tests/test_operating_point.py::test_that_was_renamed_away",),
                proved_by="n/a",
            ),
        )
    )
    assert any("has no test_that_was_renamed_away" in p for p in rows[0].problems)


def test_a_test_file_that_is_gone_is_reported_as_drift():
    rows, _ = audit(
        registry=(
            Entry(
                match=REAL,
                status=ENFORCED,
                tests=("tests/test_deleted.py::test_anything",),
                proved_by="n/a",
            ),
        )
    )
    assert any("does not exist" in p for p in rows[0].problems)


def test_an_invariant_with_no_entry_is_reported_as_drift():
    _, loose = audit(
        registry=(),
        bullets=["**A brand new rule.** Nothing enforces it and nothing says so."],
    )
    assert any("has no entry in the registry" in line for line in loose)


def test_an_entry_whose_invariant_was_reworded_away_is_reported_as_drift():
    _, loose = audit(
        registry=(Entry(match="a rule nobody wrote", status=ENFORCED, tests=("x::y",)),),
        bullets=["**Some other rule.**"],
    )
    assert any("matches no invariant" in line for line in loose)


def test_an_invariant_naming_no_test_at_all_is_reported_as_drift():
    """The 2026-08-22 state of `split_by_image`: a rule with nothing behind it,
    and nothing anywhere saying so."""
    rows, _ = audit(registry=(Entry(match=REAL, status=ENFORCED, proved_by="n/a"),))
    assert any("names no test" in p for p in rows[0].problems)


def test_an_enforced_invariant_held_only_by_dataset_tests_is_reported_as_drift():
    """CI runs `-m "not dataset"`. A guard behind that marker is a guard that
    runs on one laptop with 231MB of board scans on it, which is not the same
    as a guard."""
    rows, _ = audit(
        registry=(
            Entry(
                match=REAL,
                status=ENFORCED,
                tests=("tests/test_deeppcb.py::test_official_splits_have_the_documented_sizes",),
                proved_by="n/a",
            ),
        )
    )
    assert rows[0].dataset_only
    assert any("never runs one of them" in p for p in rows[0].problems)


def test_an_unenforceable_entry_that_names_a_test_is_reported_as_drift():
    rows, _ = audit(
        registry=(Entry(match=REAL, status=UNENFORCEABLE, tests=("x::y",), note="because"),),
    )
    assert any("names tests" in p for p in rows[0].problems)


@pytest.mark.parametrize(
    "node_id, expected",
    [
        ("tests/test_deeppcb.py::test_official_splits_have_the_documented_sizes", True),
        ("tests/test_deeppcb.py::test_unknown_split_is_rejected", False),
    ],
)
def test_the_dataset_marker_is_read_off_the_decorator(node_id, expected):
    assert inspect_test(node_id).dataset is expected


def test_a_bullet_is_named_by_its_bold_lead_or_its_first_clause():
    assert title_of("**Use the official DeepPCB split.** Do not re-split.") == (
        "Use the official DeepPCB split"
    )
    assert title_of("Split train/val **by image**, never by patch -- patches leak.") == (
        "Split train/val by image, never by patch"
    )


def test_a_multi_line_bullet_is_folded_into_one_string():
    """The `match` phrase usually sits on a continuation line, not the first
    one. A parser that read only the first line would report every entry as
    matching nothing."""
    bullets = invariant_bullets(
        "## Invariants — do not quietly change these\n"
        "- **A rule.** It carries on\n"
        "  onto a second line.\n"
        "- **Another.**\n"
        "\n"
        "## Environment gotchas\n"
        "- not an invariant\n"
    )
    assert bullets == ["**A rule.** It carries on onto a second line.", "**Another.**"]
