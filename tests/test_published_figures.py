"""The numbers in README.md and CLAUDE.md have to be the ones the repo ships.

The failure this file exists for, found 2026-08-26. `README.md` opened with
"56.2% of manual review removed at a ≤0.50% escape budget over 8,143
candidates". That table was produced on **2026-08-22 by a working tree whose own
benchmarks header says `commit uncommitted`** -- so it could not be reproduced
even in principle. Two days later the registration stage was turned on, the
candidate population fell to 7,322, the model was retrained and the threshold
re-swept from 0.915 to 0.961. The retraining skill's chain was followed to the
letter and the *code* was correct throughout. Only the documentation was left
quoting the first run, and nothing anywhere compared the two.

It was the fourth stale claim found that day and the worst-placed: the headline
an interviewer reads first, unreproducible from the repository that carries it.

The rule here is derived from `docs/benchmarks.md`, which is append-only and
newest-last, so the file itself says which run is current:

1. every figure that identifies the **current** run must appear in README.md
2. a figure that identifies a **superseded** run may appear only in a paragraph
   that dates it -- because "this used to read 56.2%, and here is when it
   stopped" is exactly the sentence a reader needs, while a bare 56.2% is the
   defect

Rule 2 is the one that bites. Rule 1 alone passes on a document that quotes
both numbers and lets the reader pick.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "docs" / "benchmarks.md"
PUBLISHED = ("README.md", "README.zh-TW.md", "CLAUDE.md", "docs/architecture.md")

#: `Test split: 7322 AOI candidates from 499 unseen boards (3018 real defects,
#: 4304 false calls)` -- the line every operating-point section opens with.
SPLIT = re.compile(
    r"Test split: (\d+) AOI candidates from \d+ unseen boards "
    r"\((\d+) real defects, (\d+) false calls\)"
)

#: A date anywhere in the paragraph is what licenses a superseded figure.
DATED = re.compile(r"2026-\d{2}-\d{2}")


def runs() -> list[tuple[int, int, int]]:
    """Every operating-point run the benchmarks file records, oldest first."""
    found = SPLIT.findall(BENCHMARKS.read_text())
    return [tuple(int(n) for n in row) for row in found]


def spellings(value: int) -> set[str]:
    """`7322` and `7,322` are the same claim and both get written."""
    return {str(value), f"{value:,}"}


@pytest.fixture(scope="module")
def current():
    all_runs = runs()
    if len(all_runs) < 1:
        pytest.skip("no operating-point run recorded yet")
    return all_runs[-1]


@pytest.fixture(scope="module")
def superseded():
    all_runs = runs()
    if len(all_runs) < 2:
        pytest.skip("only one run recorded; nothing can be superseded yet")
    return all_runs[:-1]


def paragraphs(name: str) -> list[str]:
    return re.split(r"\n\s*\n", (ROOT / name).read_text())


def test_the_benchmarks_file_records_more_than_one_run(superseded):
    """Rule 2 is vacuous on a repository that has only ever measured once, and a
    guard that cannot fail is not a guard. This says so out loud."""
    assert superseded


def test_the_readme_headline_is_the_run_the_repo_ships(current):
    candidates, defects, false_calls = current
    readme = (ROOT / "README.md").read_text()

    assert spellings(candidates) & set(re.findall(r"[\d,]+", readme)), (
        f"README does not name the current candidate count {candidates:,}"
    )
    assert spellings(defects) & set(re.findall(r"[\d,]+", readme)), (
        f"README does not name the current defect count {defects:,}"
    )
    assert false_calls == candidates - defects, (
        "the benchmarks split line does not add up; the parse is wrong"
    )


@pytest.mark.parametrize("name", PUBLISHED)
def test_a_superseded_count_may_only_appear_where_it_is_dated(
    name, current, superseded
):
    """The half that bites.

    A document quoting both the old and the new number reads as though the
    reader may choose. They may not: one of them does not reproduce.
    """
    live = {spelling for value in current for spelling in spellings(value)}
    dead = {
        spelling
        for run in superseded
        for value in run
        for spelling in spellings(value)
    } - live

    offences = []
    for paragraph in paragraphs(name):
        if DATED.search(paragraph):
            continue
        tokens = set(re.findall(r"[\d,]+", paragraph))
        for stale in sorted(dead & tokens):
            offences.append((stale, paragraph.strip()[:90]))

    assert not offences, (
        f"{name} quotes a superseded run without dating it: "
        + "; ".join(f"{n} in {p!r}" for n, p in offences)
    )
