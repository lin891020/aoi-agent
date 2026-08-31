"""What `scripts/report.py` publishes, and whether it is still a curve.

The project's first invariant is that the model is reported as an
operating-point curve and never as bare accuracy: an escape ships a bad board,
a false call costs an operator a few seconds, and one number that weighs them
the same is not a result anybody on a line can act on.

Every part of that was tested except the part a reader sees. The arithmetic
lives in `vision/operating_point.py` and is held hard by
`tests/test_operating_point.py`; the *report* imported into nothing. Emptying
`BUDGETS` -- which publishes one accuracy figure, no table, and no trade-off --
left all 691 tests green, and the headline invariant of this repository was
the only one whose breach nothing could notice.

So this file runs the script. Not against a string: the report is generated
over a synthetic split with a real cost trade-off built into it, the expected
operating points are computed independently from `sweep`, and what is asserted
is that each budget the script publishes comes back as a row carrying both the
escape rate it achieved and the review it removed, in that order of emphasis.
Rewording the prose is free. Publishing one point, or leading with accuracy,
is not.
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pytest

import report
from aoi_agent.vision.operating_point import best_at_escape_budget, sweep

LABEL_NAMES = ["false_call", "open", "short"]
FALSE_CALL = LABEL_NAMES.index("false_call")

#: Percentages are printed to one or two decimal places, so a published figure
#: sits within half of the last place of the value it came from.
TOLERANCE = 6e-4


def synthetic_predictions(path):
    """A split on which the budget genuinely changes the answer.

    100 defects and 300 false calls, with P(false_call) spread across the range
    on both, so a tighter escape budget really does force a higher threshold
    and clear less of the queue. A separable toy split would make every budget
    return the same point, and the test would then pass on a report that
    published one point six times.
    """
    defects = np.linspace(0.0, 0.99, 100)
    false_calls = np.linspace(0.2, 1.0, 300)

    p_false_call = np.concatenate([defects, false_calls])
    # Both defect classes are populated: the report also breaks the escapes
    # down per class, and a class with no rows in the split is not a case
    # this file is here to exercise.
    labels = np.concatenate([np.tile([1, 2], 50), np.full(300, FALSE_CALL)])

    probabilities = np.zeros((len(labels), len(LABEL_NAMES)))
    probabilities[:, FALSE_CALL] = p_false_call
    probabilities[:, 1] = (1.0 - p_false_call) * 0.7
    probabilities[:, 2] = (1.0 - p_false_call) * 0.3

    np.savez(
        path,
        probabilities=probabilities,
        labels=labels,
        label_names=np.array(LABEL_NAMES),
    )
    return probabilities, labels


def run_report(tmp_path, monkeypatch) -> str:
    predictions = tmp_path / "test_predictions.npz"
    out = tmp_path / "benchmarks.md"
    synthetic_predictions(predictions)
    monkeypatch.setattr(
        sys,
        "argv",
        # `--cv` at a path that does not exist: these tests are about the curve,
        # and letting the script find the real selection record would put the
        # deployed threshold of the day into a fixture built from synthetic
        # scores.
        ["report.py", "--predictions", str(predictions), "--out", str(out),
         "--cv", str(tmp_path / "no-such-selection.json")],
    )
    assert report.main() == 0
    return out.read_text()


def expected_points(tmp_path):
    """The operating points the script should be publishing, computed here."""
    predictions = tmp_path / "expected.npz"
    probabilities, labels = synthetic_predictions(predictions)
    points = sweep(probabilities[:, FALSE_CALL], labels, FALSE_CALL)
    return [best_at_escape_budget(points, budget) for budget in report.BUDGETS]


def tables(text: str) -> list[list[str]]:
    """Every markdown table in the document, as runs of its non-rule rows."""
    found: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("|"):
            if not set(line) <= set("|- "):
                current.append(line)
        elif current:
            found.append(current)
            current = []
    if current:
        found.append(current)
    return found


def column(rows: list[str], keyword: str) -> list[str]:
    """One column of a table, picked out by a word in its header.

    Reading the column by name rather than by position is what lets the report
    gain, drop or reorder columns without this file caring, while still
    holding it to publishing the one the invariant is about.
    """
    cells = [[c.strip() for c in row.strip("|").split("|")] for row in rows]
    index = next(i for i, name in enumerate(cells[0]) if keyword in name.lower())
    return [row[index] for row in cells[1:]]


def percentages(text: str) -> list[float]:
    """Every percentage in a stretch of text, as a fraction."""
    return [float(m) / 100.0 for m in re.findall(r"(\d+(?:\.\d+)?)%", text)]


def carries(row: str, value: float) -> bool:
    return any(abs(found - value) <= TOLERANCE for found in percentages(row))


@pytest.fixture
def published(tmp_path, monkeypatch) -> str:
    return run_report(tmp_path, monkeypatch)


def test_the_report_sweeps_more_than_one_escape_budget():
    """A curve needs points. One budget -- or none -- is a number with a table
    drawn around it, and it is the single easiest way to lose this invariant:
    `BUDGETS` is a module-level list and nothing else in the tree reads it."""
    budgets = report.BUDGETS
    assert len(set(budgets)) >= 3, (
        f"BUDGETS={budgets}: the report publishes a curve, so it has to sweep "
        "several escape budgets. One point is bare accuracy with more columns"
    )
    assert list(budgets) == sorted(budgets), (
        f"BUDGETS={budgets}: budgets are published in order, tightest first"
    )
    assert max(budgets) >= 10 * min(budgets), (
        f"BUDGETS={budgets}: the range has to span the tolerances a line might "
        "actually choose between, or the curve answers no question"
    )


def test_every_budget_is_published_with_what_it_costs_and_what_it_buys(
    tmp_path, published
):
    """The row is the unit of the invariant. Each budget must come back with
    both halves of the trade-off -- the escape rate it achieved and the review
    it removed -- and both must be the numbers the sweep actually computes,
    not a figure the report arrived at some other way."""
    rows = tables(published)[0]
    header, data = rows[0], rows[1:]
    assert "escape" in header.lower() and "review" in header.lower(), header

    assert len(data) == len(report.BUDGETS) >= 3, (
        f"{len(data)} rows for {len(report.BUDGETS)} budgets -- a curve is a "
        "row per budget, and at least a few of them:\n" + "\n".join(rows)
    )

    for budget, point, row in zip(report.BUDGETS, expected_points(tmp_path), data):
        assert point is not None, f"budget {budget} is unreachable on this split"
        assert carries(row, point.escape_rate), (
            f"budget {budget}: the row does not carry the escape rate the sweep "
            f"gives ({point.escape_rate:.4f}) -- {row}"
        )
        assert carries(row, point.review_reduction), (
            f"budget {budget}: the row does not carry the review removed "
            f"({point.review_reduction:.4f}) -- {row}"
        )


def test_the_published_points_are_a_trade_off_and_not_one_point_repeated(published):
    """Six rows holding the same operating point are a table, not a curve. On
    this split the budgets are genuinely different decisions, so a report that
    is still sweeping must show the review removed rising as the escape budget
    loosens -- monotonically, because a looser budget can only ever admit more
    thresholds."""
    removed = [
        percentages(cell)[0] for cell in column(tables(published)[0], "review")
    ]

    assert len(set(removed)) >= 3, (
        f"{removed}: every budget publishes the same point, so nothing here is "
        "a trade-off between an escape and a review"
    )
    assert removed == sorted(removed), (
        f"{removed}: review removed has to rise as the escape budget loosens"
    )


def test_accuracy_is_not_the_headline(published):
    """Accuracy is reported -- it is the number an outside reader arrives
    expecting -- but it is reported after the curve and without emphasis. A
    report that leads with it has abandoned the invariant while still printing
    the table."""
    accuracy = next(
        line for line in published.splitlines() if "accuracy" in line.lower()
    )
    table = tables(published)[0]

    assert published.index(accuracy) > published.index(table[-1]), (
        "the accuracy figure is published before the operating-point table:\n"
        f"{accuracy}"
    )
    assert not re.search(r"\*\*[^*]*%[^*]*\*\*", accuracy), (
        f"accuracy is emphasised as though it were the result: {accuracy}"
    )

    emphasised = [
        value
        for span in re.findall(r"\*\*([^*]+)\*\*", published)
        for value in percentages(span)
    ]
    review_removed = {
        round(value, 6) for cell in column(table, "review") for value in percentages(cell)
    }
    for value in emphasised:
        assert round(value, 6) in review_removed, (
            f"{value:.1%} is emphasised in the report but is not one of the "
            "operating-point figures"
        )


def test_the_deployed_threshold_is_published_beside_the_oracle_it_replaced(
    tmp_path, monkeypatch
):
    """The section added on 2026-08-31, and the reason it exists.

    The sweep table gives every budget the best threshold *this* split reaches,
    which is an oracle: fair between engines, and not deployable. A threshold
    chosen elsewhere has to be reported here as what it does on this split, and
    the two have to appear together -- printing only the oracle is how a figure
    that had seen the answers was published as a deployment number for nine
    days.
    """
    import json

    predictions = tmp_path / "test_predictions.npz"
    out = tmp_path / "benchmarks.md"
    synthetic_predictions(predictions)
    record = tmp_path / "cv.json"
    record.write_text(json.dumps({
        "folds": 5, "budget": 0.005,
        "upper_bound": {"threshold": 0.8, "escapes": 3, "defects_total": 900,
                        "escape_rate": 0.0033, "review_reduction": 0.4},
    }))
    monkeypatch.setattr(
        sys, "argv",
        ["report.py", "--predictions", str(predictions), "--out", str(out),
         "--cv", str(record)],
    )
    assert report.main() == 0
    text = out.read_text()

    assert "0.8000" in text, "the deployed threshold itself is not printed"
    assert "out-of-fold" in text, "the deployed row does not say what chose it"
    assert "not deployable" in text, (
        "the oracle row is printed without saying it is one, which is the "
        "confusion this section exists to end"
    )
    deployed = text.index("deployed")
    assert text.index("oracle on this split", deployed) > deployed, (
        "the oracle is printed above the deployed row; the deployed row is the "
        "one a reader should take away"
    )
