"""One escape budget over six classes that the work instructions do not treat
alike -- and why the obvious fix for that is not available.

The finding this guards is a negative one, which is the kind that rots quietest:
a class-aware dismissal rule looks like the natural answer to a critical class
escaping at several times the aggregate, and it does not work, because the
escaped instances carry no signal to veto on. If a future checkpoint changes
that, these tests should fail and somebody should go and read the report again.

**Nothing here names a class, and that is the point.** These tests said `open`
until 2026-08-24. They were written when `open` was the worst critical class at
1.35%, and a retrain moved that to `short` at 1.55% while `open` fell to 0.83%.
One assertion went red and the rest stayed green while the file's own docstring
had become false -- a finding pinned to a class name survives the class it was
about. The property is "a class WI-201 or WI-202 governs", which is a column in
`GOVERNS`, so that is what these read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aoi_agent.vision.inference import DEFAULT_DISMISS_THRESHOLD  # noqa: E402
from class_escape_report import GOVERNS  # noqa: E402

PREDICTIONS = Path(__file__).resolve().parents[1] / "models/test_predictions.npz"

#: The classes whose work instruction admits no acceptable instance at all.
#: Read off `GOVERNS` rather than listed, so a document that changes what it
#: permits moves these tests with it.
CRITICAL = sorted(name for name, (_doc, _says, critical) in GOVERNS.items() if critical)


@pytest.fixture
def scores():
    if not PREDICTIONS.exists():
        pytest.skip("no stored predictions; run scripts/train.py")
    data = np.load(PREDICTIONS, allow_pickle=False)
    names = [str(n) for n in data["label_names"]]
    return data["probabilities"], data["labels"], names


def rates(scores):
    """Per-class escape rate, and the aggregate they are averaged into."""
    probabilities, labels, names = scores
    false_call = names.index("false_call")
    dismissed = probabilities[:, false_call] >= DEFAULT_DISMISS_THRESHOLD
    is_defect = labels != false_call

    per_class = {}
    for name in names:
        if name == "false_call":
            continue
        mask = labels == names.index(name)
        per_class[name] = float((dismissed & mask).sum()) / int(mask.sum())
    aggregate = float((dismissed & is_defect).sum()) / int(is_defect.sum())
    return per_class, aggregate


def test_every_defect_class_names_the_document_that_governs_it(scores):
    """The table is an argument about the work instructions, so it has to cover
    every class the model can emit -- a missing row is a class quietly left out
    of the argument."""
    _probabilities, _labels, names = scores

    assert set(GOVERNS) == {n for n in names if n != "false_call"}


def test_the_argument_has_a_critical_class_to_be_about(scores):
    """`CRITICAL` is derived, so it can silently empty out -- and an empty list
    turns every parametrised test below into zero tests that pass. This is the
    one assertion the others cannot make for themselves."""
    assert CRITICAL, "no class is marked critical in GOVERNS; the finding has no subject"


def test_the_aggregate_no_longer_meets_the_budget_it_publishes(scores):
    """This asserted the opposite until 2026-08-31, and it was true then for a
    reason that stopped being acceptable.

    The aggregate met QP-110 at 0.50% because the threshold reporting that
    figure had been swept on this very split. Chosen out-of-fold instead --
    `scripts/threshold_cv.py`, 6,569 defects, the interval's upper bound -- the
    threshold is 0.912 and the same split escapes 0.66%. The class split below
    is unchanged in shape and worse in size, which is the point: the aggregate
    was never the reassurance it read as.

    Kept as an assertion rather than deleted, so that a future retrain which
    genuinely brings the aggregate back inside the budget has to come here and
    say so, rather than sliding under a test that no longer looks.
    """
    _per_class, aggregate = rates(scores)

    assert aggregate > 0.005, (
        f"the aggregate escape rate is {aggregate:.2%}, back inside QP-110's "
        f"0.5%. If a retrain did that, rewrite this test and the paragraph in "
        f"CLAUDE.md it holds; if a threshold was re-tuned on this split, that "
        f"is the thing 2026-08-31 removed."
    )


@pytest.mark.parametrize("name", CRITICAL)
def test_a_class_that_admits_no_instance_exceeds_the_budget_it_is_averaged_into(
    scores, name
):
    """The finding. Every class whose document permits nothing exceeds QP-110 --
    which the aggregate they are averaged into used to meet, and since
    2026-08-31 does not either."""
    per_class, aggregate = rates(scores)

    assert per_class[name] > 0.005, (
        f"{name} is governed by a document that admits no acceptable instance "
        f"and now escapes at {per_class[name]:.2%}, inside the budget. That is "
        f"good news and it breaks the argument in docs/benchmarks.md -- go and "
        f"re-read it rather than relaxing this."
    )
    assert per_class[name] > aggregate


def test_the_worst_critical_class_is_several_times_the_aggregate(scores):
    """The magnitude, kept separate from the direction above. `2x` is the claim
    docs/benchmarks.md makes; the class it is true of is not part of the claim
    and has already changed once."""
    per_class, aggregate = rates(scores)
    worst = max(CRITICAL, key=lambda name: per_class[name])

    assert per_class[worst] > aggregate * 2, (
        f"the worst critical class is {worst} at {per_class[worst]:.2%} against "
        f"an aggregate of {aggregate:.2%}"
    )


@pytest.mark.parametrize("name", CRITICAL)
def test_the_escapes_are_confident_errors_not_uncertain_ones(scores, name):
    """Why a class-aware veto cannot work here, stated as the property rather
    than as the failed sweep.

    A veto needs the escaped instances to sit somewhere separable in `P(class)`.
    They do not: the model puts them below anything it puts a kept instance at,
    so no cut exists that keeps one population and drops the other.
    """
    probabilities, labels, names = scores
    false_call = names.index("false_call")
    index = names.index(name)
    dismissed = probabilities[:, false_call] >= DEFAULT_DISMISS_THRESHOLD
    mask = labels == index

    escaped = probabilities[dismissed & mask][:, index]
    kept = probabilities[~dismissed & mask][:, index]

    # At most one escaped instance is anywhere a veto could reach. Until the
    # threshold moved to 0.912 on 2026-08-31 it was none; at the lower threshold
    # one escaped `short` carries P(short) 0.078, and a veto does recover it --
    # see the test below for what that buys, which is not the budget.
    reachable = escaped[escaped >= 0.05]
    assert len(reachable) <= 1, (
        f"{len(reachable)} escaped {name}s carry P({name}) >= 0.05; the "
        f"'confidently wrong' reading is about a population that no longer "
        f"looks like this and wants re-measuring"
    )
    assert float(np.sort(escaped)[::-1][1:].max()) < 0.01, (
        f"every escaped {name} but the first carries essentially no P({name})"
    )
    assert float(np.median(kept)) > 0.9, f"kept {name}s carry almost all of it"
    # Confident, not uncertain: every one is a `false_call` argmax, well clear.
    assert probabilities[dismissed & mask][:, false_call].min() > 0.9


@pytest.mark.parametrize("name", CRITICAL)
def test_a_veto_low_enough_to_bite_costs_more_review_than_it_recovers(scores, name):
    """The trade, stated as what it cannot buy rather than as what it moves.

    Until 2026-08-31 this asserted that nothing moved at any veto worth trying,
    which was true at a threshold of 0.961. At 0.912 a veto at P(short) > 0.05
    recovers one of the eight escaped shorts for 0.22 points of review -- so
    "nothing moves" is no longer the finding. What survives is the finding that
    mattered: **no veto brings a critical class inside the budget it is
    averaged into**, because the instances a veto can reach are one, and the
    other fourteen sit below 0.0086 in their own class probability.
    """
    probabilities, labels, names = scores
    false_call = names.index("false_call")
    index = names.index(name)
    dismissed = probabilities[:, false_call] >= DEFAULT_DISMISS_THRESHOLD
    mask = labels == index
    defects = int(mask.sum())

    for veto in (0.30, 0.10, 0.05, 0.02, 0.01):
        kept = dismissed & ~(probabilities[:, index] > veto)
        rate = int((kept & mask).sum()) / defects
        assert rate > 0.005, (
            f"a veto at P({name}) > {veto} brings {name} to {rate:.2%}, inside "
            f"QP-110's 0.5%. The report's negative result is that no veto on "
            f"this model's own output can do that -- re-read it rather than "
            f"relaxing this."
        )


# ---- the prose under the table has to agree with the table ---------------


def test_the_prose_reports_the_same_open_escapes_the_table_does(scores):
    """The report's closing paragraphs are about `open` -- the class WI-201
    sends to electrical test -- and they were hand-written.

    By 2026-08-26 they read "these eight ... 8 escapes in 594 opens has a 95%
    interval of 0.68% to 2.63%" three lines under a table saying **5 of 602**.
    True of the run they were written on, reprinted verbatim on every run since,
    and contradicting the table directly above them. Nothing failed, because
    nothing was comparing a sentence to a number.

    So the numbers are derived, and this is what stops them being written by
    hand again: whatever the table says about `open`, the prose says the same.
    """
    import re

    import class_escape_report

    probabilities, labels, names = scores
    report = class_escape_report.render(probabilities, labels, names)

    open_index = names.index("open")
    false_call = names.index("false_call")
    dismissed = probabilities[:, false_call] >= DEFAULT_DISMISS_THRESHOLD
    mask = labels == open_index
    escaped, total = int((dismissed & mask).sum()), int(mask.sum())

    row = re.search(r"\|\s*\*\*open\*\*\s*\|[^|]*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", report)
    assert row, "no open row in the rendered table"
    assert (int(row.group(2)), int(row.group(1))) == (escaped, total)

    sentence = re.search(r"(\d+) escapes in (\d+) opens", report)
    assert sentence, "the limits paragraph no longer states its counts"
    assert (int(sentence.group(1)), int(sentence.group(2))) == (escaped, total), (
        "the prose and the table disagree about open -- the 2026-08-26 defect"
    )

    covered = re.search(r"these (\d+) are already covered", report)
    assert covered and int(covered.group(1)) == escaped


def test_the_published_interval_is_the_one_wilson_gives(scores):
    """A hand-typed interval is the same failure one indirection along."""
    import re

    import class_escape_report
    from aoi_agent.stats import wilson

    probabilities, labels, names = scores
    report = class_escape_report.render(probabilities, labels, names)

    open_index, false_call = names.index("open"), names.index("false_call")
    dismissed = probabilities[:, false_call] >= DEFAULT_DISMISS_THRESHOLD
    mask = labels == open_index
    low, high = wilson(int((dismissed & mask).sum()), int(mask.sum()))

    published = re.search(r"95% interval of ([\d.]+)% to ([\d.]+)%", report)
    assert published, "the interval is no longer published"
    assert float(published.group(1)) == pytest.approx(low * 100, abs=0.01)
    assert float(published.group(2)) == pytest.approx(high * 100, abs=0.01)
