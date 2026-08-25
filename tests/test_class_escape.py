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


def test_the_aggregate_meets_the_budget_it_publishes(scores):
    """Half of the finding, and the half that makes the other half worth
    stating. If this ever fails the headline is wrong, not just the split."""
    _per_class, aggregate = rates(scores)

    assert aggregate <= 0.005


@pytest.mark.parametrize("name", CRITICAL)
def test_a_class_that_admits_no_instance_exceeds_the_budget_it_is_averaged_into(
    scores, name
):
    """The finding. Every class whose document permits nothing exceeds QP-110,
    while the aggregate they are averaged into meets it."""
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

    assert escaped.max() < 0.05, f"escaped {name}s carry essentially no P({name})"
    assert float(np.median(kept)) > 0.9, f"kept {name}s carry almost all of it"
    # Confident, not uncertain: every one is a `false_call` argmax, well clear.
    assert probabilities[dismissed & mask][:, false_call].min() > 0.9


@pytest.mark.parametrize("name", CRITICAL)
def test_a_veto_low_enough_to_bite_costs_more_review_than_it_recovers(scores, name):
    """The trade, kept as a number so a future checkpoint that changes it is
    visible. At every veto worth trying, nothing is recovered at all."""
    probabilities, labels, names = scores
    false_call = names.index("false_call")
    index = names.index(name)
    dismissed = probabilities[:, false_call] >= DEFAULT_DISMISS_THRESHOLD
    mask = labels == index

    baseline = int((dismissed & mask).sum())
    for veto in (0.30, 0.10, 0.05, 0.02):
        kept = dismissed & ~(probabilities[:, index] > veto)
        assert int((kept & mask).sum()) == baseline, (
            f"a veto at {veto} started recovering {name}s -- the report's "
            f"negative result no longer holds and wants re-reading"
        )
