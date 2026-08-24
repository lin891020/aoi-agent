"""One escape budget over six classes that the work instructions do not treat
alike -- and why the obvious fix for that is not available.

The finding this guards is a negative one, which is the kind that rots quietest:
a class-aware dismissal rule looks like the natural answer to `open` escaping at
three times the aggregate, and it does not work, because the escaped opens carry
no signal to veto on. If a future checkpoint changes that, these tests should
fail and somebody should go and read the report again.
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


@pytest.fixture
def scores():
    if not PREDICTIONS.exists():
        pytest.skip("no stored predictions; run scripts/train.py")
    data = np.load(PREDICTIONS, allow_pickle=False)
    names = [str(n) for n in data["label_names"]]
    return data["probabilities"], data["labels"], names


def test_every_defect_class_names_the_document_that_governs_it(scores):
    """The table is an argument about the work instructions, so it has to cover
    every class the model can emit -- a missing row is a class quietly left out
    of the argument."""
    _probabilities, _labels, names = scores

    assert set(GOVERNS) == {n for n in names if n != "false_call"}


def test_the_critical_class_escapes_worse_than_the_budget_it_is_averaged_into(scores):
    """The finding. `open` admits no acceptable instance under WI-201 and is the
    class exceeding QP-110, while the aggregate meets it."""
    probabilities, labels, names = scores
    false_call = names.index("false_call")
    dismissed = probabilities[:, false_call] >= DEFAULT_DISMISS_THRESHOLD
    is_defect = labels != false_call

    aggregate = float((dismissed & is_defect).sum()) / int(is_defect.sum())
    is_open = labels == names.index("open")
    open_rate = float((dismissed & is_open).sum()) / int(is_open.sum())

    assert aggregate <= 0.005, "the aggregate budget is met"
    assert open_rate > 0.005, "and the critical class is not"
    assert open_rate > aggregate * 2


def test_the_escaped_opens_are_confident_errors_not_uncertain_ones(scores):
    """Why a class-aware veto cannot work here, stated as the property rather
    than as the failed sweep.

    A veto needs the escaped opens to sit somewhere separable in `P(open)`. They
    do not: the model puts them below anything it puts a kept open at, so no cut
    exists that keeps one population and drops the other.
    """
    probabilities, labels, names = scores
    false_call = names.index("false_call")
    open_index = names.index("open")
    dismissed = probabilities[:, false_call] >= DEFAULT_DISMISS_THRESHOLD
    is_open = labels == open_index

    escaped = probabilities[dismissed & is_open][:, open_index]
    kept = probabilities[~dismissed & is_open][:, open_index]

    assert escaped.max() < 0.05, "escaped opens carry essentially no P(open)"
    assert float(np.median(kept)) > 0.9, "kept opens carry almost all of it"
    # Confident, not uncertain: every one is a `false_call` argmax, well clear.
    assert probabilities[dismissed & is_open][:, false_call].min() > 0.9


def test_a_veto_low_enough_to_bite_costs_more_review_than_it_recovers(scores):
    """The trade, kept as a number so a future checkpoint that changes it is
    visible. At the only veto that recovers any open at all, the review
    reduction lost is worth more than the three escapes bought."""
    probabilities, labels, names = scores
    false_call = names.index("false_call")
    open_index = names.index("open")
    dismissed = probabilities[:, false_call] >= DEFAULT_DISMISS_THRESHOLD
    is_open = labels == open_index

    baseline_opens = int((dismissed & is_open).sum())
    for veto in (0.30, 0.10, 0.05, 0.02):
        kept = dismissed & ~(probabilities[:, open_index] > veto)
        assert int((kept & is_open).sum()) == baseline_opens, (
            f"a veto at {veto} started recovering opens -- the report's negative "
            f"result no longer holds and wants re-reading"
        )
