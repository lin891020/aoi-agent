"""How the dismissal threshold is chosen, and what may not choose it.

Until 2026-08-31 the shipped threshold was swept on `test_predictions.npz` --
the split every published figure is read from. As a comparison between engines
that is fair, because each gets its own oracle. As a deployment number it has
seen the answers, and measured, the difference is not cosmetic: choosing on the
single validation split instead put the test escape rate at 0.93% against a
0.5% budget.

Two rules are held here:

1. **A budget is a promise about defects nobody has seen**, so a threshold
   clears it with the *upper bound* of its interval on the set that chose it,
   not with the point estimate. One escape in a 969-defect validation split is
   0.10%, and a rule that reads only the point estimate will happily sit one
   unlucky board away from the budget.
2. **The recipe has one source.** `scripts/threshold_cv.py` trains its folds
   through `train.fit` with `train.DEFAULTS`; a second copy of the
   hyper-parameters is how a fold model quietly stops being the model whose
   threshold is being chosen.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cv():
    pytest.importorskip("torch", reason="threshold_cv imports the training stack")
    return _script("threshold_cv")


def _points(rows):
    """(threshold, escapes, defects, reduction) -> objects `choose` accepts."""
    from types import SimpleNamespace

    return [
        SimpleNamespace(threshold=t, escapes=e, defects_total=d,
                        escape_rate=e / d, review_reduction=r)
        for t, e, d, r in rows
    ]


def test_the_upper_bound_rule_refuses_a_point_estimate_that_only_just_fits(cv):
    """4/969 is 0.41% and inside a 0.5% budget; its interval reaches 1.05%.

    This is the shipped model's own validation split. The point-estimate rule
    picks it and escapes 0.93% on the test split -- nearly twice the budget --
    which is the measurement this rule exists because of.
    """
    points = _points([
        (0.61, 4, 969, 0.459),   # inside the budget on the point estimate only
        (0.99, 0, 969, 0.222),   # clears it on the interval too
    ])
    optimistic, conservative = cv.choose(points, 0.005)
    assert optimistic.threshold == 0.61
    assert conservative.threshold == 0.99


def test_a_larger_selection_set_buys_back_review_at_the_same_budget(cv):
    """The reason the threshold is chosen out-of-fold rather than on one split.

    With 969 defects the upper bound admits **zero** escapes; with 3,000 it
    admits seven, so the same rule stops having to sit at the very top of the
    curve. Nothing here is about the model -- only about how many defects the
    choice is standing on.
    """
    small = _points([(0.99, 0, 969, 0.22), (0.80, 3, 969, 0.50)])
    large = _points([(0.99, 0, 3000, 0.22), (0.80, 7, 3000, 0.50)])
    assert cv.choose(small, 0.005)[1].threshold == 0.99
    assert cv.choose(large, 0.005)[1].threshold == 0.80


def test_no_threshold_clears_the_budget_is_a_refusal_not_a_fallback(cv):
    """`None`, so a caller cannot mistake "the least bad" for "inside budget"."""
    points = _points([(0.5, 40, 1000, 0.7), (0.9, 20, 1000, 0.4)])
    optimistic, conservative = cv.choose(points, 0.005)
    assert optimistic is None and conservative is None


def test_the_folds_train_with_the_shipped_recipe_not_a_copy_of_it(cv):
    """`threshold_cv` reads `train.DEFAULTS`; nothing may re-declare them."""
    train = _script("train")
    source = (ROOT / "scripts" / "threshold_cv.py").read_text()
    for key in ("epochs", "batch_size", "lr", "val_fraction", "escape_budget"):
        assert f'DEFAULTS["{key}"]' in source or key not in source, (
            f"threshold_cv.py names {key} without reading train.DEFAULTS"
        )
    assert train.DEFAULTS["escape_budget"] == 0.005
    assert callable(train.fit)


def test_the_folds_cover_every_candidate_exactly_once(cv):
    """Out-of-fold means every candidate is predicted by a model that never
    trained on it -- and every candidate is predicted, or the sweep is over a
    subset nobody declared."""
    import numpy as np

    class FakeSet:
        image_index = np.repeat(np.arange(20), 5)

    parts = cv.folds_by_image(FakeSet(), 5, seed=0)
    assert sum(len(p) for p in parts) == 20
    assert len({int(i) for p in parts for i in p}) == 20
