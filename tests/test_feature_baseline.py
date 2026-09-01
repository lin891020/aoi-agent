"""The hand-feature floor: what the features must measure, and what may fit them.

The measurement needs the dataset, so what is held here is the part that can be
wrong without erroring. Two things:

1. **The feature vector and its names stay in step.** The names are what the
   benchmarks entry ranks by importance, so a feature inserted without its name
   would silently re-label every row of that ranking.
2. **The tree's one hyper-parameter may not be chosen on test.** The first
   version of this script let sklearn's early stopping split by row, which puts
   patches from one board on both sides -- the leak `split_by_image` exists to
   prevent. It also meant the seed never reached the model, which is how it was
   caught: three seeds returned an identical number.
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
def baseline():
    pytest.importorskip("sklearn", reason="the floor needs the tree")
    pytest.importorskip("torch", reason="it imports train.split_by_image")
    return _script("feature_baseline")


@pytest.fixture
def numpy():
    return pytest.importorskip("numpy")


def test_every_feature_has_a_name(baseline, numpy):
    patch = numpy.zeros((3, 64, 64), dtype=numpy.uint8)
    patch[0, 20:40, 20:40] = 255          # template: a block of foreground
    patch[1, 20:40, 20:30] = 255          # test: half of it missing
    patch[2] = numpy.abs(patch[1].astype(int) - patch[0].astype(int)).astype(numpy.uint8)
    assert len(baseline.features(patch)) == len(baseline.NAMES)


def test_an_empty_difference_produces_numbers_rather_than_nan(baseline, numpy):
    """A candidate whose difference falls below the AOI's own threshold still
    has to score: dropping it would quietly change the denominator."""
    patch = numpy.zeros((3, 64, 64), dtype=numpy.uint8)
    patch[0] = patch[1] = 200
    row = baseline.features(patch)
    assert len(row) == len(baseline.NAMES)
    assert all(numpy.isfinite(value) for value in row)


def test_the_blob_is_measured_at_the_threshold_the_candidates_came_from(baseline):
    # Measuring the blob at a different grey level would describe a different
    # object from the one the detector drew a box around.
    from aoi_agent.vision import patches as patch_module
    assert baseline.DIFF_THRESHOLD == 60
    assert getattr(patch_module, "DIFF_THRESHOLD", 60) == baseline.DIFF_THRESHOLD


def test_sklearn_early_stopping_is_off_so_the_split_reaches_the_model(baseline):
    """`early_stopping=True` splits by row, and with `validation_fraction=None`
    it does not split at all -- both make the by-image split decorative."""
    import ast

    source = (ROOT / "scripts" / "feature_baseline.py").read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and getattr(node.func, "id", None) == "HistGradientBoostingClassifier"]
    assert len(calls) == 1, "one place builds the tree, or this test guards the wrong one"
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert isinstance(keywords.get("early_stopping"), ast.Constant)
    assert keywords["early_stopping"].value is False
    # sklearn's own validation split is by row; passing it anything would put
    # patches from one board on both sides of the choice.
    assert "validation_fraction" not in keywords
    assert "staged_predict_proba" in source, "iterations must be chosen on the by-image val half"


def test_the_shape_of_a_sliver_and_a_blob_differ_in_the_feature_that_ranked_first(baseline, numpy):
    """`elongation` has to separate a registration residual from a defect, or
    the benchmarks entry's mechanism claim is about a number that means
    something else."""
    def diff_only(mask):
        patch = numpy.zeros((3, 64, 64), dtype=numpy.uint8)
        patch[2][mask] = 255
        return dict(zip(baseline.NAMES, baseline.features(patch)))

    sliver = numpy.zeros((64, 64), dtype=bool)
    sliver[30, 10:50] = True
    compact = numpy.zeros((64, 64), dtype=bool)
    compact[26:38, 26:38] = True
    assert diff_only(sliver)["elongation"] > diff_only(compact)["elongation"] * 3
