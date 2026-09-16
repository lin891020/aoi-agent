"""The second DINOv2 experiment is read by the same procedure as the first, or not at all.

Five ways this one could report a number meaning something other than what it
says, each silent:

* it writes its features into DeepPCB's cache, whose filenames carry the model,
  the input mode and the split but not the dataset -- the first run would
  overwrite 145 MB of cached DeepPCB features with this dataset's, and the only
  guard is a candidate count that happens to differ;
* the configuration that becomes DINOv2's result is picked by its test column;
* the bootstrap resamples images by reading `image_index` off the PatchSet while
  the scores come from `test_predictions.npz`, which stores no image index, so a
  row order that did not survive is a bootstrap over the wrong boards;
* the bootstrap resamples candidates rather than boards, counting patches from
  one board as independent evidence;
* the verdict is read from medians after the fact rather than from the interval
  the docstring pre-registered.

No weights are downloaded and no dataset is read.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import dinov2_crops_report as crops
import dinov2_probe as probe
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["Bad_podu", "Bad_qiaojiao", "false_call"]
FALSE_CALL = 2


def _patch_set(labels, image_index):
    from aoi_agent.vision.patches import PatchSet

    labels = np.asarray(labels, dtype=np.int64)
    return PatchSet(
        patches=np.zeros((len(labels), 3, 64, 64), dtype=np.uint8), labels=labels,
        label_names=NAMES, image_index=np.asarray(image_index, dtype=np.int64),
        boxes=np.zeros((len(labels), 4), dtype=np.int64),
    )


# ---- the feature cache is not DeepPCB's ----------------------------------


def test_the_feature_cache_is_not_the_deeppcb_one():
    assert crops.FEATURES != ROOT / "data" / "features"
    assert crops.FEATURES.name == "features_pcbaoi"


@pytest.mark.parametrize("spelling", ["data/features", "./data/features",
                                      "data/../data/features"])
def test_the_deeppcb_cache_is_refused_however_it_is_spelled(spelling, monkeypatch):
    """The first guard compared the argument as typed against an absolute path,
    so a relative `--features-dir data/features` walked past it."""
    monkeypatch.chdir(ROOT)
    assert crops.shares_deeppcb_cache(Path(spelling))
    assert crops.shares_deeppcb_cache(ROOT / "data" / "features")


def test_this_datasets_own_cache_is_allowed():
    assert not crops.shares_deeppcb_cache(crops.FEATURES)


def test_the_record_is_not_the_probes_record():
    """`dinov2_latency.py` reads models/dinov2_probe.json to pick the pooling it times."""
    assert crops.OUT_JSON != ROOT / "models" / "dinov2_probe.json"


def test_the_resnet_runs_do_not_overwrite_the_published_checkpoint():
    assert "{seed}" in crops.RESNET_DIR
    assert crops.RESNET_DIR.format(seed=0) != "pcbaoi_reverifier"


# ---- the candidate set is the published one -------------------------------


def test_the_published_basis_is_accepted():
    labels = [0] * crops.PUBLISHED_FLAGGED_DEFECTS + [FALSE_CALL] * crops.PUBLISHED_FALSE_CALLS
    images = np.arange(len(labels)) % crops.PUBLISHED_IMAGES
    basis = crops.check_basis(_patch_set(labels, images))
    assert basis["candidates"] == crops.PUBLISHED_CANDIDATES
    assert basis["defects"] == crops.PUBLISHED_FLAGGED_DEFECTS
    assert round(basis["prevalence"], 3) == crops.PUBLISHED_PREVALENCE


def test_a_rebuilt_candidate_set_is_refused():
    """A different detector or floor makes a queue no published figure describes."""
    labels = [0] * 400 + [FALSE_CALL] * 100
    images = np.arange(len(labels)) % crops.PUBLISHED_IMAGES
    with pytest.raises(SystemExit):
        crops.check_basis(_patch_set(labels, images))


def test_the_comparators_are_the_published_figures():
    benchmarks = (ROOT / "docs" / "benchmarks.md").read_text()
    assert f"{crops.PUBLISHED_CANDIDATES} candidates at the detector's floor" in benchmarks
    assert (f"({crops.PUBLISHED_FLAGGED_DEFECTS} covering a defect, "
            f"{crops.PUBLISHED_FALSE_CALLS} false calls)") in benchmarks
    assert f"**{crops.PUBLISHED_PREVALENCE:.1%} genuine defects**" in benchmarks
    assert f"removes **{crops.PUBLISHED_RESNET_REVIEW:.1%}**" in benchmarks
    assert f"removes **{crops.PUBLISHED_DETECTOR_REVIEW:.1%}**" in benchmarks
    assert (f"**Basis: {crops.PUBLISHED_IMAGES} test images, "
            f"{crops.PUBLISHED_DEFECTS_ANNOTATED} annotated defects, of which "
            f"{crops.PUBLISHED_UNFLAGGED} were never boxed") in benchmarks


# ---- the rows a score came from are the rows the bootstrap resamples -------


def _predictions(tmp_path: Path, labels, names=NAMES, scores=None) -> Path:
    path = tmp_path / "test_predictions.npz"
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.zeros((len(labels), len(names)), dtype=np.float32)
    probabilities[:, names.index("false_call")] = 0.5 if scores is None else scores
    np.savez_compressed(path, probabilities=probabilities, labels=labels,
                        label_names=np.array(names))
    return path


def test_scores_are_returned_when_the_rows_line_up(tmp_path):
    labels = [0, 2, 1, 2]
    path = _predictions(tmp_path, labels, scores=[0.1, 0.2, 0.3, 0.4])
    scores = crops.aligned_scores(path, _patch_set(labels, [0, 0, 1, 1]), FALSE_CALL)
    assert np.allclose(scores, [0.1, 0.2, 0.3, 0.4])


def test_a_row_order_that_did_not_survive_is_refused(tmp_path):
    path = _predictions(tmp_path, [0, 2, 1, 2])
    with pytest.raises(SystemExit):
        crops.aligned_scores(path, _patch_set([0, 1, 2, 2], [0, 0, 1, 1]), FALSE_CALL)


def test_a_different_class_table_is_refused(tmp_path):
    path = _predictions(tmp_path, [0, 2], names=["false_call", "open"])
    with pytest.raises(SystemExit):
        crops.aligned_scores(path, _patch_set([0, 2], [0, 1]), FALSE_CALL)


# ---- both columns, one rule, and the deployable one never sees test --------


def test_the_deployable_threshold_cannot_be_chosen_on_test():
    """By signature: validation scores and labels, a class index, a budget. No test array."""
    assert list(inspect.signature(crops.deployable_point).parameters) == [
        "val_scores", "val_labels", "false_call_index", "budget"]


def test_the_deployable_threshold_is_the_shipped_selection_rule():
    import threshold_cv

    assert crops.choose is threshold_cv.choose


def test_both_columns_are_read_for_one_ordering():
    """Sized like the real splits: the upper-bound rule needs ~740 validation defects
    before zero escapes can clear a 0.5% budget, and this dataset's split holds 1,328."""
    rng = np.random.default_rng(0)
    n = 4000
    labels = np.where(rng.random(n) < 0.5, FALSE_CALL, 0)
    scores = np.where(labels == FALSE_CALL, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
    half = n // 2
    row = crops.read_columns(scores[:half], labels[:half], scores[half:], labels[half:],
                             FALSE_CALL, 0.005)
    assert row["deployable_threshold"] is not None
    assert 0.0 <= row["oracle_review_reduction"] <= 1.0
    assert row["oracle_escape_rate"] <= 0.005


# ---- the configuration is chosen out-of-fold ------------------------------


def test_the_configuration_is_still_chosen_from_out_of_fold_alone():
    assert crops.probe.select_configuration is probe.select_configuration
    assert list(inspect.signature(probe.select_configuration).parameters) == ["oof_guarded"]


# ---- the bootstrap resamples boards ---------------------------------------


def _two_orderings(n=300, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.where(rng.random(n) < 0.3, FALSE_CALL, 0)
    strong = np.where(labels == FALSE_CALL, 0.99, 0.01)
    weak = rng.uniform(0.0, 1.0, n)
    image_index = np.arange(n) % 10
    return strong, weak, labels, image_index


def test_a_dominant_ordering_produces_an_interval_above_zero():
    strong, weak, labels, image_index = _two_orderings()
    interval = crops.paired_bootstrap(strong, weak, labels, image_index, FALSE_CALL, 0.005,
                                      resamples=200)
    assert interval["low"] > 0 and interval["median"] > 0


def test_the_same_ordering_twice_is_exactly_zero():
    strong, _weak, labels, image_index = _two_orderings()
    interval = crops.paired_bootstrap(strong, strong, labels, image_index, FALSE_CALL, 0.005,
                                      resamples=100)
    assert (interval["low"], interval["median"], interval["high"]) == (0.0, 0.0, 0.0)


def test_the_bootstrap_is_deterministic():
    strong, weak, labels, image_index = _two_orderings()
    first = crops.paired_bootstrap(strong, weak, labels, image_index, FALSE_CALL, 0.005,
                                   resamples=100)
    second = crops.paired_bootstrap(strong, weak, labels, image_index, FALSE_CALL, 0.005,
                                    resamples=100)
    assert first == second


def test_whole_boards_are_resampled_not_candidates(monkeypatch):
    """Every resample is a union of whole boards, so its size is a sum of board sizes."""
    sizes = []
    real = crops.review_at_budget

    def spy(scores, labels, false_call_index, budget):
        sizes.append(len(labels))
        return real(scores, labels, false_call_index, budget)

    monkeypatch.setattr(crops, "review_at_budget", spy)
    labels = np.array([0, 0, 0, FALSE_CALL, FALSE_CALL, FALSE_CALL, 0, 0, 0], dtype=np.int64)
    image_index = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    crops.paired_bootstrap(np.linspace(0, 1, 9), np.linspace(1, 0, 9), labels, image_index,
                           FALSE_CALL, 0.005, resamples=25)
    assert sizes and all(size == 9 for size in sizes)


# ---- the verdict was written before the run -------------------------------


@pytest.mark.parametrize(("low", "high", "label"), [
    (0.4, 2.1, "helps"),
    (-0.2, 1.8, "indistinguishable"),
    (0.0, 1.8, "indistinguishable"),
    (-2.0, -0.3, "worse"),
])
def test_the_verdict_is_the_sign_of_the_paired_interval(low, high, label):
    interval = {"median": (low + high) / 2, "low": low, "high": high,
                "resamples": 2000, "skipped": 0}
    assert crops.verdict(interval, 0.05, 0.04)[0] == label


def test_the_interval_is_quoted_in_percentage_points():
    """Review reduction is a fraction; the first draft printed fractions and called
    them points, so a real -1.7 to +1.5 interval rendered as "-0.0 to +0.0"."""
    interval = {"median": -0.0019, "low": -0.0168, "high": 0.0148,
                "resamples": 2000, "skipped": 0}
    _label, sentence = crops.verdict(interval, 0.010, 0.012)
    assert "-1.7" in sentence and "+1.5" in sentence
    assert "+0.0" not in sentence and "-0.0" not in sentence


def test_two_orderings_under_one_percent_are_named_as_neither_ordering():
    interval = {"median": 0.1, "low": -0.4, "high": 0.6, "resamples": 2000, "skipped": 0}
    _label, sentence = crops.verdict(interval, 0.008, 0.006)
    assert "neither ordering is one" in sentence
    _label, other = crops.verdict(interval, 0.05, 0.006)
    assert "neither ordering is one" not in other


def test_only_the_pre_registered_run_may_append():
    assert crops.publishable([2, 0, 1], 5, probe.INPUTS, probe.POOLINGS)
    assert not crops.publishable([0], 5, probe.INPUTS, probe.POOLINGS)
    assert not crops.publishable([0, 1, 2], 3, probe.INPUTS, probe.POOLINGS)
    assert not crops.publishable([0, 1, 2], 5, ("stacked",), probe.POOLINGS)
