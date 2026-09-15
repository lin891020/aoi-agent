"""The DINOv2 probe is read by the same procedure as the re-verifier, or not at all.

Four ways this experiment could report a number that means something other than
what it says, each silent:

* the configuration that becomes DINOv2's result is picked by its test column,
  which is choosing on test with eight tries;
* a fold's probe sees an image it is later asked to predict, or the threshold
  is chosen by a paraphrase of `threshold_cv.choose` rather than the thing;
* a pooling reads the wrong tokens -- the centre 4x4 window indexes a grid, and
  an off-by-one there raises nothing;
* the per-channel input puts the three images in a different order than the
  features are concatenated in.

No weights are downloaded and no dataset is read: the backbone is a stub whose
outputs are known.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import dinov2_probe as probe
import numpy as np
import pytest
import threshold_cv
import torch

from aoi_agent.vision.operating_point import OperatingPoint
from aoi_agent.vision.patches import PatchSet

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["false_call", "open", "short", "mousebite", "spur", "copper", "pin-hole"]


def _point(review: float) -> OperatingPoint:
    return OperatingPoint(threshold=0.9, escape_rate=0.004, review_reduction=review,
                          reviewed=0, dismissed=0, escapes=0, defects_total=1,
                          false_calls_dismissed=0, false_calls_total=1)


def _synthetic(images: int = 40, per_image: int = 6, dims: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = images * per_image
    labels = np.where(rng.random(n) < 0.55, 0, rng.integers(1, len(NAMES), n))
    x = rng.normal(size=(n, dims)).astype(np.float32)
    x[:, 0] += (labels == 0) * 2.0
    patch_set = PatchSet(
        patches=np.zeros((n, 3, 64, 64), dtype=np.uint8), labels=labels.astype(np.int64),
        label_names=NAMES, image_index=np.repeat(np.arange(images), per_image),
        boxes=np.zeros((n, 4), dtype=np.int64),
    )
    return patch_set, x


# ---- the result is chosen without test -----------------------------------


def test_the_configuration_is_chosen_from_out_of_fold_results_alone():
    """By signature: the function cannot be handed a test column."""
    assert list(inspect.signature(probe.select_configuration).parameters) == ["oof_guarded"]
    chosen = probe.select_configuration({
        ("per_channel", "cls"): _point(0.60),
        ("stacked", "mean"): _point(0.40),
        ("stacked", "cls"): None,
    })
    assert chosen == ("per_channel", "cls")


def test_a_tie_keeps_the_earlier_configuration():
    chosen = probe.select_configuration({
        ("per_channel", "mean"): _point(0.50), ("stacked", "cls"): _point(0.50)})
    assert chosen == ("per_channel", "mean")


def test_no_configuration_inside_the_budget_is_no_result():
    assert probe.select_configuration({("stacked", "cls"): None}) is None


# ---- the procedure is the shipped one ------------------------------------


def test_the_selection_pieces_are_threshold_cvs_own_not_copies():
    assert probe.choose is threshold_cv.choose
    assert probe.folds_by_image is threshold_cv.folds_by_image
    assert probe.inner_split is threshold_cv.inner_split
    assert probe.indices_for is threshold_cv.indices_for


def test_no_fold_probe_trains_on_an_image_it_predicts(monkeypatch):
    patch_set, x = _synthetic()
    monkeypatch.setattr(probe, "C_GRID", (1.0,))
    trained_on: list[set] = []
    real = probe.choose_c

    def spy(x_, labels, train_idx, val_idx, **kwargs):
        trained_on.append(set(patch_set.image_index[train_idx])
                          | set(patch_set.image_index[val_idx]))
        return real(x_, labels, train_idx, val_idx, **kwargs)

    monkeypatch.setattr(probe, "choose_c", spy)
    oof, _optimistic, _guarded, cs = probe.oof_select(
        x, patch_set, folds=5, budget=0.005, seed=0, log=lambda *_: None)

    held_out = threshold_cv.folds_by_image(patch_set, 5, 0)
    assert len(trained_on) == 5 and cs == [1.0] * 5
    for images, held in zip(trained_on, held_out, strict=True):
        assert not images & set(held)
    assert np.allclose(oof.sum(axis=1), 1.0, atol=1e-5)


def test_the_class_weights_are_the_networks():
    from aoi_agent.vision.dataset import class_weights

    patch_set, _x = _synthetic()
    network = class_weights(patch_set).numpy()
    ours = probe.inverse_frequency(patch_set.labels, len(NAMES))
    assert np.allclose([ours[i] for i in range(len(NAMES))], network, atol=1e-6)


def test_a_class_the_fit_never_saw_gets_a_zero_column():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 20)
    x = rng.normal(size=(40, 3)) + y[:, None]
    model = probe.fit_probe(x, y, 1.0, 0, len(NAMES))
    p = probe.probabilities(model, x, len(NAMES))
    assert p.shape == (40, len(NAMES))
    assert np.all(p[:, 2:] == 0) and np.allclose(p.sum(axis=1), 1.0)


# ---- inputs and poolings --------------------------------------------------


def _unnormalise(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(probe.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(probe.IMAGENET_STD).view(1, 3, 1, 1)
    return x * std + mean


def _constant_channels() -> np.ndarray:
    batch = np.zeros((2, 3, 64, 64), dtype=np.uint8)
    batch[:, 0], batch[:, 1], batch[:, 2] = 0, 128, 255
    return batch


def test_per_channel_input_is_template_test_difference_in_that_order():
    raw = _unnormalise(probe.prepare(_constant_channels(), "per_channel"))
    assert raw.shape == (6, 3, probe.INPUT_SIZE, probe.INPUT_SIZE)
    expected = [0.0, 128 / 255, 1.0] * 2
    for image, value in zip(raw, expected, strict=True):
        assert torch.allclose(image, torch.full_like(image, value), atol=1e-4)


def test_stacked_input_keeps_the_three_as_rgb():
    raw = _unnormalise(probe.prepare(_constant_channels(), "stacked"))
    assert raw.shape == (2, 3, probe.INPUT_SIZE, probe.INPUT_SIZE)
    for channel, value in enumerate([0.0, 128 / 255, 1.0]):
        assert torch.allclose(raw[:, channel], torch.full_like(raw[:, channel], value), atol=1e-4)


def test_an_unknown_input_mode_is_refused():
    with pytest.raises(ValueError):
        probe.prepare(_constant_channels(), "grey")


def _tokens(m: int = 2, side: int = 16):
    tokens = np.zeros((m, side * side, 3), dtype=np.float32)
    for r in range(side):
        for c in range(side):
            tokens[:, r * side + c, 0] = r
            tokens[:, r * side + c, 1] = c
            tokens[:, r * side + c, 2] = 1.0 if 6 <= r <= 9 and 6 <= c <= 9 else 0.0
    cls = np.full((m, 3), 7.0, dtype=np.float32)
    return cls, tokens


def test_the_four_poolings_read_the_tokens_they_name():
    cls, tokens = _tokens()
    pooled = probe.pool(cls, tokens)
    assert np.array_equal(pooled["cls"], cls)
    assert np.allclose(pooled["mean"], [7.5, 7.5, 16 / 256])
    assert pooled["cls_mean"].shape == (2, 6)
    assert np.allclose(pooled["cls_mean"][:, :3], 7.0)
    assert np.allclose(pooled["centre"], [7.5, 7.5, 1.0])


def test_the_device_pooling_matches_the_extraction_pooling():
    cls, tokens = _tokens()
    pooled = probe.pool(cls, tokens)
    for name in probe.POOLINGS:
        on_device = probe.pool_torch(torch.from_numpy(cls), torch.from_numpy(tokens), name)
        assert np.allclose(on_device.numpy(), pooled[name])


def test_a_token_count_that_is_not_a_grid_is_refused():
    with pytest.raises(ValueError):
        probe.pool(np.zeros((1, 3)), np.zeros((1, 250, 3)))


class _StubBackbone:
    """Every token of an image carries that image's mean input value."""

    def forward_features(self, x):
        value = x.mean(dim=(1, 2, 3))
        return {"x_norm_clstoken": value[:, None].repeat(1, 4),
                "x_norm_patchtokens": value[:, None, None].repeat(1, 256, 4)}


def test_per_channel_features_concatenate_in_input_order():
    features = probe.extract(_constant_channels(), _StubBackbone(), "per_channel",
                             torch.device("cpu"), batch_size=2, log=lambda *_: None)
    cls = features["cls"]
    assert cls.shape == (2, 12)
    assert np.all(cls[:, :4] < cls[:, 4:8]) and np.all(cls[:, 4:8] < cls[:, 8:])


def test_stacked_features_are_one_image_per_patch():
    features = probe.extract(_constant_channels(), _StubBackbone(), "stacked",
                             torch.device("cpu"), batch_size=1, log=lambda *_: None)
    assert {name: values.shape for name, values in features.items()} == {
        "cls": (2, 4), "mean": (2, 4), "cls_mean": (2, 8), "centre": (2, 4)}


# ---- the verdict was written before the run -------------------------------


def _rows(review: float, escape: float) -> list[dict]:
    return [{"test_review_reduction": review, "test_escape_rate": escape}] * 3


@pytest.mark.parametrize(("review", "escape", "label"), [
    (0.60, 0.005, "helps"),
    (0.60, 0.00663, "helps"),
    (0.60, 0.007, "no_help"),
    (0.5559, 0.004, "no_help"),
    (0.52, 0.004, "no_help"),
    (0.49, 0.004, "no_help"),
    (0.45, 0.004, "worse"),
    (0.10, 0.004, "below_floor"),
])
def test_the_verdict_is_the_pre_registered_rule(review, escape, label):
    assert probe.verdict(_rows(review, escape))[0] == label


def test_the_comparators_are_the_published_figures():
    """The rule compares against numbers a reader can find in the runs they name."""
    benchmarks = (ROOT / "docs" / "benchmarks.md").read_text()
    assert f"{probe.RESNET_DEPLOYED_MIN:.2%}–{probe.RESNET_DEPLOYED_MAX:.2%}" in benchmarks
    assert f"{probe.RESNET_ORACLE_MIN:.2%}–{probe.RESNET_ORACLE_MAX:.2%}" in benchmarks
    assert f"{probe.RESNET_ESCAPE_MIN:.3%}–{probe.RESNET_ESCAPE_MAX:.3%}" in benchmarks
    assert f"**{probe.RESNET_DEPLOYED_MEDIAN:.2%}**" in benchmarks
    assert f"**{probe.FEATURE_FLOOR:.2%}**" in benchmarks


def test_only_the_pre_registered_run_may_append():
    assert probe.publishable([2, 0, 1], 5, probe.INPUTS, probe.POOLINGS)
    assert not probe.publishable([0], 5, probe.INPUTS, probe.POOLINGS)
    assert not probe.publishable([0, 1, 2], 3, probe.INPUTS, probe.POOLINGS)
    assert not probe.publishable([0, 1, 2], 5, ("stacked",), probe.POOLINGS)
