"""Deep prompt tuning on the detector's crops reads this queue, not DeepPCB's.

Five ways this arm could report something other than what it claims, none of
which raises:

* the entry quotes a DeepPCB figure, and a reader compares two different queues;
* the bootstrap pairs one model's seed with another seed's, so the difference
  carries two training splits instead of one;
* the verdict's comparators drift from the entry they were read from;
* the test scores are not row-aligned with the patches the bootstrap resamples;
* a shorter run appends, so the published rows are not the registered recipe.

No weights are downloaded: the backbone is the stub from `test_dinov2_vpt.py`.
"""

from __future__ import annotations

from pathlib import Path

import dinov2_crops_report as crops
import dinov2_vpt as vpt
import dinov2_vpt_crops as vptc
import numpy as np
import pytest
import torch
from test_dinov2_vpt import DIM, _StubBackbone

from aoi_agent.vision.patches import PatchSet

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Bad_podu", "Bad_qiaojiao", "false_call"]


def _interval(low: float, high: float, median: float | None = None) -> dict:
    return {"low": low, "high": high, "resamples": 2000,
            "median": (low + high) / 2 if median is None else median}


# ---- the verdict was written before the run ------------------------------------


@pytest.mark.parametrize(("low", "high", "label"), [
    (0.004, 0.030, "helps"),
    (0.000001, 0.02, "helps"),
    (-0.017, 0.015, "indistinguishable"),
    (0.0, 0.02, "indistinguishable"),
    (-0.02, 0.0, "indistinguishable"),
    (-0.030, -0.004, "worse"),
])
def test_the_verdict_is_the_sign_of_the_paired_interval(low, high, label):
    assert vptc.verdict(_interval(low, high), 0.05, 0.05)[0] == label


def test_an_interval_in_fractions_is_reported_in_points():
    _label, sentence = vptc.verdict(_interval(-0.017, 0.015), 0.05, 0.05)
    assert "-1.7 to +1.5 points" in sentence


def test_two_orderings_under_one_percent_are_called_not_orderings():
    _label, weak = vptc.verdict(_interval(-0.017, 0.015), 0.009, 0.009)
    _label, strong = vptc.verdict(_interval(-0.017, 0.015), 0.009, 0.012)
    assert "neither ordering is one" in weak and "0.3%" in weak
    assert "neither ordering is one" not in strong


def test_the_comparators_are_this_queues_published_figures():
    benchmarks = (ROOT / "docs" / "benchmarks.md").read_text()
    assert (f"ResNet-18 {vptc.RESNET_ORACLE_MIN:.1%}–{vptc.RESNET_ORACLE_MAX:.1%}, "
            f"DINOv2 {vptc.PROBE_ORACLE_MIN:.1%}–{vptc.PROBE_ORACLE_MAX:.1%}") in benchmarks
    assert vptc.PROBE_ORACLE_MIN <= vptc.PROBE_ORACLE_MEDIAN <= vptc.PROBE_ORACLE_MAX


def test_only_the_pre_registered_configuration_may_append():
    assert vptc.publishable([2, 1, 0], 10, 10, 1e-3, 64)
    assert not vptc.publishable([0], 10, 10, 1e-3, 64)
    assert not vptc.publishable([0, 1, 2], 1, 10, 1e-3, 64)
    assert not vptc.publishable([0, 1, 2], 10, 10, 3e-4, 64)


def test_the_recipe_is_the_deeppcb_runs_recipe_and_not_a_second_one():
    assert (vptc.PATCHES.name, vptc.OUT_JSON.name) == ("patches_pcbaoi", "dinov2_vpt_crops.json")
    assert vptc.OUT_JSON != crops.OUT_JSON
    assert vptc.OUT_JSON.name != "dinov2_vpt.json"


# ---- the bootstrap reads one seed of each model, and it is the same seed --------


@pytest.mark.parametrize(("oracles", "expected"), [
    ([0.003, 0.012, 0.028], 1),
    ([0.028, 0.012, 0.003], 1),
    ([0.012, 0.012, 0.012], 1),
    ([None, 0.012, 0.028], 1),
    ([None, None, 0.028], 1),
])
def test_the_bootstrap_reads_the_median_seed(oracles, expected):
    rows = [{"seed": seed, "oracle_review_reduction": value}
            for seed, value in enumerate(oracles)]
    assert vptc.median_seed(rows) == expected


def test_a_seed_without_a_threshold_does_not_drop_out_of_the_ranking():
    # Dropping it would make seed 2 the median of two rows, which is a different seed.
    rows = [{"seed": 0, "oracle_review_reduction": None},
            {"seed": 1, "oracle_review_reduction": 0.012},
            {"seed": 2, "oracle_review_reduction": 0.028}]
    assert vptc.median_seed(rows) == 1
    assert vptc.median_seed([r for r in rows if r["oracle_review_reduction"] is not None]) == 2


# ---- one seed, end to end, on a stub backbone ----------------------------------


def _patch_set(n: int, images: int, rng) -> PatchSet:
    labels = np.array([2, 0, 1] * (n // 3), dtype=np.int64)[:n]
    return PatchSet(
        patches=rng.integers(0, 256, size=(n, 3, 64, 64), dtype=np.uint8),
        labels=labels, label_names=CLASSES,
        image_index=np.repeat(np.arange(images), n // images),
        boxes=np.zeros((n, 4), dtype=np.int64))


class _Args:
    prompts, epochs, batch_size, lr, budget = 3, 1, 8, 1e-3, 0.005


def test_a_seed_trains_a_three_class_head_and_scores_every_test_candidate(monkeypatch):
    monkeypatch.setattr(vptc.probe, "load_backbone", lambda _device: _StubBackbone())
    rng = np.random.default_rng(0)
    trainval, test = _patch_set(48, 8, rng), _patch_set(24, 4, rng)

    row = vptc.one_seed(trainval, test, 1, _Args(), torch.device("cpu"), log=lambda *_: None)

    assert row["seed"] == 1 and row["backbone_unchanged"]
    assert row["trainable_parameters"] == 3 * 3 * DIM + DIM * 3 + 3
    assert len(row["test_scores"]) == len(test.labels)
    assert set(row) >= {"oracle_review_reduction", "test_review_reduction", "best_epoch"}


def test_test_scores_that_are_not_row_aligned_are_refused(monkeypatch):
    monkeypatch.setattr(vptc.probe, "load_backbone", lambda _device: _StubBackbone())
    rng = np.random.default_rng(0)
    trainval, test = _patch_set(48, 8, rng), _patch_set(24, 4, rng)
    shuffled = test.labels[::-1].copy()      # a real reordering, not a period of the pattern

    real_predict = vptc.predict

    def predict_in_another_order(model, loader, device):
        probabilities, _labels = real_predict(model, loader, device)
        return probabilities, shuffled

    monkeypatch.setattr(vptc, "predict", predict_in_another_order)
    with pytest.raises(SystemExit, match="row-aligned"):
        vptc.one_seed(trainval, test, 1, _Args(), torch.device("cpu"), log=lambda *_: None)


# ---- the entry is about this queue ---------------------------------------------


def _record(vpt_oracles, resnet_oracles, interval) -> dict:
    def rows(values, extra):
        return [{"seed": seed, "oracle_review_reduction": value, "oracle_escape_rate": 0.0045,
                 "test_review_reduction": 0.012, "test_escape_rate": 0.0067,
                 "trainable_parameters": 48775, **extra}
                for seed, value in enumerate(values)]
    return {
        "date": "2026-09-18", "commit": "abc1234", "device": "mps", "epochs": 10,
        "basis": {"images": 60, "candidates": 578, "defects": 449, "false_calls": 129,
                  "prevalence": 0.777},
        "wall_minutes": 140.0, "bootstrap": interval, "bootstrap_seed_read": 1,
        "vpt_median_oracle": sorted(vpt_oracles)[1],
        "resnet_median_oracle": sorted(resnet_oracles)[1],
        "verdict": "indistinguishable",
        "verdict_sentence": vptc.verdict(interval, sorted(vpt_oracles)[1],
                                         sorted(resnet_oracles)[1])[1],
        "seeds": rows(vpt_oracles, {"best_epoch": 7, "backbone_unchanged": True}),
        "resnet_rows": rows(resnet_oracles, {}),
    }


def test_the_entry_quotes_no_deeppcb_figure():
    body = vptc.render(_record([0.007, 0.010, 0.012], [0.003, 0.012, 0.028],
                               _interval(-0.017, 0.015)))
    for deeppcb in ("7,322", "3,018", f"{vpt.RESNET_ORACLE_MIN:.2%}", "0.663%"):
        assert deeppcb not in body
    assert "578 candidates" in body and "60 test images" in body
    assert "may be read against the DeepPCB curve" in body


def test_the_entry_carries_the_basis_the_verdict_and_both_arms():
    body = vptc.render(_record([0.007, 0.010, 0.012], [0.003, 0.012, 0.028],
                               _interval(-0.017, 0.015)))
    assert "77.7% genuine defects" in body and "22.3%" in body
    assert "Indistinguishable" in body
    assert "ResNet-18 (2026-09-16 checkpoints)" in body
    assert f"{vptc.PROBE_ORACLE_MIN:.1%}–{vptc.PROBE_ORACLE_MAX:.1%}" in body
    assert "Read on seed 1" in body
