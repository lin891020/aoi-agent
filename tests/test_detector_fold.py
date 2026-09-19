"""The fold experiment changes one variable, and says so when it does not.

Five ways it could report something other than what it claims, none of which
raises:

* it writes over `data/patches_pcbaoi`, and the published entries' candidate
  set silently becomes this weaker detector's;
* the two arms read different test candidates, so the comparison is two queues;
* the arms share a board, so "seen" and "unseen" overlap;
* the entry quotes the published tables, which were produced by a detector
  trained on twice the boards;
* the arms differ in size and the entry does not say so.

Nothing here trains a detector or reads an image: the arms are built from
hand-made `PatchSet`s and the detector is a stub.
"""

from __future__ import annotations

from pathlib import Path

import detector_fold_report as fold
import numpy as np
import pytest

from aoi_agent.vision.patches import PatchSet

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Bad_podu", "Bad_qiaojiao", "false_call"]


def _patch_set(n: int, images: int) -> PatchSet:
    rng = np.random.default_rng(0)
    return PatchSet(
        patches=rng.integers(0, 256, size=(n, 3, 64, 64), dtype=np.uint8),
        labels=np.array([2, 0, 1] * (n // 3), dtype=np.int64)[:n],
        label_names=CLASSES, image_index=np.repeat(np.arange(images), n // images),
        boxes=np.zeros((n, 4), dtype=np.int64))


# ---- the verdict was written before the run ------------------------------------


@pytest.mark.parametrize(("in_sample", "out_of_sample", "label"), [
    (0.0067, 0.0045, "the_shift"),
    (0.0051, 0.0050, "the_shift"),
    (0.0067, 0.0060, "part_of_it"),
    (0.0067, 0.0067, "not_the_shift"),
    (0.0060, 0.0080, "not_the_shift"),
    (0.0040, 0.0030, "nothing_to_attribute"),   # in-sample held the budget: no shift here
    (0.0050, 0.0060, "nothing_to_attribute"),
    (None, 0.0045, "unreadable"),
    (0.0067, None, "unreadable"),
])
def test_the_verdict_is_the_pre_registered_ladder(in_sample, out_of_sample, label):
    assert fold.verdict(in_sample, out_of_sample)[0] == label


def test_the_budget_and_the_published_worst_escape_are_the_published_ones():
    """The 2026-09-16 entry's own worst deployable escape, and the budget it missed."""
    benchmarks = (ROOT / "docs" / "benchmarks.md").read_text()
    assert f"| {fold.PUBLISHED_WORST_ESCAPE:.2%} |" in benchmarks
    assert fold.BUDGET == 0.005


def test_only_the_pre_registered_configuration_may_append():
    assert fold.publishable([2, 1, 0], 10, 60)
    assert not fold.publishable([0], 10, 60)
    assert not fold.publishable([0, 1, 2], 1, 60)
    assert not fold.publishable([0, 1, 2], 10, 1)


# ---- the published candidate set is not touched --------------------------------


def test_the_published_patches_are_a_different_directory():
    assert fold.PATCHES.resolve() != (ROOT / "data" / "patches_pcbaoi").resolve()
    assert fold.CHECKPOINT.name != "detector_pcbaoi.pt"
    assert fold.OUT_JSON.name not in {"dinov2_crops.json", "detector_history.json"}


def test_a_run_pointed_at_the_published_patches_refuses(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv", ["detector_fold_report.py", "--patches", "data/patches_pcbaoi", "--dry-run"])
    assert fold.main() == 2
    assert "refusing to write over" in capsys.readouterr().err


# ---- one variable, and a warning when it is two --------------------------------


class _StubDetector:
    """Counts what it was asked to look at; the crops themselves come from the stub build."""

    def __init__(self):
        self.seen = []


def test_both_arms_are_written_the_same_test_crops(tmp_path, monkeypatch):
    test_set, arms = _patch_set(24, 4), _patch_set(48, 8)
    monkeypatch.setattr(fold.crops_build, "build", lambda *_a, **_k: (
        test_set if not _a[1] else arms, {"floor": np.array(0.01)}))
    monkeypatch.setattr(fold, "items_for", lambda stems: [f"item-{s}" for s in stems])
    monkeypatch.setattr(fold.pcbaoi, "load", lambda _subset: [])
    monkeypatch.setattr(fold.pcbaoi, "leaks", lambda _a, _b: set())

    built = fold.build_arms(_StubDetector(), {"in_sample": ["a"], "out_of_sample": ["b"]},
                            tmp_path, 0.01, log=lambda *_: None)

    for arm in fold.ARMS:
        assert np.array_equal(PatchSet.load(tmp_path / arm / "test.npz").labels, test_set.labels)
    assert set(built) == {"test", *fold.ARMS}


def test_a_detector_that_drew_nothing_is_named_as_the_problem(tmp_path, monkeypatch):
    empty = PatchSet(patches=np.zeros((0, 3, 64, 64), dtype=np.uint8),
                     labels=np.zeros(0, dtype=np.int64), label_names=CLASSES,
                     image_index=np.zeros(0, dtype=np.int64),
                     boxes=np.zeros((0, 4), dtype=np.int64))
    monkeypatch.setattr(fold.crops_build, "build", lambda *_a, **_k: (empty, {}))
    monkeypatch.setattr(fold, "items_for", lambda stems: [])
    monkeypatch.setattr(fold.pcbaoi, "load", lambda _subset: [])
    monkeypatch.setattr(fold.pcbaoi, "leaks", lambda _a, _b: set())
    with pytest.raises(SystemExit, match="drew no box"):
        fold.build_arms(_StubDetector(), {"in_sample": ["a"], "out_of_sample": ["b"]},
                        tmp_path, 0.01, log=lambda *_: None)


def test_a_board_in_both_arms_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(fold, "items_for", lambda stems: [])
    monkeypatch.setattr(fold.pcbaoi, "load", lambda _subset: [])
    monkeypatch.setattr(fold.pcbaoi, "leaks", lambda _a, _b: set())
    with pytest.raises(SystemExit, match="share a board"):
        fold.build_arms(_StubDetector(), {"in_sample": ["a"], "out_of_sample": ["a"]},
                        tmp_path, 0.01, log=lambda *_: None)


def test_a_test_board_inside_a_fold_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(fold, "items_for", lambda stems: [])
    monkeypatch.setattr(fold.pcbaoi, "load", lambda _subset: [])
    monkeypatch.setattr(fold.pcbaoi, "leaks", lambda _a, _b: {"shared"})
    with pytest.raises(SystemExit, match="test board"):
        fold.build_arms(_StubDetector(), {"in_sample": ["a"], "out_of_sample": ["b"]},
                        tmp_path, 0.01, log=lambda *_: None)


def test_the_unseen_arm_is_cut_to_the_size_of_the_seen_one():
    unseen = [f"b{i:02d}" for i in range(86)]
    drawn = fold.matched(unseen, to=70)
    assert len(drawn) == 70 and set(drawn) <= set(unseen)
    assert drawn == fold.matched(unseen, to=70)          # deterministic for the fold seed
    assert fold.matched(unseen, to=200) == sorted(unseen)   # never invents boards


@pytest.mark.parametrize(("sizes", "gap"), [
    ({"in_sample": 100, "out_of_sample": 100}, 0.0),
    ({"in_sample": 100, "out_of_sample": 80}, 0.2),
    ({"in_sample": 40, "out_of_sample": 100}, 0.6),
])
def test_the_size_gap_is_measured(sizes, gap):
    assert fold.size_gap(sizes) == pytest.approx(gap)


# ---- the entry is about this detector's queue ----------------------------------


def _record(in_sample_escape, out_escape, patches) -> dict:
    rows = [{"arm": arm, "seed": seed, "oracle_review_reduction": 0.012,
             "oracle_escape_rate": 0.0045, "test_review_reduction": 0.009,
             "test_escape_rate": escape}
            for arm, escape in (("in_sample", in_sample_escape), ("out_of_sample", out_escape))
            for seed in (0, 1, 2)]
    return {
        "date": "2026-09-20", "commit": "abc1234", "device": "mps", "budget": 0.005,
        "boards": {"in_sample": 24, "out_of_sample": 30}, "patches": patches,
        "size_gap": fold.size_gap(patches), "test_labels": [0] * 512,
        "median_escape": {"in_sample": in_sample_escape, "out_of_sample": out_escape},
        "verdict": fold.verdict(in_sample_escape, out_escape)[0],
        "verdict_sentence": fold.verdict(in_sample_escape, out_escape)[1],
        "rows": rows, "wall_minutes": 72.0,
    }


def test_the_entry_quotes_no_published_crop_figure():
    body = fold.render(_record(0.0067, 0.0060, {"in_sample": 5000, "out_of_sample": 4800}))
    for published in ("578", "2.8%", "22.3%", "55.6%"):
        assert published not in body
    assert "may be read against the 2026-08-28 or 2026-09-16 tables" in body


def test_an_unbalanced_pair_of_arms_says_so_in_the_entry():
    balanced = fold.render(_record(0.0067, 0.0060, {"in_sample": 5000, "out_of_sample": 4800}))
    lopsided = fold.render(_record(0.0067, 0.0060, {"in_sample": 5000, "out_of_sample": 2000}))
    assert "second variable" not in balanced
    assert "second variable" in lopsided


def test_the_entry_carries_both_arms_and_the_verdict():
    body = fold.render(_record(0.0067, 0.0045, {"in_sample": 5000, "out_of_sample": 4800}))
    assert "in-sample" in body and "out-of-sample" in body
    assert "The in-sample boxes were the shift" in body
    assert "(over budget)" in body      # the in-sample rows, flagged against the budget
