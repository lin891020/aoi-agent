"""The detector front end's three decisions, held without ultralytics.

`vision.detector` owns exactly three things: the confidence floor, the
definition of ``P(false_call)`` for a box, and the class-name mapping. The
first two are what the whole DeepPCB-vs-PCB-AoI comparison rests on, so they
are pinned here on a fake model rather than on weights nobody has in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from aoi_agent.aoi.matching import match
from aoi_agent.vision import detector as det
from aoi_agent.vision.detector import Detector, ScoredCandidate, false_call_probabilities
from aoi_agent.vision.operating_point import best_at_escape_budget, sweep


def test_p_false_call_is_one_minus_confidence():
    """The one line the comparison rests on."""
    assert ScoredCandidate(0, 0, 10, 10, "Bad_podu", 0.9).false_call_probability == pytest.approx(0.1)
    assert ScoredCandidate(0, 0, 10, 10, "Bad_podu", 0.0).false_call_probability == 1.0


def test_the_floor_is_below_any_operating_point_anyone_would_ship():
    assert det.CONF_FLOOR <= 0.05


def test_a_scored_candidate_matches_like_a_differencing_candidate():
    """The matcher reads ``box`` and the escape accounting reads ``area``; a
    detector box has to walk into both unchanged."""
    from aoi_agent.data.deeppcb import Annotation

    cands = [ScoredCandidate(10, 10, 30, 30, "Bad_podu", 0.8)]
    result = match(cands, [Annotation(12, 12, 28, 28, 1)])
    assert len(result.detected_annotations) == 1
    assert cands[0].area == 400


def test_the_sweep_reads_detector_scores_the_way_it_reads_the_reverifiers():
    """Three defects the detector scored high, two false calls it scored low:
    there is a threshold that dismisses both false calls and no defect."""
    cands = [
        ScoredCandidate(0, 0, 1, 1, "Bad_podu", 0.95),
        ScoredCandidate(0, 0, 1, 1, "Bad_podu", 0.80),
        ScoredCandidate(0, 0, 1, 1, "Bad_qiaojiao", 0.70),
        ScoredCandidate(0, 0, 1, 1, "Bad_podu", 0.10),
        ScoredCandidate(0, 0, 1, 1, "Bad_podu", 0.02),
    ]
    labels = np.array([0, 0, 1, 2, 2])         # 2 == false_call
    points = sweep(false_call_probabilities(cands), labels, false_call_index=2)
    best = best_at_escape_budget(points, 0.0)
    assert best is not None
    assert best.escapes == 0 and best.false_calls_dismissed == 2


class _FakeBoxes:
    def __init__(self, rows):
        import torch

        self.xyxy = torch.tensor([r[:4] for r in rows], dtype=torch.float32)
        self.conf = torch.tensor([r[4] for r in rows], dtype=torch.float32)
        self.cls = torch.tensor([r[5] for r in rows], dtype=torch.float32)


class _FakeResult:
    def __init__(self, rows):
        self.boxes = _FakeBoxes(rows)


class _FakeYOLO:
    names = {0: "Bad_podu", 1: "Bad_qiaojiao"}

    def __init__(self):
        self.seen = None

    def predict(self, image, conf, verbose, device):
        self.seen = (image.shape, conf)
        return [_FakeResult([
            [10.2, 10.6, 30.4, 30.1, 0.30, 0],
            [40.0, 40.0, 50.0, 55.0, 0.90, 1],
        ])]


@pytest.fixture
def fake_detector(monkeypatch, tmp_path):
    ckpt = tmp_path / "d.pt"
    ckpt.write_bytes(b"not weights")
    fake = _FakeYOLO()

    import types

    monkeypatch.setitem(__import__("sys").modules, "ultralytics", types.SimpleNamespace(YOLO=lambda path: fake))
    d = Detector(ckpt)
    return d, fake


def test_detect_returns_boxes_highest_confidence_first_with_the_datasets_names(fake_detector):
    d, fake = fake_detector
    out = d.detect(np.zeros((600, 600, 3), dtype=np.uint8))
    assert [c.confidence for c in out] == pytest.approx([0.9, 0.3], abs=1e-6)
    assert out[0].class_name == "Bad_qiaojiao" and out[0].box == (40, 40, 50, 55)
    assert out[1].box == (10, 11, 30, 30)
    assert fake.seen[1] == det.CONF_FLOOR


def test_a_grey_frame_is_expanded_not_refused(fake_detector):
    d, fake = fake_detector
    d.detect(np.zeros((64, 64), dtype=np.uint8))
    assert fake.seen[0] == (64, 64, 3)


def test_a_missing_checkpoint_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_detector"):
        Detector(tmp_path / "nope.pt")
