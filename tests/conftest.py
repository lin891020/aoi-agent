import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from aoi_agent.aoi.simulator import Candidate
from aoi_agent.data.deeppcb import Annotation


@pytest.fixture
def blank_board() -> np.ndarray:
    """A 128x128 board with some copper on it."""
    board = np.zeros((128, 128), dtype=np.uint8)
    board[20:100, 30:40] = 255   # a vertical trace
    board[20:30, 30:100] = 255   # a horizontal trace
    return board


def make_candidate(x1, y1, x2, y2, area=None) -> Candidate:
    return Candidate(x1, y1, x2, y2, area if area is not None else (x2 - x1) * (y2 - y1))


def make_annotation(x1, y1, x2, y2, class_id=1) -> Annotation:
    return Annotation(x1, y1, x2, y2, class_id)
