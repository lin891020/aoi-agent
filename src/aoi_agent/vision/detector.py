"""The detector front end: boxes and scores in one step, no template.

On DeepPCB a candidate is a differencing blob and the re-verifier supplies its
score. On PCB-AoI there is no template to difference against, so the detector
supplies both the box and the score -- and the project's accounting runs over
its output unchanged: a candidate's ``P(false_call)`` is ``1 - confidence``,
the operating point is swept over that with `vision.operating_point.sweep`
against ground truth via `aoi.matching.match`, and a defect no box covers at
the floor is *unflagged*, the outcome the escape accounting already names.

This module is deliberately thin. It owns three decisions and nothing else:

* the confidence **floor** below which a box is not a candidate at all
  (``CONF_FLOOR``, low on purpose -- the sweep decides where to cut, and a
  floor high enough to be a decision would be a second threshold nobody
  swept);
* the definition of ``P(false_call)`` for a box, which is the one line the
  whole comparison rests on;
* the mapping from the detector's class index back to the dataset's own
  names, which are never DeepPCB's.

Everything about training lives in `scripts/train_detector.py`; everything
about scoring lives in `scripts/detector_report.py`. Importing ``ultralytics``
happens inside the loader so the rest of the package does not pay for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[3] / "models" / "detector_pcbaoi.pt"

#: A box below this confidence is not a candidate. 0.01, not 0.25: the point
#: of the sweep is to find the cut, and every box the head emits is evidence
#: about where that cut should sit. What the floor removes is the noise below
#: any conceivable operating point, so the queue is not ten thousand boxes a
#: board.
CONF_FLOOR = 0.01


@dataclass(frozen=True)
class ScoredCandidate:
    """A detector box: where, what the detector thinks it is, and how sure."""

    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str
    confidence: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def false_call_probability(self) -> float:
        """The one definition. ``1 - confidence``: a box the detector is 0.9
        sure is a defect is a box it is 0.1 sure is nothing. The sweep reads
        this the way it reads the re-verifier's own ``P(false_call)``."""
        return 1.0 - self.confidence


class Detector:
    """Loads a trained YOLO checkpoint once and detects on numpy images."""

    def __init__(self, checkpoint: Path | str = DEFAULT_CHECKPOINT, device: str | None = None):
        from ultralytics import YOLO

        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"{checkpoint} not found. Train the detector first:\n"
                "  uv run python scripts/train_detector.py"
            )
        self.checkpoint = checkpoint
        self.model = YOLO(str(checkpoint))
        self.device = device
        self.class_names: dict[int, str] = dict(self.model.names)

    def detect(self, image: np.ndarray, floor: float = CONF_FLOOR) -> list[ScoredCandidate]:
        """Every box at or above the floor, highest confidence first.

        ``image`` is HxWx3 RGB or HxW grey; the detector was trained on RGB
        and a grey frame is expanded rather than refused, because the AOI
        images this is for are colour and the expansion is only here so a
        test can hand it a synthetic frame.
        """
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        results = self.model.predict(
            image[:, :, ::-1],  # ultralytics reads BGR from arrays, as cv2 does
            conf=floor, verbose=False, device=self.device,
        )
        out: list[ScoredCandidate] = []
        for result in results:
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                out.append(ScoredCandidate(
                    int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)),
                    self.class_names[int(k)], float(c),
                ))
        out.sort(key=lambda c: -c.confidence)
        return out


def false_call_probabilities(candidates: list[ScoredCandidate]) -> np.ndarray:
    """The array the operating-point sweep takes, in candidate order."""
    return np.array([c.false_call_probability for c in candidates], dtype=np.float64)
