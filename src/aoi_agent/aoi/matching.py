"""Label AOI candidates against ground truth.

A candidate that overlaps a real defect is a true detection and carries that
defect's class. A candidate that overlaps nothing is a **false call** — the
thing the re-verification model exists to dismiss.

DeepPCB's own benchmark counts a detection as correct at IoU > 0.33, so the
same threshold is used here to stay comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

from aoi_agent.aoi.simulator import Candidate
from aoi_agent.data.deeppcb import CLASS_NAMES, FALSE_CALL, Annotation

IOU_THRESHOLD = 0.33

#: Candidates this close to a real defect are treated as ambiguous rather than
#: as false calls. Measured on 100 trainval images: 6.1% of unmatched
#: candidates overlap or touch a real defect and a further 8.7% sit in the halo
#: around one. Those are fragments of a defect the detector split into several
#: blobs, not spurious flags, and training on them as ``false_call`` would
#: teach the model to dismiss real defects.
FRAGMENT_GAP_PX = 20

FRAGMENT = "fragment"

Box = tuple[int, int, int, int]


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two axis-aligned boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    if intersection == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def coverage(candidate: Box, target: Box) -> float:
    """Fraction of ``target`` that ``candidate`` covers.

    Reported alongside IoU because a candidate can fully contain a small defect
    while scoring poorly on IoU. For the AOI stage "did the flag land on the
    defect" matters more than how tightly it fits.
    """
    ix1, iy1 = max(candidate[0], target[0]), max(candidate[1], target[1])
    ix2, iy2 = min(candidate[2], target[2]), min(candidate[3], target[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    target_area = (target[2] - target[0]) * (target[3] - target[1])
    if target_area == 0:
        return 0.0
    return (iw * ih) / target_area


def box_gap(a: Box, b: Box) -> float:
    """Euclidean gap between two boxes; ``0.0`` when they overlap."""
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return float((dx * dx + dy * dy) ** 0.5)


@dataclass(frozen=True)
class LabelledCandidate:
    """A candidate with the label the re-verification model must predict."""

    candidate: Candidate
    label: str
    """One of the six defect names, ``false_call``, or ``fragment``."""

    matched_annotation: Annotation | None
    best_iou: float

    @property
    def trainable(self) -> bool:
        """Fragments are ambiguous and are held out of the training set."""
        return self.label != FRAGMENT


@dataclass(frozen=True)
class MatchResult:
    labelled: list[LabelledCandidate]
    detected_annotations: list[Annotation]
    missed_annotations: list[Annotation]

    @property
    def false_calls(self) -> list[LabelledCandidate]:
        return [c for c in self.labelled if c.label == FALSE_CALL]

    @property
    def fragments(self) -> list[LabelledCandidate]:
        return [c for c in self.labelled if c.label == FRAGMENT]

    @property
    def true_detections(self) -> list[LabelledCandidate]:
        return [c for c in self.labelled if c.label in CLASS_NAMES.values()]

    @property
    def trainable(self) -> list[LabelledCandidate]:
        return [c for c in self.labelled if c.trainable]


def match(
    candidates: list[Candidate],
    annotations: list[Annotation],
    iou_threshold: float = IOU_THRESHOLD,
    coverage_threshold: float = 0.5,
) -> MatchResult:
    """Assign each candidate a label and report which defects were missed.

    A candidate matches an annotation when either the IoU clears
    ``iou_threshold`` or the candidate covers at least ``coverage_threshold``
    of the annotation.
    """
    labelled: list[LabelledCandidate] = []
    hit: set[int] = set()

    for candidate in candidates:
        best_iou = 0.0
        best_index: int | None = None

        for index, annotation in enumerate(annotations):
            candidate_iou = iou(candidate.box, annotation.box)
            matches = (
                candidate_iou >= iou_threshold
                or coverage(candidate.box, annotation.box) >= coverage_threshold
            )
            if matches and candidate_iou >= best_iou:
                best_iou, best_index = candidate_iou, index

        if best_index is None:
            # Unmatched, but is it spurious or just a piece of a defect the
            # detector broke apart? Anything sitting on or beside a real defect
            # is ambiguous and gets held out rather than called spurious.
            gap = min(
                (box_gap(candidate.box, a.box) for a in annotations), default=float("inf")
            )
            label = FRAGMENT if gap <= FRAGMENT_GAP_PX else FALSE_CALL
            labelled.append(LabelledCandidate(candidate, label, None, best_iou))
        else:
            hit.add(best_index)
            annotation = annotations[best_index]
            labelled.append(
                LabelledCandidate(
                    candidate, annotation.class_name, annotation, best_iou
                )
            )

    detected = [a for i, a in enumerate(annotations) if i in hit]
    missed = [a for i, a in enumerate(annotations) if i not in hit]
    return MatchResult(labelled, detected, missed)
