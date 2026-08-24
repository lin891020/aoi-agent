"""Aligning a tested board to its golden template.

The stage this project did not have. DeepPCB ships pre-registered, so the
detector was built and measured on pairs that were already aligned, and
`scripts/registration_report.py` measured what a misaligned pair costs: at a
4 px shift, AOI recall falls 95.0% to 90.4% and the queue triples. This is the
stage that answers the first half of that.

**Phase correlation, and the choice is not incidental.** The board is a
repeating field of copper on a dark ground with no texture to key on, so
feature matching -- ORB, SIFT -- has little to match: the descriptors that
survive binarisation are corners of traces that look like every other corner of
every other trace. Cross-correlation in the Fourier domain uses all of the image
at once instead of trusting a few points of it, runs in O(n log n), and returns
a sub-pixel peak. It is also what the literature on this problem reaches for
first, and what a real AOI's alignment amounts to when the board is placed by a
conveyor rather than a robot.

**What it recovers is a translation, and only that.** That bound is the honest
half of this module. A board arrives on a conveyor slightly crooked as well as
slightly off, and no amount of shifting corrects a rotation -- the report beside
this measures exactly how far the recovery holds and where it stops, because a
registration stage that silently half-works is worse than none: it moves the
image, reports success, and leaves the residual to be read downstream as
defects.

Nothing here estimates rotation. Doing so is an ECC or a log-polar step and both
want their own measurement; until one exists, `estimate_shift` returns what it
found and `confidence` says how much the correlation peak stood out, so a caller
can refuse rather than trust.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np


@dataclass(frozen=True)
class Alignment:
    """What registration found, and how sure the correlation was."""

    dx: float
    dy: float

    confidence: float
    """The correlation peak's height, in [0, 1].

    **Not sufficient as a refusal signal, which was measured rather than
    assumed.** The first version of this module said it was. On the 60 aligned
    pairs it was tested against, four produced estimated shifts between 240 and
    355 pixels on a 640-pixel frame -- correlation failures, not boards -- and
    two of those carried confidences of 0.134 and 0.076, comfortably above any
    floor low enough to admit the real cases. Confidence degrades under
    *rotation*, where the peak genuinely smears; it does not degrade when the
    peak is simply in the wrong place.

    What it is good for is the rotation case, and it is reported for that. What
    refuses a wrong answer is the magnitude, below.
    """

    refused: str | None = None
    """Why nothing was moved, or ``None`` if it was.

    A stage that silently declines is one nobody can debug, and on this one the
    decline is the interesting event: three of its four outcomes end here rather
    than in a warp.
    """

    @property
    def magnitude(self) -> float:
        return float(np.hypot(self.dx, self.dy))


#: Below this the correlation peak is not distinct enough to act on. Kept, but
#: it is the weaker of the two guards -- see `Alignment.confidence`.
MIN_CONFIDENCE = 0.05

#: Below this, do not warp at all.
#:
#: A sub-pixel correction costs more than it buys. These images are binarised,
#: so interpolating them produces grey where there was black and white, and the
#: detector reads that as difference along every edge it touched. Measured: on
#: 60 already-aligned pairs the median estimated shift is 0.48 px, and
#: registering on it made 17 of the 60 *worse*. At this floor that falls to 3.
MIN_SHIFT_PX = 1.0

#: Above this, refuse: as a fraction of the smaller image side.
#:
#: A board is misplaced by millimetres, not by a third of the frame. An estimate
#: this large is a correlation that failed or a panel that is not this product,
#: and either way warping by it destroys the pair. Four of 60 aligned pairs
#: produced estimates of 240-355 px on a 640 px frame, and confidence caught
#: only two of them -- which is why this guard exists separately.
MAX_SHIFT_FRACTION = 0.05


def estimate_shift(template: np.ndarray, test: np.ndarray) -> Alignment:
    """How far `test` has moved relative to `template`.

    A Hann window first. Both images have hard edges at the frame, and an
    unwindowed FFT reads those edges as a strong periodic signal in both images
    -- which correlates with itself perfectly and pulls the peak toward zero
    shift. The window is the difference between measuring the board and
    measuring the crop.
    """
    a = template.astype(np.float32)
    b = test.astype(np.float32)
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    return Alignment(dx=float(dx), dy=float(dy), confidence=float(response))


def align(
    template: np.ndarray, test: np.ndarray, alignment: Alignment | None = None
) -> tuple[np.ndarray, Alignment]:
    """Move `test` back onto `template`, and say by how much.

    The *test* image moves, never the template. The template is the reference
    the defect boxes are expressed against, and warping it would move the
    ground truth along with it -- which would make every measurement here look
    better than it is, by construction.

    Three reasons to leave the pair alone, and `refused` says which:

    ``low_confidence`` -- the peak is not distinct enough to act on.

    ``implausible`` -- the estimate is a larger fraction of the frame than a
    board is ever misplaced by. This is the guard that catches a correlation
    failure, and confidence is not: a 355 px estimate on a 640 px frame came
    back with a confidence of 0.134.

    ``already_aligned`` -- the correction is sub-pixel. These images are
    binarised, so warping them by half a pixel writes grey along every edge and
    the detector reads it as difference. Doing nothing is strictly better.
    """
    found = alignment or estimate_shift(template, test)
    limit = MAX_SHIFT_FRACTION * min(test.shape[:2])

    if found.confidence < MIN_CONFIDENCE:
        return test, replace(found, refused="low_confidence")
    if found.magnitude > limit:
        return test, replace(found, refused="implausible")
    if found.magnitude < MIN_SHIFT_PX:
        return test, replace(found, refused="already_aligned")

    matrix = np.float32([[1, 0, -found.dx], [0, 1, -found.dy]])
    corrected = cv2.warpAffine(
        test,
        matrix,
        (test.shape[1], test.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return corrected, found
