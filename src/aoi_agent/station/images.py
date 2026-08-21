"""Rendering what the operator actually needs to look at.

The vision model sees a 64 px window because that is what its architecture was
trained on. A person cannot judge anything at 64 px on a screen, and a person
also needs to see the region in its surroundings -- a break in a trace only
looks like a break if you can see the trace. So the station renders a wider
window than the model's, at a legible scale, and marks the flagged region
inside it.

Three panels, because the disposition depends on all three: the golden
template, the board under test, and their difference. The difference alone is
what the AOI saw, and looking at it alone is exactly the mistake that produces
false calls.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

from aoi_agent.aoi.simulator import Candidate
from aoi_agent.store.boards import load_board_images
from aoi_agent.vision.patches import PATCH_SIZE, crop

CONTEXT_SIZE = 192
"""Window side in source pixels. Three times the model's ``PATCH_SIZE``, so the
flagged region sits in enough surrounding circuitry to be read as a trace, a
pad or a gap rather than as an isolated blob."""

SCALE = 3
"""Nearest-neighbour upscale. These are binarised line-scan images; smoothing
them would invent edges that are not in the data the model saw."""

PANEL_GAP = 8
LABELS = ("template", "test", "difference")


def _panels(stem: str, candidate: Candidate) -> list[np.ndarray]:
    template, test = load_board_images(stem)
    template_window = crop(template, candidate, CONTEXT_SIZE)
    test_window = crop(test, candidate, CONTEXT_SIZE)
    difference = np.abs(
        test_window.astype(np.int16) - template_window.astype(np.int16)
    ).astype(np.uint8)
    return [template_window, test_window, difference]


def _box_within_window(candidate: Candidate) -> tuple[int, int, int, int]:
    """Where the flagged box lands inside the centred context window."""
    cx = (candidate.x1 + candidate.x2) // 2
    cy = (candidate.y1 + candidate.y2) // 2
    half = CONTEXT_SIZE // 2
    return (
        candidate.x1 - (cx - half),
        candidate.y1 - (cy - half),
        candidate.x2 - (cx - half),
        candidate.y2 - (cy - half),
    )


def triptych(stem: str, candidate: Candidate, mark: bool = True) -> bytes:
    """Return a PNG of template / test / difference, side by side."""
    panels = _panels(stem, candidate)
    side = CONTEXT_SIZE * SCALE
    width = side * 3 + PANEL_GAP * 2

    canvas = Image.new("RGB", (width, side), (24, 24, 27))
    x1, y1, x2, y2 = _box_within_window(candidate)

    for index, panel in enumerate(panels):
        image = Image.fromarray(panel).convert("RGB")
        image = image.resize((side, side), Image.NEAREST)
        if mark:
            draw = ImageDraw.Draw(image)
            draw.rectangle(
                [x1 * SCALE, y1 * SCALE, x2 * SCALE - 1, y2 * SCALE - 1],
                outline=(239, 68, 68),
                width=2,
            )
        canvas.paste(image, (index * (side + PANEL_GAP), 0))

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def model_patch(stem: str, candidate: Candidate) -> bytes:
    """The 64 px window the model actually classified, upscaled.

    Shown so an operator who disagrees can see whether the model was even
    looking at the right thing -- a disagreement caused by a badly centred
    window is a different bug from one caused by a bad classifier.
    """
    template, test = load_board_images(stem)
    template_patch = crop(template, candidate, PATCH_SIZE)
    test_patch = crop(test, candidate, PATCH_SIZE)
    difference = np.abs(
        test_patch.astype(np.int16) - template_patch.astype(np.int16)
    ).astype(np.uint8)

    side = PATCH_SIZE * SCALE * 2
    width = side * 3 + PANEL_GAP * 2
    canvas = Image.new("RGB", (width, side), (24, 24, 27))
    for index, panel in enumerate([template_patch, test_patch, difference]):
        image = Image.fromarray(panel).convert("RGB").resize((side, side), Image.NEAREST)
        canvas.paste(image, (index * (side + PANEL_GAP), 0))

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def candidate_from_record(record: dict) -> Candidate:
    return Candidate(
        x1=record["x1"], y1=record["y1"], x2=record["x2"], y2=record["y2"],
        area=record["area"],
    )
