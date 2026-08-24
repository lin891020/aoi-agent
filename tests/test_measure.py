"""The ruler: what makes a reading meaningful, and what makes one a lie.

Three classes are accepted on a ratio -- mousebite at >=80% of nominal width,
spur at >=50% of nominal clearance, pin-hole under 25% of conductor width --
and WI-203 says of its own class "escalate for measurement". The operator is
the last stop, so that instruction had nowhere to go: there was nothing on the
page to measure with.

What this module holds is that the tool refuses the readings that would be
wrong rather than returning a plausible number for them. A ruler that answers
when it should not is worse on an acceptance decision than no ruler, because
the number it produces is indistinguishable from a real one.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HARNESS = Path(__file__).parent / "fixtures" / "measure_harness.js"
MEASURE_JS = (
    Path(__file__).resolve().parents[1]
    / "src" / "aoi_agent" / "station" / "static" / "measure.js"
)


def _node_or_reason() -> str | None:
    if shutil.which("node"):
        return None
    if os.environ.get("AOI_REQUIRE_NODE") == "1":
        raise RuntimeError(
            "AOI_REQUIRE_NODE=1 and node is not on PATH. These tests drive "
            "station/static/measure.js, and skipping them would leave the job "
            "green with the acceptance ruler unexercised."
        )
    return "node is needed to drive measure.js"


pytestmark = pytest.mark.skipif(
    _node_or_reason() is not None, reason="node is needed to drive measure.js"
)

#: The triptych as `station/images.py` renders it: three 576 px panels with two
#: 8 px gaps. Asserted against the server's own constants below rather than
#: copied and hoped over.
WIDTH = 192 * 3 * 3 + 8 * 2
HEIGHT = 192 * 3
GAP = 8 / WIDTH


def drive(**scenario) -> dict:
    scenario.setdefault("width", WIDTH)
    scenario.setdefault("height", HEIGHT)
    scenario.setdefault("gap", GAP)
    finished = subprocess.run(
        [shutil.which("node"), str(HARNESS), str(MEASURE_JS)],
        input=json.dumps(scenario), capture_output=True, text=True, timeout=30,
    )
    assert finished.returncode == 0, finished.stderr
    return json.loads(finished.stdout)


def at(x: float, y: float) -> dict:
    return {"x": x, "y": y}


# ---------------------------------------------------------------------------
# The layout this measures on
# ---------------------------------------------------------------------------

def test_the_panel_geometry_matches_the_one_the_server_renders():
    """Two copies of a layout are two things to keep true. This is the one
    place they are checked against each other."""
    from aoi_agent.station import images

    assert WIDTH == images.CONTEXT_SIZE * images.SCALE * 3 + images.PANEL_GAP * 2
    assert HEIGHT == images.CONTEXT_SIZE * images.SCALE


def test_a_point_in_the_gap_belongs_to_no_panel():
    """The gap is canvas, not board."""
    panels = drive(defect_class="mousebite")["panels"]

    assert panels[0] == 0
    assert panels[3] == 2
    assert -1 not in (panels[0], panels[3])


# ---------------------------------------------------------------------------
# What it refuses
# ---------------------------------------------------------------------------

def test_a_segment_spanning_two_panels_is_not_a_length():
    """The gap is 8 px of canvas between two crops. A segment across it is a
    number that looks like a length and is not one."""
    result = drive(
        defect_class="mousebite",
        reference=[at(0.30, 0.5), at(0.40, 0.5)],  # crosses the first gap
    )

    assert result["reference"] is None


def test_two_segments_on_different_panels_do_not_make_a_ratio():
    """Arithmetically fine -- the panels are the same scale -- and that is
    exactly why it needs refusing by name: it is a ratio of a length on the
    golden template to a length on the board under test."""
    result = drive(
        defect_class="mousebite",
        reference=[at(0.05, 0.4), at(0.05, 0.8)],   # panel 0, the template
        measured=[at(0.40, 0.4), at(0.40, 0.7)],    # panel 1, the test board
    )

    assert result["reading"]["verdict"] == "incomparable"
    assert result["reading"]["ratio"] is None


def test_a_zero_length_reference_yields_no_reading():
    """Two clicks in the same place. A ratio over zero is an infinity on an
    acceptance decision."""
    result = drive(
        defect_class="mousebite",
        reference=[at(0.05, 0.5), at(0.05, 0.5)],
        measured=[at(0.05, 0.4), at(0.05, 0.8)],
    )

    assert result["reading"] is None


def test_a_class_judged_on_no_ratio_offers_no_criterion():
    """`open` and `short` admit no acceptable instance, so there is nothing to
    measure and the tool must not imply there is."""
    for defect_class in ("open", "short", "copper"):
        assert drive(defect_class=defect_class)["criterion"] is None


# ---------------------------------------------------------------------------
# What it answers
# ---------------------------------------------------------------------------

def test_the_ratio_is_scale_free_which_is_why_it_needs_no_calibration():
    """DeepPCB carries no mm-per-pixel, so a ruler reporting a length in
    millimetres would be inventing the only number that mattered. The same two
    gestures at any zoom give the same ratio -- and all three criteria are
    already written as ratios."""
    small = drive(
        defect_class="mousebite",
        reference=[at(0.05, 0.30), at(0.05, 0.70)],
        measured=[at(0.10, 0.30), at(0.10, 0.64)],
        width=WIDTH, height=HEIGHT,
    )
    large = drive(
        defect_class="mousebite",
        reference=[at(0.05, 0.30), at(0.05, 0.70)],
        measured=[at(0.10, 0.30), at(0.10, 0.64)],
        width=WIDTH * 4, height=HEIGHT * 4,
    )

    assert small["reading"]["ratio"] == pytest.approx(large["reading"]["ratio"])


def test_a_mousebite_keeping_most_of_its_width_is_within_limits():
    """WI-203: acceptable at >= 80% of the nominal width. 0.34/0.40 = 85%."""
    result = drive(
        defect_class="mousebite",
        reference=[at(0.05, 0.30), at(0.05, 0.70)],
        measured=[at(0.10, 0.30), at(0.10, 0.64)],
    )

    assert result["reading"]["ratio"] == pytest.approx(0.85, abs=0.01)
    assert result["reading"]["verdict"] == "within"


def test_a_deeper_mousebite_is_outside_them():
    """0.28/0.40 = 70%, under WI-203's 80%."""
    result = drive(
        defect_class="mousebite",
        reference=[at(0.05, 0.30), at(0.05, 0.70)],
        measured=[at(0.10, 0.30), at(0.10, 0.58)],
    )

    assert result["reading"]["verdict"] == "outside"


def test_the_direction_of_the_limit_follows_the_document_not_the_arithmetic():
    """A mousebite must keep *at least* 80% of its width; a pin-hole must stay
    *under* 25% of it. The same ratio is a pass for one and a reject for the
    other, so which side is acceptable is data on the criterion rather than a
    branch somebody can invert.
    """
    points = dict(
        reference=[at(0.05, 0.30), at(0.05, 0.70)],
        measured=[at(0.10, 0.30), at(0.10, 0.64)],   # 85%
    )

    assert drive(defect_class="mousebite", **points)["reading"]["verdict"] == "within"
    assert drive(defect_class="pin-hole", **points)["reading"]["verdict"] == "outside"


def test_every_ratio_judged_class_carries_the_limit_its_document_states():
    """The three numbers, against the work instructions they come from."""
    expected = {"mousebite": 0.80, "spur": 0.50, "pin-hole": 0.25}

    for defect_class, limit in expected.items():
        criterion = drive(defect_class=defect_class)["criterion"]
        assert criterion["limit"] == limit
        assert criterion["reference"] and criterion["measured"], (
            "the tool has to say which two things it is asking to be measured"
        )


def test_a_width_across_a_trace_compares_with_a_length_along_it():
    """The case every test above missed, and the one the tool exists for.

    A nominal conductor width is measured across the trace and a notch depth
    often along it, so the two segments are at right angles. Every scenario
    here was vertical until this was written, and a ratio of two vertical
    lengths is unchanged by scaling the y axis — so measuring `dy` in fractions
    of the *width* passed all of them. The triptych is three times wider than
    it is tall, so that bug reads a right-angled ratio 3x wrong, in the
    direction that passes a reject.
    """
    # Same true length on the board: 0.4 of the panel height vertically,
    # and the identical distance horizontally inside one panel.
    panel = (1 - GAP * 2) / 3
    horizontal = 0.4 * HEIGHT / WIDTH

    result = drive(
        defect_class="mousebite",
        reference=[at(0.05, 0.30), at(0.05, 0.70)],                  # vertical
        measured=[at(0.05, 0.50), at(0.05 + horizontal, 0.50)],      # horizontal
    )

    assert 0.05 + horizontal < panel, "the horizontal leg must stay in panel 0"
    assert result["reading"]["ratio"] == pytest.approx(1.0, abs=0.01), (
        "two equal lengths at right angles have to read as a ratio of one"
    )
