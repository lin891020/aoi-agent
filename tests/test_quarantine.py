"""Which explanations get marked as citing a rule nobody wrote.

No database is opened. The half of that script that walks two tables is
plumbing; the half that decides whether a given sentence disposition an open by
a pad -- and whether it has already been marked -- is where a mistake is
silent. Over-matching puts a correction banner on a correct pin-hole
explanation, which teaches an operator to ignore banners. Under-matching leaves
a fabricated acceptance rule in a quality record with nothing beside it.
"""

from __future__ import annotations

from quarantine_fabricated_criteria import BANNER, MARKER, is_fabricated, mark

FABRICATED = (
    "The vision model reports a 100 % confident open, but the acceptance "
    "criteria require that we determine whether the open is inside a pad "
    "(critical) or outside pads (release)."
)

CORRECT_FOR_OPEN = (
    "Any confirmed open is a critical defect; there is no width or length "
    "below which it is acceptable, so the region is escalated for confirmation."
)

CORRECT_FOR_PIN_HOLE = (
    "The void is within limits and outside any pad, so WI-206 releases it. "
    "Inside a pad it would be rejected."
)


def test_an_open_dispositioned_by_a_pad_is_marked():
    assert is_fabricated(FABRICATED, "open")


def test_the_same_sentence_about_a_pin_hole_is_left_alone():
    """WI-206 is where the pad rule comes from. Marking it there would be the
    mirror image of the defect: a correct citation flagged as a fabrication."""
    assert not is_fabricated(CORRECT_FOR_PIN_HOLE, "pin-hole")


def test_a_correct_open_explanation_is_left_alone():
    assert not is_fabricated(CORRECT_FOR_OPEN, "open")


def test_an_outage_note_is_left_alone():
    """The LLM being unreachable writes its own reason, and it cites nothing."""
    assert not is_fabricated("the model did not answer (ReadTimeout)", "open")


def test_nothing_is_read_out_of_an_empty_rationale():
    assert not is_fabricated(None, "open")
    assert not is_fabricated("", "open")


def test_marking_is_idempotent_so_the_script_can_be_re_run():
    once = mark(FABRICATED)
    assert once.startswith(MARKER)
    assert mark(once) == once
    assert not is_fabricated(once, "open")


def test_the_original_text_survives_verbatim():
    """The record of what the operator was shown is the point of keeping it."""
    assert mark(FABRICATED).endswith(FABRICATED)
    assert mark(FABRICATED) == BANNER + FABRICATED


def test_the_banner_names_the_rule_that_actually_applies():
    """A banner that only says "this is wrong" leaves the reader with the
    fabricated rule as the only rule in front of them."""
    assert "WI-201" in BANNER
    assert "critical" in BANNER
    assert "WI-206" in BANNER
