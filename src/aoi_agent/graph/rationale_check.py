"""Does the rationale cite a figure the model was never shown?

The analysis page has had this check since 2026-08-29 (`analysis/claims`):
a number in the prose that renders from nothing in the stored results is
flagged beside the answer. The disposition path had nothing of the kind, and
the first Chinese rationale measured that day cited "a 0.85 threshold" -- a
number that appears in no work instruction, no threshold table and no line of
the prompt. It read exactly like the numbers beside it, which is the point:
the operator cannot tell an invented figure from a quoted one, and the
rationale is the only thing on the queue row they read.

The ground here is the prompt itself, not a tool payload -- the reason node
composes one string from the classifier's reading, the production context and
the retrieved criteria, and every figure the model may legitimately cite is in
that string. So the rule is narrow and mechanical: **a figure in the rationale
must render from a figure in the prompt.** `0.55` renders from `0.550`, `55%`
renders from `0.55`, `2201` renders from `LOT-2201`. Nothing else is
accepted, for the reason `claims._renderings` gives: every extra rendering is
a number the check waves through.

What this does not do: judge whether a grounded figure is *used* correctly.
"confidence 0.55, well above the threshold" cites a real figure and a wrong
comparison, and this module cannot see the second. That is the same boundary
the analysis checker draws, and it is stated on the page rather than hidden.

Small bare integers are not flagged. "2 passages", "the 3rd region" and "one
of 2 lots" are counts of things in the sentence, never measurements; on the
analysis page the same trade-off is made by dropping spelled-out numbers. The
cut is at `SMALL_COUNT`, and a fabricated threshold is never an integer.
"""

from __future__ import annotations

from decimal import Decimal

from aoi_agent.analysis.claims import _renderings, normalise, numbers_in

#: Bare integers at or below this are read as counts, not measurements.
SMALL_COUNT = 12


def unsourced_figures(rationale: str, prompt: str) -> list[str]:
    """Figures in ``rationale`` that render from nothing in ``prompt``.

    Returned as the strings the operator will see, in order of appearance,
    deduplicated. Empty means every figure was shown to the model -- not that
    the rationale is right.
    """
    shown: set[Decimal] = set()
    for value, _places, _s, _e in numbers_in(normalise(prompt), words=False):
        shown |= _renderings(value)

    flagged: list[str] = []
    text = normalise(rationale)
    for value, places, start, end in numbers_in(text, words=False):
        if places == 0 and value <= SMALL_COUNT:
            continue
        quantum = Decimal(1).scaleb(-places)
        if any(
            candidate.quantize(quantum) == value
            for candidate in shown
        ):
            continue
        literal = text[start:end]
        if literal not in flagged:
            flagged.append(literal)
    return flagged
