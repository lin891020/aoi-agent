"""The project skills, and whether they still describe the code they cite.

A skill in `.claude/skills/` is prose that asserts things about this codebase --
a constant's value, a helper's existence, the threshold it tests. Nothing
imports it and nothing type-checks it, so when the code moves the skill does not
break: it keeps giving a confident, stale answer, which is worse than giving
none.

This is not hypothetical. `measuring-llm-latency` shipped saying "discard the
run when `load_ms > 0`" while `Timing.was_reloaded` was already in the tree
testing `load_ms > 100`; an hour later `RESPONSE_BUDGET_S` cut the call timeout
from 600s to 10s and the skill's whole framing went out of date. Neither was
findable by following the skill -- an agent obeying it perfectly still got the
wrong answer.

Same shape as test_response_budget.py: make the document and the code disagree
loudly, at the moment the code changes, rather than quietly forever.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_skill_freshness import (  # noqa: E402
    _assign,
    check_measuring_llm_latency,
    check_retraining_the_reverifier,
)


def test_measuring_llm_latency_still_describes_the_code():
    stale = check_measuring_llm_latency()
    assert not stale, "\n".join(
        ["measuring-llm-latency cites code that has moved:", *stale]
    )


def test_retraining_the_reverifier_still_describes_the_code():
    stale = check_retraining_the_reverifier()
    assert not stale, "\n".join(
        ["retraining-the-reverifier cites code that has moved:", *stale]
    )


def test_the_checker_notices_when_a_constant_moves():
    """A freshness check that cannot fail is worse than none -- it grants a
    false all-clear to every skill it covers."""
    before = ast.parse("RESPONSE_BUDGET_S = 10.0")
    after = ast.parse("RESPONSE_BUDGET_S = 30.0")
    assert _assign(before, "RESPONSE_BUDGET_S") == 10.0
    assert _assign(after, "RESPONSE_BUDGET_S") == 30.0
    assert _assign(before, "NEVER_DEFINED") is None
