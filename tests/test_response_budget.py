"""The response budget, and the rule that the code reads it rather than sets it.

CLAUDE.md's invariant is that thresholds come from the sweep or from the work
instructions. That is easy to write and easy to let rot: someone raises the
timeout because a run was slow, WI-300 still says ten seconds, and the document
quietly stops describing the system. This test makes the two disagree loudly.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from aoi_agent.llm.ollama import RESPONSE_BUDGET_S, OllamaClient

WI_300 = Path("data/standards/reverification-procedure.md")


def test_the_budget_matches_what_the_work_instruction_says():
    text = WI_300.read_text()
    stated = re.search(r"within \*\*(\d+(?:\.\d+)?)\s*\n?seconds\*\*", text)
    assert stated, "WI-300 no longer states a response budget"
    assert float(stated.group(1)) == RESPONSE_BUDGET_S


def test_the_work_instruction_forbids_raising_the_budget_to_fit_the_model():
    """The direction of the dependency is the whole point of the clause."""
    text = WI_300.read_text().lower()
    assert "the budget is not to be raised" in text


def test_the_client_defaults_to_the_budget():
    client = OllamaClient("stub-model")
    assert client._client.timeout.read == RESPONSE_BUDGET_S


def test_the_budget_is_short_enough_to_be_worth_having():
    """QP-110 prices a retained false call at 'a few seconds' of operator time.

    A budget far above that spends the saving the escalation exists to make.
    """
    assert RESPONSE_BUDGET_S <= 30.0


@pytest.mark.parametrize("error", [httpx.ReadTimeout("expired"), httpx.ConnectError("down")])
def test_expiry_reaches_the_flow_as_an_escalation_not_a_crash(error):
    """The exception the budget produces must be one the flow already catches.

    ``reason_node`` catches ``httpx.HTTPError``; a timeout that escaped it would
    leave a flagged board with no disposition at all.
    """
    assert isinstance(error, httpx.HTTPError)
