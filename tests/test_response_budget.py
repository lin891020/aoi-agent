"""The response budget, the explanation deadline, and why they are two numbers.

CLAUDE.md's invariant is that thresholds come from the sweep or from the work
instructions. That is easy to write and easy to let rot: someone raises the
timeout because a run was slow, WI-300 still says ten seconds, and the document
quietly stops describing the system. These tests make the two disagree loudly.

They also hold the separation that the 2026-08-23 review forced. One constant
was doing two jobs -- WI-300's promise to the operator about when a verdict
lands, and the httpx client's timeout -- and the two pull opposite ways. A
promise must not follow the model. A resource bound must follow the measurement.
Held together at 10s against a model whose measured service time has a median of
12.5s, more than half of the station's explanations failed by construction, and
the only surviving job of the LLM is writing them. So:

- ``RESPONSE_BUDGET_S`` is WI-300's, bounds the verdict, and lives on the
  disposition path
- ``EXPLANATION_DEADLINE_S`` is the client's, bounds a wait nobody blocks on,
  and is sized from ``scripts/latency_report.py``

The test that matters most is the one asserting they are *not* the same object
in the same role. Re-merging them is the defect, and it would otherwise be a
one-line change that nothing notices.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from aoi_agent.graph.flow import RESPONSE_BUDGET_S
from aoi_agent.llm.ollama import EXPLANATION_DEADLINE_S, OllamaClient

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


def test_the_client_timeout_is_not_the_response_budget():
    """The regression this file exists for.

    They were one constant until 2026-08-23. A client that waits exactly as long
    as an operator was promised sounds tidy and is the defect: the operator's
    promise is about the verdict, which the classifier produces in milliseconds,
    while the client is waiting for prose the disposition does not depend on. At
    10s it discarded 20 of 24 measured calls and wrote a ``ReadTimeout`` into
    the operator's evidence panel.
    """
    client = OllamaClient("stub-model")
    assert client._client.timeout.read != RESPONSE_BUDGET_S
    assert client._client.timeout.read == EXPLANATION_DEADLINE_S


def test_the_deadline_clears_the_measured_distribution():
    """Sized from ``scripts/latency_report.py``, not chosen.

    The slowest of 24 reason-node calls on a verified-quiet machine was 21.1s.
    A deadline at or under that discards healthy work, which is what the old
    one did.
    """
    slowest_measured_s = 21.1
    assert EXPLANATION_DEADLINE_S > slowest_measured_s


def test_the_deadline_is_still_a_bound_and_not_an_absence_of_one():
    """Contention can multiply an LLM call's wall time by 25x, and no deadline
    should absorb that. The value in this role before 10s was 600s, which turned
    a busy GPU into a ten-minute blocked workstation."""
    assert EXPLANATION_DEADLINE_S <= 120.0


def test_the_budget_is_short_enough_to_be_worth_having():
    """QP-110 prices a retained false call at 'a few seconds' of operator time.

    A budget far above that spends the saving the escalation exists to make.
    """
    assert RESPONSE_BUDGET_S <= 30.0


def test_the_work_instruction_says_which_of_the_two_it_governs():
    """Without this clause the split in the code is a coincidence someone tidies
    away, and WI-300 reads as though it still deadlines the rationale."""
    text = WI_300.read_text()
    assert "Rationale deadline" in text
    assert re.search(r"not covered by the\s+response budget", text)


def test_the_work_instruction_requires_the_absence_to_be_counted():
    """A failure mode nobody counts is one nobody fixes -- which is how a queue
    came to hold an escalation reading only ``the model did not answer``."""
    assert "the absence shall be counted" in WI_300.read_text().lower()


@pytest.mark.parametrize("error", [httpx.ReadTimeout("expired"), httpx.ConnectError("down")])
def test_expiry_reaches_the_flow_as_an_escalation_not_a_crash(error):
    """The exception the deadline produces must be one the flow already catches.

    ``reason_node`` catches ``httpx.HTTPError``; a timeout that escaped it would
    leave a flagged board with no disposition at all.
    """
    assert isinstance(error, httpx.HTTPError)


def test_a_warm_request_is_not_mistaken_for_a_reload():
    """Measured: a warm, resident gpt-oss:20b reports ~168ms of load_duration.

    A gate at zero -- or at 100ms -- marks every healthy request as evicted, and
    a benchmark that drops every measurement reports nothing while looking like
    it ran.
    """
    from aoi_agent.llm.ollama import Timing

    warm = Timing(wall_ms=1500, load_ms=168, prompt_eval_ms=100,
                  eval_ms=1150, prompt_tokens=200, eval_tokens=32)
    assert not warm.was_reloaded


def test_a_genuine_reload_is_still_caught():
    """Pulling 12GB back onto the GPU takes seconds, not milliseconds."""
    from aoi_agent.llm.ollama import Timing

    reloaded = Timing(wall_ms=30000, load_ms=18000, prompt_eval_ms=100,
                      eval_ms=1150, prompt_tokens=200, eval_tokens=32)
    assert reloaded.was_reloaded
