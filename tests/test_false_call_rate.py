"""The tool that answers the system's own subject -- carefully.

The independent evaluation found six of thirty-five foreman questions asking
for an aggregate false-call rate -- per machine, per shift, per line -- and no
tool could answer any of them. The refusal was correct and the question was
not wrong, which is the worst standing state a capability gap can have.

It went unbuilt for a reason worth keeping: the honest number is not the one
the name promises. This store's production records carry no ground truth, so a
"false call rate" here can only mean *the fraction of flagged regions this
system itself dismissed* -- the re-verifier's judgement, not the world's. That
is still the number a process engineer wants ("of what M22 flags, how much do
we throw away"), but it reads as truth unless the payload itself says
otherwise. These tests hold the tool to saying otherwise.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aoi_agent.mcp_servers.production import GROUP_BY, query_false_call_rate
from aoi_agent.store import boards as boards_module
from aoi_agent.store.models import (
    Board,
    CandidateRecord,
    create_all,
    make_session_factory,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Two lines, three days, and rates chosen to be checkable by hand.

    L1: 4 flagged, 3 dismissed -> 0.75.  L2: 4 flagged, 1 dismissed -> 0.25.
    Ground truth deliberately disagrees with the predictions on two regions,
    so any assertion that passes is passing on the *prediction* column.
    """
    url = f"sqlite:///{tmp_path / 'fcr.db'}"
    create_all(url)
    factory = make_session_factory(url)
    monkeypatch.setattr(boards_module, "_session_factory", factory)

    plan = [
        # line, machine, shift, [predicted classes], ground truths
        ("L1", "M11", "A", ["false_call", "false_call", "false_call", "open"],
                          ["false_call", "open",       "false_call", "open"]),
        ("L2", "M21", "B", ["false_call", "open", "short", "spur"],
                          ["open",       "open", "short", "spur"]),
    ]
    with factory() as session:
        for index, (line, machine, shift, predicted, truths) in enumerate(plan):
            board = Board(
                stem=f"9000000{index}", split="test", lot_id="LOT-1",
                line_id=line, machine_id=machine, shift=shift,
                inspected_at=datetime(2026, 8, 1 + 2 * index, 8, 0),
            )
            session.add(board)
            session.flush()
            for position, (klass, truth) in enumerate(zip(predicted, truths)):
                session.add(
                    CandidateRecord(
                        board_id=board.id, index_on_board=position,
                        x1=1, y1=1, x2=9, y2=9, area=64,
                        predicted_class=klass, confidence=0.9,
                        false_call_probability=0.9 if klass == "false_call" else 0.1,
                        ground_truth=truth,
                    )
                )
        session.commit()
    return factory


def hunt(payload, needle: str) -> bool:
    """Whether a key appears anywhere in a nested payload."""
    if isinstance(payload, dict):
        return any(k == needle or hunt(v, needle) for k, v in payload.items())
    if isinstance(payload, list):
        return any(hunt(item, needle) for item in payload)
    return False


# ---- the arithmetic ------------------------------------------------------


def test_the_rate_is_dismissals_over_flags_per_group(store):
    out = query_false_call_rate("line", days=30)

    by = {row["group"]: row for row in out["by_group"]}
    assert by["L1"]["flagged"] == 4 and by["L1"]["dismissed_as_false_call"] == 3
    assert by["L1"]["false_call_rate"] == 0.75
    assert by["L2"]["false_call_rate"] == 0.25
    assert out["false_call_rate"] == 0.5


def test_the_worst_group_is_first(store):
    """The question that reaches this tool is "who over-flags", so the answer
    leads with them."""
    assert [row["group"] for row in query_false_call_rate("line", days=30)["by_group"]] \
        == ["L1", "L2"]


def test_every_axis_in_the_vocabulary_actually_groups(store):
    for axis in GROUP_BY:
        out = query_false_call_rate(axis, days=30)
        assert len(out["by_group"]) == 2, f"{axis} did not partition the store"


def test_the_rate_reads_the_prediction_and_never_the_answer_key(store):
    """The fixture plants two regions where ground truth disagrees with the
    prediction. A rate of 0.75 on L1 is only reachable through the prediction
    column -- the truth column would give 0.5."""
    by = {row["group"]: row for row in query_false_call_rate("line", days=30)["by_group"]}

    assert by["L1"]["false_call_rate"] == 0.75, (
        "the rate moved when ground truth disagreed, so something read it"
    )


# ---- the epistemic contract ----------------------------------------------


def test_the_payload_carries_no_ground_truth(store):
    """Same boundary as every other surface. This payload feeds the synthesis
    prompt and the page; the answer key reaches neither."""
    assert not hunt(query_false_call_rate("machine", days=30), "ground_truth")


def test_the_payload_says_whose_judgement_the_number_is(store):
    """The caveat travels with the number, not with the documentation. A reader
    of the payload -- the synthesis model included -- sees the basis or sees
    nothing."""
    out = query_false_call_rate("line", days=30)

    assert "not ground truth" in out["basis"]
    assert "escape" in out["basis"], (
        "the basis must say the one error this rate cannot see"
    )


def test_a_window_larger_than_the_store_is_reported_not_relabelled(store):
    """The days=14-over-a-9-day-store defect, refused at birth. The store here
    spans two days; asking for 30 must say what was actually covered."""
    out = query_false_call_rate("line", days=30)

    assert out["window_days_requested"] == 30
    assert out["window_days_covered"] == 2


def test_an_axis_outside_the_vocabulary_is_an_error_payload(store):
    out = query_false_call_rate("operator")

    assert "error" in out and "operator" in out["error"]
