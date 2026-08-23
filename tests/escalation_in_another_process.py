"""Raise one escalation, then exit.

Run as a program, not collected as a test -- the point is that the interpreter
that suspends the run is gone by the time anything tries to resume it. Nothing
here is shared with the resuming process except the checkpoint file named by
``AOI_AGENT_CHECKPOINT_PATH``.

    python tests/escalation_in_another_process.py <candidate_ref>

Prints the interrupt payload and the trace as JSON so the parent can check that
what it later reads off disk is the same run, not a fresh one.
"""

from __future__ import annotations

import json
import sys

from aoi_agent.graph import flow

from test_graph import StubClient


def stub_the_tools() -> None:
    """No GPU, no Ollama, no store. The subject is the checkpointer."""
    flow.classify_defect = lambda candidate_ref: {
        "candidate_ref": candidate_ref,
        "predicted_class": "open",
        "confidence": 0.55,          # below ESCALATE_BELOW: this is what escalates
        "false_call_probability": 0.10,
        "recommendation": "escalate",
    }
    flow.query_board_context = lambda board: {
        "lot_id": "LOT-1", "line_id": "L2", "machine_id": "M22",
        "shift": "A", "lot_defects_per_board": 7.2,
    }
    flow.query_machine_stats = lambda defect, days=30: {
        "defect_type": defect, "fleet_share_of_defects": 0.225,
        "machines": [{"machine": "L2-M22", "share_of_defects": 0.321, "per_board": 2.3}],
    }
    flow.search_standards = lambda query, top_k=2, defect_class=None: {
        "passages": [{"document": "open-circuit", "heading": "Classification",
                      "text": "Any confirmed open is a critical defect."}]
    }


def main() -> int:
    reference = sys.argv[1]
    stub_the_tools()

    # No checkpointer argument: the durable default is what is under test.
    graph = flow.build_graph(StubClient(confident=False, rationale="raised elsewhere"))
    state = graph.invoke(
        {"candidate_ref": reference},
        config={"configurable": {"thread_id": reference}},
    )

    if "__interrupt__" not in state:
        print(json.dumps({"error": "the run did not suspend", "state_keys": list(state)}))
        return 1

    print(json.dumps({
        "payload": state["__interrupt__"][0].value,
        "trace": state["trace"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
