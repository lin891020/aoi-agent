"""The flow view on /ask: what it derives from the stream, and what it draws.

The diagram is a pure function of the events already on the wire -- `plan`
carries the branch count and their names, each `tool` carries one completion,
`synthesising` carries the join -- so the thing that can be wrong is the
derivation, not the drawing. That derivation is JavaScript, which is why these
tests shell out to node: `tests/fixtures/flow_harness.js` evaluates
`station/static/flow.js` in a bare context, feeds it a scenario and prints back
the model, the stage states, the status line and the element tree that was
built.

The document the harness hands `render` throws on `innerHTML`, read or write.
That is the rule this page is held to, and building an SVG on the client is
exactly where somebody reaches past it.

These skip when node is missing rather than failing on a machine that has no
reason to have it. `ubuntu-latest` ships node, and the workflow runs pytest
with `-rs` so a skip here is printed rather than swallowed -- the same reason
the dataset-marked tests are listed at the end of every CI job.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

HARNESS = Path(__file__).parent / "fixtures" / "flow_harness.js"
FLOW_JS = (
    Path(__file__).resolve().parents[1]
    / "src" / "aoi_agent" / "station" / "static" / "flow.js"
)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is needed to drive flow.js"
)


def drive(events: list[dict]) -> dict:
    """Run the flow view over these events and return everything it produced."""
    finished = subprocess.run(
        [shutil.which("node"), str(HARNESS), str(FLOW_JS)],
        input=json.dumps({"events": events}),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert finished.returncode == 0, finished.stderr
    return json.loads(finished.stdout)


def plan_event(*calls: str) -> dict:
    return {
        "event": "plan",
        "data": {
            "interpretation": "M22 against the fleet",
            "calls": [
                {"tool": tool, "key": f"{index}:{tool}"}
                for index, tool in enumerate(calls)
            ],
        },
    }


def tool_event(index: int, tool: str, ok: bool = True, ms: float = 12.0) -> dict:
    return {
        "event": "tool",
        "data": {"tool": tool, "key": f"{index}:{tool}", "ok": ok,
                 "error": None if ok else "index unreachable", "elapsed_ms": ms},
    }


def texts(node: dict) -> list[str]:
    """Every string the diagram put in the document, in document order."""
    found = [node["text"]] if node["text"] is not None else []
    for child in node["children"]:
        found.extend(texts(child))
    return found


def state_of(result: dict, stage: str) -> str:
    return next(s["state"] for s in result["stages"] if s["id"] == stage)


def test_before_any_event_the_run_is_planning_and_nothing_else_has_started():
    result = drive([])

    assert result["phase"] == "planning"
    assert result["label"] == "規劃中…"
    assert [s["state"] for s in result["stages"]] == [
        "active", "pending", "pending", "pending"
    ]
    assert result["running"] is True


def test_a_plan_with_one_call_fans_out_into_one_branch():
    result = drive([plan_event("search_standards")])

    assert result["phase"] == "running"
    assert [b["tool"] for b in result["branches"]] == ["search_standards"]
    assert result["outstanding"] == 1
    assert state_of(result, "plan") == "done"
    assert state_of(result, "fan") == "active"
    assert result["label"] == "查詢中…（0/1 完成）"


def test_a_plan_with_several_calls_tracks_which_are_still_outstanding():
    result = drive([
        plan_event("query_machine_stats", "search_standards", "query_defect_history"),
        # Out of arrival order on purpose: branches complete in whatever order
        # the tools take, which is not plan order, and the row that resolves
        # has to be the row whose key came back.
        tool_event(2, "query_defect_history"),
        tool_event(0, "query_machine_stats"),
    ])

    assert result["outstanding"] == 1
    assert [b["state"] for b in result["branches"]] == ["ok", "running", "ok"]
    assert result["label"] == "查詢中…（2/3 完成）"
    assert state_of(result, "fan") == "active"
    assert state_of(result, "write") == "pending"


def test_a_refusal_plans_no_branches_and_never_reaches_the_answer():
    """A plan with no calls routes straight to `report`: nothing fans out,
    nothing joins, no answer is written. The stages that never ran must not be
    drawn as stages that finished."""
    result = drive([
        {"event": "plan", "data": {"interpretation": "涵蓋不到那個區間", "calls": []}},
        {"event": "done", "data": {"run_id": 3}},
    ])

    assert result["phase"] == "refused"
    assert result["branches"] == []
    assert result["label"] == "沒有可執行的查詢"
    assert state_of(result, "plan") == "done"
    assert [state_of(result, s) for s in ("fan", "join", "write")] == [
        "skipped", "skipped", "skipped"
    ]
    assert result["running"] is False, "nothing is running, so nothing may tick"


def test_the_join_moves_the_run_to_writing_the_answer():
    """The event this whole change exists for. Until it arrives the page has
    every branch ticked and no reason to think anything is still happening."""
    result = drive([
        plan_event("query_machine_stats", "search_standards"),
        tool_event(0, "query_machine_stats"),
        tool_event(1, "search_standards"),
        {"event": "synthesising", "data": {"tools": 2}},
    ])

    assert result["phase"] == "synthesising"
    assert result["label"] == "撰寫回答中…"
    assert [state_of(result, s) for s in ("plan", "fan", "join", "write")] == [
        "done", "done", "done", "active"
    ]
    assert result["running"] is True, "the wait is not over, so the timer runs"


def test_done_finishes_every_stage():
    result = drive([
        plan_event("search_standards"),
        tool_event(0, "search_standards"),
        {"event": "synthesising", "data": {"tools": 1}},
        {"event": "done", "data": {"run_id": 7}},
    ])

    assert result["phase"] == "done"
    assert [s["state"] for s in result["stages"]] == ["done"] * 4
    assert result["running"] is False


def test_a_streamed_error_stops_the_run_and_the_later_done_does_not_undo_it():
    """A plan that does not validate streams its reasons and then a `done`,
    because the refusal is still a viewable run. The stages it never reached
    must stay skipped: a `done` arriving afterwards must not repaint them as
    work that happened."""
    result = drive([
        plan_event("query_defect_history"),
        {"event": "error", "data": {"message": "line_id='L9' is not a line the store holds"}},
        {"event": "done", "data": {"run_id": 9}},
    ])

    assert result["stopped"] is True
    assert result["running"] is False, "the timer must stop when the run does"
    assert [state_of(result, s) for s in ("fan", "join", "write")] == [
        "skipped", "skipped", "skipped"
    ]


def test_a_failing_tool_marks_its_branch_and_does_not_stop_the_run():
    """A branch that raised returns data rather than killing the flow -- the
    answer is written over what did come back."""
    result = drive([
        plan_event("query_machine_stats", "search_standards"),
        tool_event(1, "search_standards", ok=False),
    ])

    assert result["failed"] == 1
    assert result["running"] is True
    assert [b["state"] for b in result["branches"]] == ["running", "failed"]


def test_the_diagram_draws_one_box_per_branch_and_names_each_tool():
    result = drive([
        plan_event("query_machine_stats", "search_standards"),
        tool_event(0, "query_machine_stats"),
    ])
    drawn = texts(result["svg"])

    assert "✓ query_machine_stats" in drawn
    assert "· search_standards" in drawn
    assert "規劃" in drawn and "撰寫回答" in drawn
    assert result["svg"]["attrs"]["viewBox"].startswith("0 0 ")


def test_the_diagram_grows_with_the_branch_count_rather_than_being_a_fixed_picture():
    one = drive([plan_event("search_standards")])
    three = drive([plan_event("a", "b", "c")])

    def boxes(result):
        return sum(1 for child in result["svg"]["children"] if child["tag"] == "rect")

    assert boxes(three) == boxes(one) + 2
    assert int(three["svg"]["attrs"]["viewBox"].split()[3]) > int(
        one["svg"]["attrs"]["viewBox"].split()[3]
    )


def test_the_branch_column_before_a_plan_is_not_the_same_as_no_branches():
    """Drawn from the first moment, before any event: an empty diagram would
    reserve a box of nothing, and the planning phase is the page's other dead
    zone. What the column says then is that the branches are not known yet --
    which is a different statement from the model having declined to plan
    any."""
    waiting = drive([])

    assert "尚未規劃" in texts(waiting["svg"])
    assert "沒有查詢" not in texts(waiting["svg"])


def test_the_diagram_marks_the_refusal_rather_than_drawing_branches_that_never_ran():
    result = drive([
        {"event": "plan", "data": {"interpretation": "無法規劃", "calls": []}},
        {"event": "done", "data": {"run_id": 4}},
    ])

    assert "沒有查詢" in texts(result["svg"])


def test_every_label_the_diagram_draws_is_a_value_it_was_given():
    """`textContent`, and nothing else. The harness's document throws on
    `innerHTML` in either direction, so a build that reached for it would not
    get as far as this assertion -- this is the other half: every string in the
    tree arrived as text on a node, which is where the tool names go."""
    result = drive([plan_event("query_board_context")])
    tree = result["svg"]

    assert all(child["text"] is None or child["tag"] == "text"
               for child in tree["children"]), (
        "only <text> nodes carry text; nothing is written into a shape"
    )
    assert "· query_board_context" in texts(tree)


@pytest.mark.parametrize("forbidden", ["快", "加速", "省", "faster", "speed", "saved"])
def test_the_diagram_claims_nothing_about_time(forbidden):
    """The fan-out is the shape of the work. The tools cost milliseconds either
    side of two model calls of around eight seconds each, so a diagram implying
    the parallelism is what makes the page fast would be a lie the page tells.
    It is drawn parallel because the lookups are independent, and it says
    nothing else."""
    result = drive([
        plan_event("query_machine_stats", "search_standards"),
        tool_event(0, "query_machine_stats"),
        {"event": "synthesising", "data": {"tools": 2}},
    ])

    for text in texts(result["svg"]) + [result["label"]]:
        assert forbidden not in text
    assert forbidden not in FLOW_JS.read_text().split("*/", 1)[1], (
        "not in the code the diagram is built from either"
    )
