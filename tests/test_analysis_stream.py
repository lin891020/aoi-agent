"""Progress events, and the guarantee that the page works without them."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aoi_agent.analysis import graph as analysis
from aoi_agent.llm.ollama import ChatResult, Timing
from aoi_agent.station import app as station_app
from aoi_agent.store import analysis as analysis_store
from aoi_agent.store.models import create_all, make_session_factory
from conftest import sign_in

PLAN = {
    "interpretation": "M22 against the fleet",
    "assumptions": ["fleet average"],
    "calls": [
        {"tool": "query_machine_stats", "args": {"defect_type": "open", "days": 7},
         "why": "a"},
        {"tool": "search_standards", "args": {"query": "open", "top_k": 2},
         "why": "b"},
    ],
}
DOMAINS = {"line_id": {"L2"}, "machine_id": {"M22"},
           "defect_type": {"open"}, "max_days": 9}


@dataclass
class StubClient:
    plan: dict = field(default_factory=lambda: PLAN)

    def chat(self, messages, **kwargs) -> ChatResult:
        text = json.dumps(self.plan) if kwargs.get("response_format") else "done"
        return ChatResult(text=text, tool_calls=[], thinking="",
                          timing=Timing(1.0, 0.0, 1.0, 1.0, 10, 10))


@pytest.fixture
def client(tmp_path, monkeypatch, operators):
    url = f"sqlite:///{tmp_path / 'a.db'}"
    create_all(url)
    monkeypatch.setattr("aoi_agent.store.boards._session_factory",
                        make_session_factory(url))
    for name in ("query_machine_stats", "search_standards"):
        monkeypatch.setitem(analysis.PLANNABLE_TOOLS, name, lambda **kw: {"ok": 1})
    monkeypatch.setattr(station_app, "_analysis_graph",
                        analysis.build_analysis_graph(StubClient(), DOMAINS))
    return sign_in(TestClient(station_app.app))


def events(body: str) -> list[dict]:
    out = []
    for block in body.strip().split("\n\n"):
        name, payload = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
        if name:
            out.append({"event": name, "data": payload})
    return out


def test_the_stream_reports_the_plan_then_each_tool_then_done(client):
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    names = [e["event"] for e in events(body)]

    assert names[0] == "plan"
    assert names.count("tool") == 2
    assert names[-1] == "done"


def test_the_plan_event_carries_what_will_run_so_the_page_can_list_it(client):
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    first = events(body)[0]["data"]

    assert [c["tool"] for c in first["calls"]] == [
        "query_machine_stats", "search_standards"
    ]
    assert first["interpretation"]


def test_the_done_event_carries_the_run_id_to_navigate_to(client):
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    done = events(body)[-1]["data"]

    assert isinstance(done["run_id"], int)
    assert client.get(f"/ask/{done['run_id']}").status_code == 200


def test_the_run_the_stream_persists_is_the_same_one_the_page_renders(client):
    """The stream must not be a second, parallel execution -- that would double
    the cost and could disagree with itself.

    This is also the merge-correctness test: ``stream_mode="updates"`` hands
    back one update per ``Send`` branch, so a naive ``state.update(payload)``
    treats each branch's single-item ``results`` list -- and each node's
    contribution to ``timings_ms`` -- as last-write-wins, keeping only
    whichever update happened to land last. That made the previous version of
    this test order-dependent: it passed only when ``query_machine_stats``
    happened to be the branch folded in last, and failed against
    ``state.update`` otherwise. Asserting both tool names and both a `plan`
    and a `synthesise` timing closes that gap -- any one of the three being
    silently dropped fails it, regardless of arrival order.
    """
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    run_id = events(body)[-1]["data"]["run_id"]

    run = analysis_store.get_run(run_id)
    assert sorted(r["tool"] for r in run["results"]) == [
        "query_machine_stats", "search_standards"
    ]
    assert "plan" in run["timings"]
    assert "synthesise" in run["timings"]

    page = client.get(f"/ask/{run_id}").text
    assert "query_machine_stats" in page
    assert "search_standards" in page


def test_a_failing_tool_is_streamed_as_a_failed_tool_event(client, monkeypatch):
    def boom(**kw):
        raise RuntimeError("index unreachable")

    monkeypatch.setitem(analysis.PLANNABLE_TOOLS, "search_standards", boom)
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    tools = [e["data"] for e in events(body) if e["event"] == "tool"]

    assert any(t["ok"] is False and "unreachable" in t["error"] for t in tools)


def test_a_rejected_plan_ends_the_stream_with_an_error_event(client, monkeypatch):
    bad = {"interpretation": "i", "assumptions": [],
           "calls": [{"tool": "query_defect_history",
                      "args": {"line_id": "L9"}, "why": "w"}]}
    monkeypatch.setattr(station_app, "_analysis_graph",
                        analysis.build_analysis_graph(StubClient(plan=bad), DOMAINS))
    body = client.get("/ask/stream", params={"question": "L9"}).text
    names = [e["event"] for e in events(body)]

    assert names[-1] == "done", "a rejected plan still produces a viewable run"
    assert any(e["event"] == "error" for e in events(body))


def test_two_calls_to_the_same_tool_get_distinct_rows(client, monkeypatch):
    """Two calls to one tool used to collide on the element id `tool-<name>`
    (keyed by tool name alone), so the second call's `tool` event found
    nothing to update and its row stayed reading "running" forever. Different
    arguments must produce different keys -- one per planned call -- and every
    `tool` event's key must be claimable by exactly one of them, which is the
    actual mechanism that keeps both rows resolving on the page."""
    twice = {
        "interpretation": "two standards lookups",
        "assumptions": [],
        "calls": [
            {"tool": "search_standards", "args": {"query": "open", "top_k": 2},
             "why": "a"},
            {"tool": "search_standards", "args": {"query": "short", "top_k": 2},
             "why": "b"},
        ],
    }
    monkeypatch.setattr(station_app, "_analysis_graph",
                        analysis.build_analysis_graph(StubClient(plan=twice), DOMAINS))
    body = client.get("/ask/stream", params={"question": "two lookups"}).text
    plan_event = next(e for e in events(body) if e["event"] == "plan")
    tool_events = [e["data"] for e in events(body) if e["event"] == "tool"]

    plan_keys = [c["key"] for c in plan_event["data"]["calls"]]
    assert len(set(plan_keys)) == 2, "two different calls must get two different keys"

    tool_keys = sorted(t["key"] for t in tool_events)
    assert tool_keys == sorted(plan_keys), (
        "every tool event must be claimable by exactly one planned call's row"
    )


def test_two_byte_identical_calls_to_the_same_tool_both_resolve(client, monkeypatch):
    """`_tool_base_key` alone (tool + args) is not enough: `validate_plan` has
    no rule against two calls that are byte-identical -- same tool, same
    args -- so content gives them nothing to disambiguate by. This is the
    shape that keying purely on `_tool_base_key` still collides on: both
    `plan` calls carry the exact same base key, and without a further
    tie-break both `tool` events would too, so one page row would never
    resolve. Asserts two distinct keys on the plan side, and that both keys
    get claimed by the two (indistinguishable) tool events -- i.e. both rows
    resolve, not just one of them twice."""
    identical = {
        "interpretation": "same lookup twice",
        "assumptions": [],
        "calls": [
            {"tool": "search_standards", "args": {"query": "open", "top_k": 2},
             "why": "a"},
            {"tool": "search_standards", "args": {"query": "open", "top_k": 2},
             "why": "a"},
        ],
    }
    monkeypatch.setattr(station_app, "_analysis_graph",
                        analysis.build_analysis_graph(StubClient(plan=identical), DOMAINS))
    body = client.get("/ask/stream", params={"question": "same lookup twice"}).text
    plan_event = next(e for e in events(body) if e["event"] == "plan")
    tool_events = [e["data"] for e in events(body) if e["event"] == "tool"]

    plan_keys = sorted(c["key"] for c in plan_event["data"]["calls"])
    assert len(plan_keys) == 2
    assert plan_keys[0] != plan_keys[1], (
        "two byte-identical calls must still get two distinct keys"
    )

    tool_keys = sorted(t["key"] for t in tool_events)
    assert tool_keys == plan_keys, "both rows must resolve, not just one of them twice"


def test_the_form_still_works_with_no_javascript(client):
    """The stream is an enhancement. The station runs on shop-floor browsers."""
    response = client.post("/ask", data={"question": "M22 正常嗎"},
                           follow_redirects=False)
    assert response.status_code == 303


def test_each_result_keeps_its_own_why_when_the_tools_finish_out_of_order(
    client, monkeypatch
):
    """The "why" column has to belong to the tool it sits beside.

    Every other multi-call test here fans out over fakes that return instantly,
    so the branches finish in whatever order they were dispatched and pairing
    result *i* with `plan.calls[i]` happens to look right. It is not right.
    `POST /ask` reads the accumulated state, where the reducer applied the
    branch writes in plan order; this route rebuilds its own copy from
    `stream_mode="updates"`, which reports each branch as it *completes*. With
    three tools at 300/150/10ms the two orders are exact reverses, so the
    fastest tool's row carried the slowest tool's justification -- on the path
    the page actually uses whenever JavaScript is on.

    Asserted on the stored run rather than on the rendered page, because that
    is where the pairing is now decided: `why` travels on the result, and
    `service.in_plan_order` puts the rows back in the order the plan asked
    for. The unequal sleeps are the whole point of the fixture -- with equal
    ones this test cannot fail.
    """
    slow_first = {
        "interpretation": "three lookups of very different cost",
        "assumptions": [],
        "calls": [
            {"tool": "query_defect_history", "args": {"line_id": "L2"},
             "why": "the slow one"},
            {"tool": "query_machine_stats", "args": {"defect_type": "open"},
             "why": "the middling one"},
            {"tool": "search_standards", "args": {"query": "open"},
             "why": "the fast one"},
        ],
    }
    delays = {"query_defect_history": 0.30, "query_machine_stats": 0.15,
              "search_standards": 0.01}

    def slow(name):
        def call(**kw):
            time.sleep(delays[name])
            return {"tool": name}
        return call

    for name in delays:
        monkeypatch.setitem(analysis.PLANNABLE_TOOLS, name, slow(name))
    monkeypatch.setattr(
        station_app, "_analysis_graph",
        analysis.build_analysis_graph(StubClient(plan=slow_first), DOMAINS),
    )

    body = client.get("/ask/stream", params={"question": "三個工具"}).text
    arrived = [e["data"]["tool"] for e in events(body) if e["event"] == "tool"]
    assert arrived == ["search_standards", "query_machine_stats",
                       "query_defect_history"], (
        "the fixture must actually complete out of plan order, or this test "
        "cannot catch the pairing bug"
    )

    run = analysis_store.get_run(events(body)[-1]["data"]["run_id"])
    assert [(r["tool"], r["why"]) for r in run["results"]] == [
        ("query_defect_history", "the slow one"),
        ("query_machine_stats", "the middling one"),
        ("search_standards", "the fast one"),
    ]

    page = client.get(f"/ask/{run['id']}").text
    for why in ("the slow one", "the middling one", "the fast one"):
        assert why in page


def test_the_streamed_run_and_the_posted_run_store_the_same_order(client, monkeypatch):
    """Two entrances, one function, so the two cannot drift again.

    `POST /ask` and `GET /ask/stream` run the identical graph and observe it
    differently. The stream used to reimplement `save_run` inline, which is
    where the ordering difference lived; both now go through
    `service.persist_run`. A reader who watched the stream and a reader who
    posted the form must be looking at the same document.
    """
    def slow(seconds, name):
        def call(**kw):
            time.sleep(seconds)
            return {"tool": name}
        return call

    monkeypatch.setitem(analysis.PLANNABLE_TOOLS, "query_machine_stats",
                        slow(0.20, "query_machine_stats"))
    monkeypatch.setitem(analysis.PLANNABLE_TOOLS, "search_standards",
                        slow(0.01, "search_standards"))

    streamed = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    streamed_run = analysis_store.get_run(events(streamed)[-1]["data"]["run_id"])

    posted = client.post("/ask", data={"question": "M22 正常嗎"},
                         follow_redirects=False)
    posted_run = analysis_store.get_run(int(posted.headers["location"].rsplit("/", 1)[1]))

    assert [(r["tool"], r["why"]) for r in streamed_run["results"]] == \
           [(r["tool"], r["why"]) for r in posted_run["results"]]
    assert [r["tool"] for r in streamed_run["results"]] == [
        "query_machine_stats", "search_standards"
    ]


def test_the_join_is_announced_before_the_answer_is_written(client):
    """The second dead zone had no event at all.

    Every tool ticks to a ✓ in about a fifth of a second, and then the second
    model call takes roughly as long as the first -- eight seconds on this
    machine -- with nothing on the wire. The panel read as finished while the
    run was half done. `synthesising` is emitted at the join, which is the
    `collect` node: under `stream_mode="updates"` every `run_tool` branch has
    already been streamed when it arrives, and `synthesise` has not been
    entered.
    """
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    names = [e["event"] for e in events(body)]

    assert names.count("synthesising") == 1
    assert names.index("synthesising") > max(
        index for index, name in enumerate(names) if name == "tool"
    ), "the join is announced after the last branch, never before one"
    assert names.index("synthesising") < names.index("done")


def test_the_join_event_names_how_many_branches_it_joined(client):
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    joined = next(e for e in events(body) if e["event"] == "synthesising")

    assert joined["data"]["tools"] == 2


def test_a_refusal_never_announces_a_synthesis_that_will_not_happen(client, monkeypatch):
    """A plan with no calls routes plan -> report, so no join and no second
    model call ever happen. Announcing one would be the same lie in the other
    direction: a phase on screen that the graph is not in."""
    refusal = {"interpretation": "資料涵蓋不到那個區間", "assumptions": [], "calls": []}
    monkeypatch.setattr(station_app, "_analysis_graph",
                        analysis.build_analysis_graph(StubClient(plan=refusal), DOMAINS))
    body = client.get("/ask/stream", params={"question": "去年的資料"}).text
    names = [e["event"] for e in events(body)]

    assert "synthesising" not in names
    assert names[-1] == "done"


def test_a_rejected_plan_never_announces_a_synthesis_either(client, monkeypatch):
    bad = {"interpretation": "i", "assumptions": [],
           "calls": [{"tool": "query_defect_history",
                      "args": {"line_id": "L9"}, "why": "w"}]}
    monkeypatch.setattr(station_app, "_analysis_graph",
                        analysis.build_analysis_graph(StubClient(plan=bad), DOMAINS))
    body = client.get("/ask/stream", params={"question": "L9"}).text

    assert "synthesising" not in [e["event"] for e in events(body)]


def test_the_join_is_on_the_wire_before_the_second_model_call_is_entered(
    client, monkeypatch
):
    """Ordering inside the finished body is not the claim.

    The claim is that the event leaves the server *before* the synthesis call
    is entered -- the only version of it that removes the dead zone, since the
    dead zone is that call. `TestClient.stream` cannot show this: its ASGI
    transport collects the whole body before handing back a line, so a test
    written over it passes just as happily against an event emitted after
    synthesis returns. Verified: written that way it took the full five-second
    timeout and still passed.

    So this drives the route's own generator instead, one chunk at a time, and
    asks what the stub client has been called with at the moment the chunk
    appears. `iterate_in_threadpool` is what `StreamingResponse` wraps a
    synchronous generator in, so stepping `body_iterator` advances the graph by
    exactly one yield.
    """
    import anyio

    entered: list[str] = []

    class Recording(StubClient):
        def chat(self, messages, **kwargs):
            entered.append("plan" if kwargs.get("response_format") else "synthesis")
            return StubClient.chat(self, messages, **kwargs)

    monkeypatch.setattr(
        station_app, "_analysis_graph",
        analysis.build_analysis_graph(Recording(), DOMAINS),
    )
    request = SimpleNamespace(state=SimpleNamespace(operator="tester"))

    async def drive() -> list[str]:
        response = station_app.ask_stream(request, "M22 正常嗎")
        seen_at = None
        async for chunk in response.body_iterator:
            if "event: synthesising" in chunk:
                seen_at = list(entered)
                break
        return seen_at

    at_the_event = anyio.run(drive)

    assert at_the_event is not None, "no join was announced at all"
    assert at_the_event == ["plan"], (
        "the join must be announced with the synthesis call still ahead of it, "
        f"not after it -- the model had been called with {at_the_event}"
    )


def test_the_streamed_run_is_planned_in_the_language_it_is_recorded_under(client, monkeypatch):
    """The stream path built its initial state without `lang`, so a question
    asked from the English station was recorded under `en` and planned and
    written in the default language -- the run said one thing and every
    sentence on it said another. The language has to reach the planner the
    same way `service.answer_question` sends it."""
    from conftest import read_in

    seen: list[list[dict]] = []
    stub = StubClient()
    original = stub.chat

    def recording_chat(messages, **kwargs):
        seen.append(messages)
        return original(messages, **kwargs)

    stub.chat = recording_chat
    monkeypatch.setattr(station_app, "_analysis_graph",
                        analysis.build_analysis_graph(stub, DOMAINS))

    read_in(client, "en").get("/ask/stream", params={"question": "is M22 high"})

    assert seen, "the stream never reached the planner"
    system = seen[0][0]["content"]
    assert "Write all prose you produce in English" in system
    assert "Traditional Chinese" not in system
