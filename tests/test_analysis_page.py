"""The analysis page: what it renders, and what it refuses to hide."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from aoi_agent.analysis import graph as analysis
from aoi_agent.llm.ollama import ChatResult, Timing
from aoi_agent.station import app as station_app
from aoi_agent.station.chart_svg import render_svg
from aoi_agent.station.result_view import (
    MAX_CHARS,
    MAX_LABEL,
    MAX_ROWS,
    error_text,
    readable_rows,
)
from aoi_agent.store.models import create_all, make_session_factory
from conftest import sign_in

PLAN = {
    "interpretation": "M22 against the fleet",
    "assumptions": ["compared with the fleet average over the whole span"],
    "calls": [{"tool": "query_machine_stats",
               "args": {"defect_type": "open", "days": 7}, "why": "fleet comparison"}],
}
DOMAINS = {"line_id": {"L2"}, "machine_id": {"M22"},
           "defect_type": {"open"}, "max_days": 9}

SPEC = {
    "kind": "bar", "title": "open share by machine",
    "x_label": "machine", "y_label": "share",
    "series": [{"name": "share of open",
                "points": [{"x": "L2-M22", "y": 0.32}, {"x": "L1-M11", "y": 0.19}]}],
}

#: Never a real value: if it appears on a page, something rendered a payload
#: it should not have.
SENTINEL = "TELLTALE-ANSWER-KEY"


@dataclass
class StubClient:
    plan: dict = field(default_factory=lambda: PLAN)
    answer: str = "M22 sits above the fleet on opens."

    def chat(self, messages, **kwargs) -> ChatResult:
        text = json.dumps(self.plan) if kwargs.get("response_format") else self.answer
        return ChatResult(text=text, tool_calls=[], thinking="",
                          timing=Timing(1.0, 0.0, 1.0, 1.0, 10, 10))


@pytest.fixture
def client(tmp_path, monkeypatch, operators):
    url = f"sqlite:///{tmp_path / 'a.db'}"
    create_all(url)
    monkeypatch.setattr("aoi_agent.store.boards._session_factory",
                        make_session_factory(url))
    monkeypatch.setitem(
        analysis.PLANNABLE_TOOLS, "query_machine_stats",
        lambda **kw: {"defect_type": "open", "fleet_share_of_defects": 0.2,
                      "machines": [{"machine": "L2-M22", "share_of_defects": 0.32,
                                    "per_board": 2.3}]},
    )
    monkeypatch.setattr(
        station_app, "_analysis_graph",
        analysis.build_analysis_graph(StubClient(), DOMAINS),
    )
    # `/ask` reads production statistics for the whole plant, which is the
    # exposure that made authentication a precondition rather than a backlog
    # item. Signed in through the real form; refusal is in test_attribution.py.
    return sign_in(TestClient(station_app.app))


def test_the_svg_draws_one_rect_per_point_and_labels_the_axes():
    svg = render_svg(SPEC)

    assert svg.count("<rect") >= 2
    assert "L2-M22" in svg
    assert "open share by machine" in svg
    assert "share" in svg


def test_the_svg_survives_a_zero_valued_series():
    """A flat series must not divide by zero on the way to a scale factor."""
    flat = {**SPEC, "series": [{"name": "s", "points": [{"x": "a", "y": 0.0}]}]}
    assert "<svg" in render_svg(flat)


def test_the_svg_returns_nothing_rather_than_raising_on_a_malformed_spec():
    """A specification comes back out of a JSON column, so this is not theory.

    A 500 here would cost the reader an answer whose figures are all still
    correct, in order to spare them a picture.
    """
    assert render_svg({}) == ""
    assert render_svg({"series": [{"name": "s", "points": [{"x": "a"}]}]}) == ""
    assert render_svg({"series": [{"name": "s", "points": [{"y": 1.0}]}]}) == ""
    assert render_svg({"series": [{"name": "s"}]}) == ""

    # A good series is still drawn when a later one is unusable.
    mixed = {**SPEC, "series": [*SPEC["series"], {"name": "broken", "points": []}]}
    assert "<rect" in render_svg(mixed)


def test_the_empty_page_shows_the_examples_and_the_coverage(client):
    page = client.get("/ask").text

    assert "L2-M22" in page, "an example question"
    assert "涵蓋" in page or "covers" in page.lower(), "the data span must be stated"


def test_asking_a_question_shows_all_five_blocks(client):
    response = client.post("/ask", data={"question": "M22 正常嗎"},
                           follow_redirects=True)
    page = response.text

    assert "M22 against the fleet" in page, "1. interpretation"
    assert "query_machine_stats" in page, "2. the calls"
    assert "fleet average" in page, "3. the assumptions"
    assert "ms" in page, "4. timing"
    assert "sits above the fleet" in page, "5. the prose"
    assert "<svg" in page, "the chart"


def test_the_page_shows_what_the_fan_out_cost_and_saved(client):
    page = client.post("/ask", data={"question": "M22 正常嗎"},
                       follow_redirects=True).text

    assert "parallel" in page.lower() or "平行" in page


def test_a_stored_run_renders_again_without_the_model(client, monkeypatch):
    """Once saved, a run is a document. Reopening it must not call anything."""
    run_id = client.post("/ask", data={"question": "M22 正常嗎"},
                         follow_redirects=True).url.path.rsplit("/", 1)[-1]

    class Exploding:
        def chat(self, *a, **k):
            raise AssertionError("the model must not be called to re-render")

    monkeypatch.setattr(station_app, "_analysis_graph",
                        analysis.build_analysis_graph(Exploding(), DOMAINS))
    page = client.get(f"/ask/{run_id}").text

    assert "<svg" in page
    assert "sits above the fleet" in page


def test_a_rejected_plan_is_shown_with_its_errors_and_no_chart(client, monkeypatch):
    bad = {"interpretation": "i", "assumptions": [],
           "calls": [{"tool": "query_defect_history",
                      "args": {"line_id": "L9", "days": 999}, "why": "w"}]}
    monkeypatch.setattr(station_app, "_analysis_graph",
                        analysis.build_analysis_graph(StubClient(plan=bad), DOMAINS))
    page = client.post("/ask", data={"question": "L9 呢"},
                       follow_redirects=True).text

    assert "L9" in page
    assert "<svg" not in page


def test_a_tool_that_answers_with_an_error_is_neither_a_success_nor_a_crash(
    client, monkeypatch
):
    """The production tools report trouble as `{"error": ...}`, not by raising.

    So the call succeeded and there is still nothing to plot. Rendering that as
    a green tick would be a lie, and rendering it as a crash would send someone
    to look for a fault in the station.
    """
    monkeypatch.setitem(
        analysis.PLANNABLE_TOOLS, "query_machine_stats",
        lambda **kw: {"error": "the store is empty; run scripts/seed_store.py"},
    )
    page = client.post("/ask", data={"question": "M22 正常嗎"},
                       follow_redirects=True).text

    assert "the store is empty" in page
    assert "<svg" not in page, "nothing plottable came back"
    assert "失敗" not in page, "the call did not fail"


def test_the_figures_the_answer_was_written_from_are_on_the_page(client):
    """The prose is a model's; the numbers are the store's. Both, or neither.

    Three of the five tools have no chart builder, so this block is the only
    thing a reader can check a confident wrong sentence against.
    """
    page = client.post("/ask", data={"question": "M22 正常嗎"},
                       follow_redirects=True).text

    assert "per_board" in page, "a returned field the chart never shows"
    assert "2.3" in page, "and its value"


class _Telltale:
    """An object whose repr carries the sentinel, as a stand-in for anything a
    future tool might return that is not JSON at all."""

    def __repr__(self) -> str:
        return SENTINEL

    __str__ = __repr__


#: Ways a sixth tool could hide the answer key in what it returns. Each is a
#: *fragment* -- one or two entries, no wrapper -- because the wrapper is the
#: other half of this table and the two are crossed below. Every leak found in
#: this module so far has been a shape that was already probed somewhere else
#: and not probed here, so the shape and the place it sits are varied
#: separately and combined exhaustively rather than by hand.
LEAKY_FRAGMENTS = [
    ("the exact key", {"ground_truth": SENTINEL}),
    ("a capitalised key", {"Ground_Truth": SENTINEL}),
    ("a padded key", {" ground_truth ": SENTINEL}),
    ("a hyphenated key", {"ground-truth": SENTINEL}),
    ("a spaced key", {"Ground Truth": SENTINEL}),
    ("shouted", {"GROUND_TRUTH": SENTINEL}),
    ("camelCase", {"groundTruth": SENTINEL}),
    ("camelCase capitalised", {"GroundTruth": SENTINEL}),
    ("run together", {"groundtruth": SENTINEL}),
    ("a key that is itself a path", {"candidate.ground_truth": SENTINEL}),
    # The name split across two levels. Whichever container does the splitting,
    # neither half is a hidden key on its own -- which is how the list form of
    # this shape reached the page while the dict form was filtered.
    ("split over a dict", {"ground": {"truth": SENTINEL}}),
    ("split over a list", {"ground": [{"truth": SENTINEL}]}),
    ("split over a list of scalars", {"ground": {"truth": [SENTINEL]}}),
    ("a hidden key holding a list", {"ground_truth": [SENTINEL]}),
    ("a hidden key holding a record", {"ground_truth": {"a": SENTINEL}}),
    ("a hidden key holding records", {"ground_truth": [{"a": SENTINEL}]}),
    ("a tuple", {"weird": ({"ground_truth": SENTINEL},)}),
    ("a non-string key", {0: {"ground_truth": SENTINEL}}),
    # Not JSON at all. These cannot reach the page -- the synthesis prompt
    # `json.dumps` the results, so the run raises before anything renders --
    # but `readable_rows` is still the boundary, so they are held to the same
    # claim at the function. The split below is made by asking `json`, not by
    # hand, so a fragment cannot be quietly excused from the page-level test.
    ("a set", {"weird": {SENTINEL}}),
    ("a tuple as a key", {("k", SENTINEL): 1}),
    ("an object as a key", {_Telltale(): 1}),
    ("an object with a telltale repr", {"weird": _Telltale()}),
    ("a list holding one", {"weird": [_Telltale()]}),
]

#: Every position a value can occupy in a tool's return value. A fragment is
#: dropped into each of them, so no position can be the one nobody thought of:
#: round 1 missed the list path, round 2 the record inside it, round 3 the
#: composed label in one branch of two.
POSITIONS = [
    ("at the top", lambda f: {"tool": "x", **f}),
    ("inside a nested dict", lambda f: {"meta": {"keep": 1, **f}}),
    ("inside a record in a list", lambda f: {"rows": [{"m": "x", **f}]}),
    ("as the whole of a record in a list", lambda f: {"rows": [dict(f)]}),
    ("inside a list of lists", lambda f: {"weird": [[dict(f)]]}),
    ("inside a dict of dicts", lambda f: {"a": {"b": dict(f)}}),
    ("inside a list held by a record", lambda f: {"rows": [{"m": "x", "h": [dict(f)]}]}),
]

#: The four positions the page renders. The rest are shapes this module drops
#: by design, and `test_every_probed_position_is_one_the_page_accounts_for`
#: holds them to that -- a leak table over positions that render nothing would
#: pass by doing nothing at all.
RENDERED_POSITIONS = {"at the top", "inside a nested dict",
                      "inside a record in a list",
                      "as the whole of a record in a list"}

#: shape x position, which is the claim: no shape leaks anywhere it can sit.
LEAK_CASES = [
    (f"{shape} {where}", place(fragment))
    for shape, fragment in LEAKY_FRAGMENTS
    for where, place in POSITIONS
]


def _serialisable(payload: object) -> bool:
    """Whether this payload can reach the page at all.

    `synthesise` `json.dumps` the tool results, so a set or a bare object
    raises there and the run never renders. That is a crash rather than a leak,
    and it belongs to the tool that would return one -- but the page-level test
    must not silently skip a shape that *is* serialisable, so the split is made
    by asking `json` rather than by hand.
    """
    try:
        json.dumps(payload)
    except TypeError:
        return False
    return True


PAGE_CASES = [(name, data) for name, data in LEAK_CASES if _serialisable(data)]


@pytest.mark.parametrize("data", [data for _, data in PAGE_CASES],
                         ids=[name for name, _ in PAGE_CASES])
def test_no_payload_shape_puts_the_answer_key_on_the_page(client, monkeypatch, data):
    """The station's hardest invariant, at the one boundary this task opens.

    Asserted on the rendered page rather than on the rows, because the claim is
    about what an operator can read, and a value can reach the page through a
    label, a repr or a nested container as easily as through a field.
    """
    monkeypatch.setitem(analysis.PLANNABLE_TOOLS, "query_machine_stats",
                        lambda **kw: data)
    page = client.post("/ask", data={"question": "M22 正常嗎"},
                       follow_redirects=True).text

    assert SENTINEL not in page
    assert "ground_truth" not in page.casefold()


@pytest.mark.parametrize("data", [data for _, data in LEAK_CASES],
                         ids=[name for name, _ in LEAK_CASES])
def test_no_payload_shape_survives_the_dict_boundary(data):
    """The same claim one layer down, where the guard actually lives.

    Every case, including the ones that cannot be serialised as far as a page.
    """
    rendered = str(readable_rows(data))

    assert SENTINEL not in rendered
    assert "ground_truth" not in rendered.casefold()


@pytest.mark.parametrize("where,place", POSITIONS,
                         ids=[where for where, _ in POSITIONS])
def test_every_probed_position_is_one_the_page_accounts_for(
    client, monkeypatch, where, place
):
    """A leak table over positions that render nothing would pass by doing nothing.

    So each position is checked with an ordinary value in it: either the page
    shows it, or the page says something was left out. Silently showing neither
    is the failure this test exists to catch.
    """
    marker = "PLAIN-READABLE-VALUE"
    monkeypatch.setitem(analysis.PLANNABLE_TOOLS, "query_machine_stats",
                        lambda **kw: place({"visible": marker}))
    page = client.post("/ask", data={"question": "M22 正常嗎"},
                       follow_redirects=True).text

    if where in RENDERED_POSITIONS:
        assert marker in page, "this position renders, so the leak test is real"
    else:
        assert marker not in page, "this position is dropped by design"
        assert "未顯示" in page, "and the reader is told it was"


def test_the_guard_drops_the_shape_it_cannot_read_and_keeps_the_rest(client):
    """Dropping everything would also pass the leak test, and be useless."""
    rows = readable_rows({"ok": 1, "ground_truth": SENTINEL,
                          "nested": {"ground_truth": SENTINEL, "keep": 2},
                          "rows": [{"ground_truth": SENTINEL, "keep": 3}],
                          "weird": [[SENTINEL]]})

    assert ("ok", "1") in rows
    assert ("nested.keep", "2") in rows
    assert ("rows[0]", "keep=3") in rows
    assert any("未顯示" in value for _, value in rows), "the omission is visible"


#: The five plannable tools' real return payloads, each read off the `return`
#: statement in its own source rather than from a schema note -- the previous
#: version of this table was fabricated for `list_candidates`, which is exactly
#: the tool the guard was dropping whole. Re-read at every round:
#: `mcp_servers/production.py:84,168,202`, `standards.py:32`, `classify.py:70`.
#:
#: `complete` is False only for `query_defect_history`: filters(5) + 4 scalars +
#: by_class(6) is 15 rows against MAX_ROWS = 14, so one class is always cut.
#: Known, and out of scope.
REAL_SHAPES = [
    (
        "query_defect_history",  # mcp_servers/production.py:84
        {
            "filters": {"lot_id": None, "line_id": "L2", "machine_id": None,
                        "defect_type": None, "days": 7},
            "window_end": "2026-08-22T03:00:00",
            "boards_inspected": 40,
            "defects_total": 91,
            "defects_per_board": 2.28,
            "by_class": {"open": 30, "short": 21, "mousebite": 15, "spur": 12,
                         "copper": 8, "pin-hole": 5},
        },
        ["defects_per_board", "2.28", "filters.line_id", "by_class.open"],
        False,
    ),
    (
        "query_machine_stats",  # mcp_servers/production.py:168
        {
            "defect_type": "open", "days": 7,
            "fleet_average_per_board": 1.42, "fleet_share_of_defects": 0.201,
            "machines": [
                {"machine": "L2-M22", "boards": 40, "defects": 92,
                 "per_board": 2.3, "share_of_defects": 0.32},
            ],
        },
        ["share_of_defects=0.32", "machine=L2-M22", "fleet_share_of_defects"],
        True,
    ),
    (
        "query_board_context",  # mcp_servers/production.py:202
        {
            "board": "20085294", "lot_id": "LOT-2026-08-14", "line_id": "L2",
            "machine_id": "M22", "shift": "B",
            "inspected_at": "2026-08-22T01:12:00",
            "lot_boards": 12, "lot_defects": 30, "lot_defects_per_board": 2.5,
        },
        ["lot_defects_per_board", "2.5", "LOT-2026-08-14"],
        True,
    ),
    (
        "search_standards",  # mcp_servers/standards.py:32
        {
            "query": "open",
            "passages": [
                {"document": "WI-300", "heading": "Opens",
                 "text": "any confirmed open is critical", "distance": 0.2143},
            ],
        },
        ["any confirmed open is critical", "document=WI-300", "distance=0.2143"],
        True,
    ),
    (
        "list_candidates",  # mcp_servers/classify.py:70
        {
            "board": "20085294", "candidate_count": 2,
            "candidates": [
                {"candidate_ref": "20085294#3", "box": [10, 20, 30, 40],
                 "predicted_class": "open", "confidence": 0.9134},
                {"candidate_ref": "20085294#7", "box": [50, 60, 70, 80],
                 "predicted_class": "mousebite", "confidence": 0.4102},
            ],
        },
        # `box` is a list inside a record -- the only nested list any tool
        # returns, and the shape that voided this whole payload in round 2.
        ["candidate_ref=20085294#3", "box=[10, 20, 30, 40]",
         "predicted_class=open", "confidence=0.9134", "20085294#7"],
        True,
    ),
]


@pytest.mark.parametrize("data,expected,complete",
                         [(d, e, c) for _, d, e, c in REAL_SHAPES],
                         ids=[name for name, _, _, _ in REAL_SHAPES])
def test_every_real_tool_payload_still_renders(data, expected, complete):
    """A whitelist that is too strict fails silently, by showing the reader less.

    The payloads are the real ones. A fabricated payload here passed while
    `list_candidates` rendered nothing but a board and a count.
    """
    rendered = str(readable_rows(data))

    for fragment in expected:
        assert fragment in rendered
    if complete:
        assert "未顯示" not in rendered, "nothing a real tool returns is dropped"


@pytest.mark.parametrize("data,expected,complete",
                         [(d, e, c) for _, d, e, c in REAL_SHAPES],
                         ids=[name for name, _, _, _ in REAL_SHAPES])
def test_every_real_tool_payload_reaches_the_page(client, monkeypatch,
                                                  data, expected, complete):
    """And reaches the reader, not just the function that builds the rows."""
    monkeypatch.setitem(analysis.PLANNABLE_TOOLS, "query_machine_stats",
                        lambda **kw: data)
    page = client.post("/ask", data={"question": "M22 正常嗎"},
                       follow_redirects=True).text

    for fragment in expected:
        assert fragment in page


def test_a_hidden_key_nested_in_a_record_is_counted_not_silently_skipped():
    """Every omission is visible, at whatever depth it happened."""
    nested = readable_rows({"meta": {"ground_truth": SENTINEL, "keep": 2}})
    in_record = readable_rows({"rows": [{"ground_truth": SENTINEL, "keep": 3}]})

    assert ("meta.keep", "2") in nested
    assert any("未顯示" in value for _, value in nested), "the nested drop is counted"
    assert ("rows[0]", "keep=3") in in_record
    assert any("未顯示" in value for _, value in in_record), "and inside a record"


#: Long enough that an unclipped row would be a page on its own. 200 items and
#: 300 characters each is 60KB if nothing cuts it.
FLOOD = ["y" * 300] * 200

#: The same flood on the other side of the row. A cap that holds only for
#: values is one a payload walks around by putting its bulk in the keys: forty
#: 500-character keys rendered a 17.6KB block.
LONG_KEYS = {f"K{index}" * 500: index for index in range(40)}

#: Each way a payload can be big, and where the bulk sits in it.
FLOODS = [
    ("a list at the top", {"meta": 1, "hist": FLOOD}),
    ("the same list inside a record", {"meta": {"hist": FLOOD}}),
    ("a list of long records",
     {"query": "open",
      "passages": [{"document": "WI-300", "heading": "Opens",
                    "text": "z" * 400, "distance": 0.2}] * 9}),
    ("long keys at the top", LONG_KEYS),
    ("long keys inside a record", {"meta": dict(LONG_KEYS)}),
    ("long keys inside a record in a list", {"rows": [dict(LONG_KEYS)]}),
]


@pytest.mark.parametrize("data", [data for _, data in FLOODS],
                         ids=[where for where, _ in FLOODS])
def test_the_data_view_truncates_rather_than_floods(data):
    """A block that floods the page is one nobody reads.

    Both list paths, because the round before last clipped one of them and
    deleted this test rather than extending it: a list that is a value of the
    payload, and the same list one level down as a field of a record -- which
    is where it went through unclipped, 10.4KB in a single row. And both sides
    of the row, because a cap on values only is one a payload steps around by
    putting its bulk in the keys.
    """
    rows = readable_rows(data)

    assert rows
    assert len(rows) <= MAX_ROWS + 1
    assert all(len(label) <= MAX_LABEL for label, _ in rows), "an unclipped label"
    assert all(len(value) <= MAX_CHARS for _, value in rows), "an unclipped value"
    assert len("".join(label + value for label, value in rows)) < 3000
    # Every cut says so, in one of the module's two ways: an ellipsis where a
    # value or a label was clipped, a counted row where something was dropped
    # whole. A payload that came back smaller with neither is the failure.
    assert any(
        "未顯示" in value or "…" in value or "…" in label for label, value in rows
    ), "cut without saying so"


@pytest.mark.parametrize("data", [data for _, data in FLOODS],
                         ids=[where for where, _ in FLOODS])
def test_a_flooding_payload_does_not_flood_the_rendered_page(client, monkeypatch,
                                                             data):
    """The same claim where it is felt: the block on the operator's screen."""
    monkeypatch.setitem(analysis.PLANNABLE_TOOLS, "query_machine_stats",
                        lambda **kw: data)
    page = client.post("/ask", data={"question": "M22 正常嗎"},
                       follow_redirects=True).text

    block = page[page.index('<details class="data"'):]
    block = block[: block.index("</details>")]
    assert len(block) < 4000, "the returned-data block is a page of its own"
    assert "未顯示" in block or "…" in block, "cut without saying so"


def test_a_pathological_key_renders_in_bounded_time():
    """A key is whatever a tool returned, so its cost is not a hypothetical.

    Checking every contiguous run of a split key was quadratic in its segments
    and cubic with the normalising: `{"a." * 2000: 1}` took 151s on this
    machine to render one row -- a synchronous hang of `GET /ask/{id}` bought
    with one dotted key -- and 800 segments took 9.0s. Bounded runs and an
    index rather than a tail slice make it 2.7ms and 1.2ms. The budget is two
    seconds because the machine is fanless and throttles; the gap being
    guarded is four orders of magnitude, not a factor of two.
    """
    for segments in (2000, 20000):
        started = time.perf_counter()
        rows = readable_rows({"a." * segments: 1})
        elapsed = time.perf_counter() - started

        assert rows, segments
        assert all(len(label) <= MAX_LABEL for label, _ in rows), segments
        assert elapsed < 2.0, f"{segments} segments took {elapsed:.1f}s"


def test_a_tool_error_is_clipped_like_everything_else_on_the_page(client,
                                                                  monkeypatch):
    """It is printed outside the rows block, so nothing else caps it."""
    monkeypatch.setitem(
        analysis.PLANNABLE_TOOLS, "query_machine_stats",
        lambda **kw: {"error": "the store is empty. " + "e" * 1_000_000},
    )
    page = client.post("/ask", data={"question": "M22 正常嗎"},
                       follow_redirects=True).text

    assert len(error_text({"error": "e" * 1_000_000})) <= MAX_CHARS
    assert "the store is empty" in page, "the reader still gets the message"
    assert len(page) < 20_000, "a megabyte of error is not a page"


def test_a_tool_error_that_is_not_a_string_is_not_str_ed_onto_the_page(
    client, monkeypatch
):
    """Every error path in the repo returns a string today. One edit from not."""
    monkeypatch.setitem(
        analysis.PLANNABLE_TOOLS, "query_machine_stats",
        lambda **kw: {"error": {"ground_truth": SENTINEL}},
    )
    page = client.post("/ask", data={"question": "M22 正常嗎"},
                       follow_redirects=True).text

    assert SENTINEL not in page
    assert "ground_truth" not in page.casefold()
    assert "無法顯示的錯誤" in page, "the reader is still told the tool reported one"


def test_an_empty_question_is_refused_before_the_model_is_asked(client):
    response = client.post("/ask", data={"question": "   "}, follow_redirects=False)
    assert response.status_code == 400


def test_an_unknown_run_is_a_404(client):
    assert client.get("/ask/99999").status_code == 404


# ---------------------------------------------------------------------------
# The second route out of a tool's payload.
# ---------------------------------------------------------------------------


def test_the_synthesis_prompt_is_filtered_by_the_same_rule_as_the_table():
    """`readable_rows` guards the table. It does not guard the page.

    The synthesis prompt serialises the raw payload into the model's context
    and `run.answer` is rendered verbatim, so a field the table drops can still
    reach a reader in a sentence -- a model told "describe what the results
    show" will reproduce whatever it was shown. Filtering one route and not the
    other enforces the invariant on neither.
    """
    from aoi_agent.analysis.prompts import build_synthesis_messages

    results = [{
        "tool": "list_candidates", "args": {"board": "20085294"}, "why": "w",
        "position": 0, "ok": True, "elapsed_ms": 1.0, "error": None,
        "data": {"regions": [{"index": 1, "ground_truth": SENTINEL,
                              "score": 0.9}]},
    }]
    body = build_synthesis_messages("q", PLAN, results)[-1]["content"]

    assert SENTINEL not in body
    assert "ground_truth" not in body
    assert "0.9" in body, "the rest of the payload must still reach the model"


def test_the_hidden_key_spellings_the_table_catches_are_caught_here_too():
    """One rule, two routes. `strip_hidden` shares `_is_hidden` with the walk
    precisely so the two cannot come to disagree about how `ground_truth` is
    spelled -- a second implementation is the drift this exists to prevent."""
    from aoi_agent.station.result_view import strip_hidden

    for payload in (
        {"groundTruth": SENTINEL},
        {"Ground-Truth": SENTINEL},
        {"groundtruth": SENTINEL},
        {"ground": {"truth": SENTINEL}},
        {"ground": [{"truth": SENTINEL}]},
        {"rows": [{"ground_truth": SENTINEL}]},
    ):
        assert SENTINEL not in json.dumps(strip_hidden(payload)), payload


def test_a_leaked_field_would_reach_the_answer_if_the_prompt_were_unfiltered(
    client, monkeypatch
):
    """The end-to-end shape of the leak, so the guard above is not a unit test
    of itself: a tool returns the key, the model repeats what it was given, and
    the page prints the answer verbatim.
    """
    leaking = f"The board's recorded label was {SENTINEL}."
    monkeypatch.setattr(
        station_app, "_analysis_graph",
        analysis.build_analysis_graph(StubClient(answer=leaking), DOMAINS),
    )
    response = client.post("/ask", data={"question": "M22 正常嗎"},
                           follow_redirects=True)

    # The stub answers with the sentinel regardless of its input -- this asserts
    # the page does print `run.answer` as written, which is why the filter has
    # to be on what the model is shown rather than on what it writes.
    assert SENTINEL in response.text


# ---------------------------------------------------------------------------
# Cross-file agreement.
# ---------------------------------------------------------------------------


def test_the_stated_coverage_is_the_one_the_validator_enforces(client, monkeypatch):
    """The page's "資料涵蓋最近 N 天" and the validator's `days` limit have to be
    the same N.

    The graph freezes its domains into the plan node at first use; the page used
    to re-read `store_domains()` on every render. After a reseed the two
    disagreed, and the page's number was the one nobody was enforcing -- a
    supervisor told the store holds four days while a nine-day plan validates,
    or the reverse. Both now read one snapshot.
    """
    monkeypatch.setattr(station_app, "_analysis_domains", None)
    calls = []

    def counted():
        calls.append(1)
        return {"line_id": {"L2"}, "machine_id": {"M22"},
                "defect_type": {"open"}, "max_days": 42}

    monkeypatch.setattr(station_app, "store_domains", counted)

    first = client.get("/ask").text
    assert "資料涵蓋最近 42 天" in first

    # The store moves under a running station. The page must not start quoting
    # a coverage the validator in the built graph knows nothing about.
    monkeypatch.setattr(station_app, "store_domains",
                        lambda: {"line_id": set(), "machine_id": set(),
                                 "defect_type": set(), "max_days": 3})
    assert "資料涵蓋最近 42 天" in client.get("/ask").text
    assert len(calls) == 1, "the snapshot is read once, not per render"


def test_the_item_count_beside_a_payload_does_not_count_the_omission_note():
    """"回傳的資料（N 項）" is a count of what the tool returned. The
    "另外 n 項未顯示" line is a statement *about* that payload, not a field of
    it, and counting it told a reader there were more items than there are."""
    from aoi_agent.station.result_view import shown_count

    wide = {f"field_{i}": i for i in range(MAX_ROWS + 6)}
    rows = readable_rows(wide)

    assert len(rows) == MAX_ROWS + 1, "the omission note is one of the rows"
    assert shown_count(rows) == MAX_ROWS
    assert rows[-1][0] == "…"


def test_the_page_prints_the_uncounted_note_and_the_honest_count(client, monkeypatch):
    monkeypatch.setitem(
        analysis.PLANNABLE_TOOLS, "query_machine_stats",
        lambda **kw: {f"field_{i}": i for i in range(MAX_ROWS + 6)},
    )
    page = client.post("/ask", data={"question": "很多欄位"},
                       follow_redirects=True).text

    assert f"回傳的資料（{MAX_ROWS} 項）" in page
    assert "另外 6 項未顯示" in page


def _progress_panel(page: str) -> str:
    """The markup between the panel's open tag and its closing `</div>`.

    Crude on purpose: the point of the assertions below is that everything the
    waiting experience adds sits inside a container carrying `hidden`, and a
    parser that understood the nesting would let a new element outside it pass
    unnoticed as long as it was well-formed.
    """
    start = page.index('<div class="progress"')
    return page[start:page.index("<p class=\"sub\">試試看", start)]


def test_the_waiting_panel_is_hidden_and_nothing_it_adds_escapes_it(client):
    """The stream is an enhancement, and enhancement means a reader with
    scripting off sees exactly what they saw before it existed. Every element
    the spinner, the timer and the flow view need lives inside one container
    that ships `hidden`; nothing was added to the page proper."""
    page = client.get("/ask").text
    panel = _progress_panel(page)

    assert panel.startswith('<div class="progress" id="progress" hidden>')
    for element in ("progress-mark", "progress-elapsed", "progress-flow",
                    "progress-question", "progress-list", "progress-head"):
        assert page.count(f'id="{element}"') == 1
        assert element in panel, "every added element is inside the hidden panel"


def test_the_page_without_scripting_still_answers(client):
    """The two entrances that involve no JavaScript at all: the form posts and
    the stored run renders."""
    response = client.post("/ask", data={"question": "M22 正常嗎"},
                           follow_redirects=False)
    assert response.status_code == 303

    page = client.get(response.headers["location"]).text
    for block in ("M22 against the fleet", "query_machine_stats",
                  "sits above the fleet"):
        assert block in page
    assert _progress_panel(page).startswith(
        '<div class="progress" id="progress" hidden>'
    )


def test_the_phase_is_readable_by_something_that_cannot_see_a_spinner(client):
    """A spinning glyph is not a phase. The status region carries the words and
    is announced; the spinner and the diagram are hidden from the accessibility
    tree so the state is read once, not twice and not as a decoration."""
    panel = _progress_panel(client.get("/ask").text)

    at = panel.index('id="progress-head"')
    head = panel[max(0, at - 200):at + 120]
    assert 'role="status"' in head and 'aria-live="polite"' in head
    for decoration in ('id="progress-mark"', 'id="progress-elapsed"',
                       'id="progress-flow"'):
        at = panel.index(decoration)
        line = panel[max(0, at - 120):at + 120]
        assert 'aria-hidden="true"' in line, f"{decoration} must not be announced"


def test_the_elapsed_count_reaches_a_listener_without_being_read_out_every_second(
    client
):
    """The visible counter ticks twice a second and is `aria-hidden`; a live
    region that re-announced it would bury the phase it belongs to. The spoken
    copy is a visually hidden span inside the status region, rewritten when the
    phase changes -- so a listener gets "writing the answer, 9 seconds in" once
    rather than a reading of the clock."""
    panel = _progress_panel(client.get("/ask").text)
    css = client.get("/static/style.css").text

    assert 'id="progress-since"' in panel and 'class="sr-only"' in panel
    at = panel.index('id="progress-since"')
    assert 'id="progress-head"' in panel[:at], "the spoken copy is inside the region"
    assert ".sr-only" in css and "clip: rect(0 0 0 0)" in css
    assert "display: none" not in css.split(".sr-only")[1].split("}")[0], (
        "hidden from the screen, not from the accessibility tree"
    )


def test_the_flow_view_is_vendored_and_served_from_the_station(client):
    """No CDN, and nothing fetched from anywhere. The station runs on a
    locked-down shop-floor browser."""
    page = client.get("/ask").text
    assert '<script src="/static/flow.js"></script>' in page

    served = client.get("/static/flow.js")
    assert served.status_code == 200
    assert "createElementNS" in served.text

    for external in ("http://", "https://", "//cdn", "unpkg", "jsdelivr"):
        assert external not in served.text.replace(
            "http://www.w3.org/2000/svg", ""
        ), "the SVG namespace is the only URL allowed in here"


def test_reduced_motion_keeps_the_state_and_drops_only_the_movement(client):
    """"Respect `prefers-reduced-motion`" cannot mean showing nothing: the
    spinner is the mark that says which row is still running. Under the query
    it stops turning and the active stage keeps a heavier outline instead of
    breathing -- the movement goes, the state stays."""
    css = client.get("/static/style.css").text
    assert css.count("@media (prefers-reduced-motion: reduce)") == 1

    reduced = css.split("@media (prefers-reduced-motion: reduce)")[1]
    reduced = reduced[:reduced.index("\n}")]
    assert ".spin { animation: none; }" in reduced
    assert "display: none" not in reduced, "the state must survive the query"
    assert "stroke-width" in reduced, "the active stage stays distinguishable"
