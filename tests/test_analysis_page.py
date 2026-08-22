"""The analysis page: what it renders, and what it refuses to hide."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from aoi_agent.analysis import graph as analysis
from aoi_agent.llm.ollama import ChatResult, Timing
from aoi_agent.station import app as station_app
from aoi_agent.station.chart_svg import render_svg
from aoi_agent.station.result_view import readable_rows
from aoi_agent.store.models import create_all, make_session_factory

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
def client(tmp_path, monkeypatch):
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
    return TestClient(station_app.app)


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


#: Shapes a sixth tool could plausibly return, each hiding the answer key
#: somewhere the previous name-matching guard did not look. Adding one is a
#: line, which is the point: the claim being tested is about every shape, not
#: about the two that were found.
LEAKY_SHAPES = [
    ("the exact key", {"ground_truth": SENTINEL}),
    ("a capitalised key", {"Ground_Truth": SENTINEL}),
    ("a padded key", {" ground_truth ": SENTINEL}),
    ("a hyphenated key", {"ground-truth": SENTINEL}),
    ("a spaced key", {"Ground Truth": SENTINEL}),
    ("shouted", {"GROUND_TRUTH": SENTINEL}),
    ("inside a record", {"rows": [{"machine": "L2-M22", "ground_truth": SENTINEL}]}),
    ("inside a nested dict", {"meta": {"keep": 1, "ground_truth": SENTINEL}}),
    ("three levels deep", {"a": {"b": {"ground_truth": SENTINEL}}}),
    ("a list of lists", {"weird": [[{"ground_truth": SENTINEL}], "other"]}),
    ("a list of lists of scalars", {"weird": [[SENTINEL]]}),
    ("a record holding a list", {"rows": [{"m": "x", "history": [SENTINEL]}]}),
    ("a record holding a dict", {"rows": [{"m": "x", "gt": {"ground_truth": SENTINEL}}]}),
    ("a dict of dicts", {"per_machine": {"M22": {"ground_truth": SENTINEL}}}),
    ("a tuple", {"weird": ({"ground_truth": SENTINEL},)}),
    ("a non-string key", {0: {"ground_truth": SENTINEL}}),
]

#: The same question asked of shapes that are not JSON at all. These cannot be
#: asserted on the page: the synthesis prompt `json.dumps` the results, so a set
#: or a bare object raises there and the run never renders. That is a crash
#: rather than a leak, and it belongs to the tools that would return one -- but
#: `readable_rows` is still the boundary, so it is held to the same claim here.
UNSERIALISABLE_SHAPES = [
    ("a set", {"weird": {SENTINEL}}),
    ("an object with a telltale repr", {"weird": _Telltale()}),
    ("a list holding one", {"weird": [_Telltale()]}),
    ("a record holding one", {"rows": [{"m": "x", "obj": _Telltale()}]}),
]


@pytest.mark.parametrize("data", [shape for _, shape in LEAKY_SHAPES],
                         ids=[name for name, _ in LEAKY_SHAPES])
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


@pytest.mark.parametrize(
    "data", [shape for _, shape in LEAKY_SHAPES + UNSERIALISABLE_SHAPES],
    ids=[name for name, _ in LEAKY_SHAPES + UNSERIALISABLE_SHAPES],
)
def test_no_payload_shape_survives_the_dict_boundary(data):
    """The same claim one layer down, where the guard actually lives."""
    rendered = str(readable_rows(data))

    assert SENTINEL not in rendered
    assert "ground_truth" not in rendered.casefold()


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


#: The five plannable tools' real return shapes, from
#: `.superpowers/sdd/2026-08-22-analysis-interface/real-tool-schemas.md`. A
#: whitelist that is too strict fails silently -- the page just shows less --
#: so each one is asserted to still render something a reader can check.
REAL_SHAPES = [
    ("query_defect_history",
     {"filters": {"line_id": "L2", "days": 7}, "window_end": "2026-08-22",
      "boards_inspected": 40, "defects_total": 91, "defects_per_board": 2.3,
      "by_class": {"open": 30, "short": 12}},
     "defects_per_board"),
    ("query_machine_stats",
     {"defect_type": "open", "days": 7, "fleet_average_per_board": 1.4,
      "fleet_share_of_defects": 0.2,
      "machines": [{"machine": "L2-M22", "per_board": 2.3,
                    "share_of_defects": 0.32}]},
     "share_of_defects"),
    ("query_board_context",
     {"board": "20085294", "lot_id": "LOT-1", "line_id": "L2",
      "machine_id": "M22", "shift": "A", "inspected_at": "2026-08-22T01:00:00",
      "lot_boards": 12, "lot_defects": 30, "lot_defects_per_board": 2.5},
     "lot_defects_per_board"),
    ("search_standards",
     {"query": "open", "passages": [{"document": "WI-300", "heading": "opens",
                                     "text": "any confirmed open is critical",
                                     "distance": 0.21}]},
     "any confirmed open is critical"),
    ("list_candidates",
     {"reference": "20085294#3", "candidates": [{"index": 3, "x": 10, "y": 20,
                                                 "predicted_class": "open"}]},
     "predicted_class"),
]


@pytest.mark.parametrize("data,expected", [(d, e) for _, d, e in REAL_SHAPES],
                         ids=[name for name, _, _ in REAL_SHAPES])
def test_every_real_tool_shape_still_renders(data, expected):
    rendered = str(readable_rows(data))

    assert expected in rendered
    assert "未顯示" not in rendered, "nothing a real tool returns should be dropped"


def test_the_data_view_truncates_rather_than_floods():
    """A passage from `search_standards` is paragraphs long; a block nobody
    reads is worth the same as no block."""
    rows = readable_rows({"passages": [{"text": "word " * 200}] * 9})

    assert len(rows) <= 8, "nine passages must not become nine paragraphs"
    assert all(len(value) <= 400 for _, value in rows)
    assert any("另外" in value for _, value in rows), "the omission must be visible"


def test_an_empty_question_is_refused_before_the_model_is_asked(client):
    response = client.post("/ask", data={"question": "   "}, follow_redirects=False)
    assert response.status_code == 400


def test_an_unknown_run_is_a_404(client):
    assert client.get("/ask/99999").status_code == 404
