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


def test_a_ground_truth_key_never_reaches_the_page(client, monkeypatch):
    """The station's hardest invariant, at the one boundary this task opens.

    No tool returns `ground_truth` today. This asserts that the page would not
    show it if one did, because the filter is in Python and not in a reviewer's
    memory.
    """
    monkeypatch.setitem(
        analysis.PLANNABLE_TOOLS, "query_machine_stats",
        lambda **kw: {"defect_type": "open", "ground_truth": "TELLTALE-TOP",
                      "fleet_share_of_defects": 0.2,
                      "machines": [{"machine": "L2-M22", "share_of_defects": 0.32,
                                    "per_board": 2.3,
                                    "ground_truth": "TELLTALE-NESTED"}]},
    )
    page = client.post("/ask", data={"question": "M22 正常嗎"},
                       follow_redirects=True).text

    assert "ground_truth" not in page
    assert "TELLTALE-TOP" not in page
    assert "TELLTALE-NESTED" not in page
    assert "per_board" in page, "the rest of the payload is still shown"


def test_the_data_view_is_filtered_at_the_dict_boundary():
    """Not by grepping the HTML: the guard lives where the dict is shaped."""
    rows = readable_rows({"ok": 1, "ground_truth": "x",
                          "nested": {"ground_truth": "x", "keep": 2},
                          "rows": [{"ground_truth": "x", "keep": 3}]})

    assert "ground_truth" not in [label for label, _ in rows]
    assert not any("ground_truth" in value for _, value in rows)
    assert not any("x" == value for _, value in rows)


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
