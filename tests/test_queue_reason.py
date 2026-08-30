"""The queue shows why a region was handed over, and no more than two
sentences of it at a glance.

The rationale is a paragraph the model wrote for the region page. On the queue
it sat whole in one cell, so a page of thirty rows was thirty paragraphs and
the column an operator scans to pick the next region was the one they could not
scan. The first two sentences stay on the row; the rest opens on demand. Nothing
is cut: the region page still shows the whole rationale, and so does the
disclosure.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from aoi_agent.station import app as station_app
from aoi_agent.station.prose import lead_and_rest
from aoi_agent.store import boards, escalations
from aoi_agent.store.models import Board, CandidateRecord, create_all, make_session_factory
from conftest import sign_in

LONG = (
    "模型判定 open，信心 0.61。這個區域在焊墊邊緣，差異圖上的斷點不完整。"
    "WI-201 規定任何確認的 open 都是 critical。建議人工確認斷點是否貫穿。"
    "若無法判斷可量測線寬比。"
)
SHORT = "模型判定 open，信心 0.61。"


def test_the_lead_is_the_first_two_sentences_in_either_language():
    lead, rest = lead_and_rest(LONG)
    assert lead == "模型判定 open，信心 0.61。這個區域在焊墊邊緣，差異圖上的斷點不完整。"
    assert rest.startswith("WI-201")
    assert lead + rest == LONG

    lead, rest = lead_and_rest("The model said open. Confidence 0.61! Check the pad. Then measure.")
    assert lead == "The model said open. Confidence 0.61!"
    assert rest == " Check the pad. Then measure."


def test_a_short_rationale_has_no_rest():
    assert lead_and_rest(SHORT) == (SHORT, "")
    assert lead_and_rest("") == ("", "")


@pytest.fixture
def queue(tmp_path, monkeypatch, operators):
    url = f"sqlite:///{tmp_path / 'q.db'}"
    create_all(url)
    factory = make_session_factory(url)
    monkeypatch.setattr(boards, "_session_factory", factory)
    with factory() as session:
        board = Board(stem="90000002", split="test", lot_id="LOT-9", line_id="L2",
                      machine_id="M22", shift="A", inspected_at=datetime(2026, 8, 25, 8, 0))
        session.add(board)
        session.flush()
        for index in range(2):
            session.add(CandidateRecord(
                board_id=board.id, index_on_board=index, x1=10, y1=10, x2=40, y2=40,
                area=900, predicted_class="open", confidence=0.61,
                false_call_probability=0.39, ground_truth="open",
            ))
        session.commit()
    escalations.raise_escalation("90000002#0", "t-0", LONG, None, "ok")
    escalations.raise_escalation("90000002#1", "t-1", SHORT, None, "ok")
    return sign_in(TestClient(station_app.app))


def test_the_queue_shows_two_sentences_and_folds_the_rest(queue):
    page = queue.get("/").text
    assert "差異圖上的斷點不完整。" in page
    folded = page.index("<details")
    assert page.index("WI-201 規定") > folded, "the third sentence is inside the disclosure"
    assert page.index("差異圖上的斷點不完整。") < folded, "the lead is outside it"
    assert page.count("<details") == 1, "a short rationale gets no disclosure"


def test_the_queue_says_which_language_the_rationale_is_written_in_and_what_it_costs(queue, monkeypatch):
    from aoi_agent.i18n import LINE_LANGUAGE_ENV

    monkeypatch.delenv(LINE_LANGUAGE_ENV, raising=False)
    page = queue.get("/").text
    assert "以中文撰寫" in page and "17 秒" in page and "12 秒" in page
    monkeypatch.setenv(LINE_LANGUAGE_ENV, "en")
    page = queue.get("/").text
    assert "以英文撰寫" in page


def test_a_markdown_rationale_reaches_the_queue_as_elements_not_asterisks(queue, monkeypatch):
    """The re-run of 2026-08-30 wrote outlines; the row printed the asterisks."""
    from aoi_agent.store import escalations
    monkeypatch.setattr(escalations, "pending", lambda limit=200: [{
        "reference": "20085299#2", "board_stem": "20085299", "index": 2,
        "thread_id": "20085299#2", "status": "pending", "raised_at": "2026-08-30T04:38:00",
        "reason": "以下為說明：\n\n1. **視覺模型輸出**\n   - 置信度 0.858。\n2. **生產背景**\n   - 該板屬於 LOT-1。\n   - 這批平均 5 個缺陷。\n3. **結論**\n   - 分類為 *false_call*。",
        "agent_verdict": "false_call", "explanation_status": "ok", "rationale_flags": [],
        "model_class": "false_call", "model_confidence": 0.858, "false_call_probability": 0.858,
        "lot_id": "LOT-1", "line_id": "L1", "machine_id": "M11", "shift": "C",
    }])
    page = queue.get("/").text
    assert "**" not in page and "*false_call*" not in page
    assert "<strong>視覺模型輸出</strong>" in page and "<em>false_call</em>" in page
