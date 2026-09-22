"""WI-201's release clause has one Chinese rendering that reverses it, and the
disposition path's prompt names it -- the planner's prompt does not change.

The note exists because 5 of the 2026-09-02 run's 60 Chinese rationales wrote
"measures continuous on electrical test" as 連續開路, a confirmed open, and
released it. It is appended only where the rationale is written: `LANGUAGE_NOTE`
is shared with `/ask`'s planning and synthesis prompts, whose figures were
measured under the note as it stands, and one sentence added there would move
a measurement taken on the same night. `AOI_RATIONALE_GLOSSARY=0` is the
control arm, the way `AOI_SQL_TOOL=0` is the SQL tool's.
"""

from __future__ import annotations

import pytest

from aoi_agent.analysis.prompts import build_planning_messages
from aoi_agent.graph import flow
from aoi_agent.i18n import LANGUAGE_NOTE, RATIONALE_GLOSSARY, RATIONALE_GLOSSARY_ENV

DOMAINS = {"line_id": ["L1"], "machine_id": ["M11"], "defect_type": ["open"],
           "relative_to": [], "max_days": 9, "date_span": ["2026-08-01", "2026-08-09"]}


@pytest.fixture
def glossary_on(monkeypatch):
    monkeypatch.delenv(RATIONALE_GLOSSARY_ENV, raising=False)


def test_the_chinese_rationale_is_told_what_continuous_means(glossary_on):
    prompt = flow.reason_system_prompt("zh-TW")
    assert RATIONALE_GLOSSARY["zh-TW"] in prompt
    assert "連續開路" in prompt and "導通" in prompt


def test_the_control_arm_leaves_the_note_out(monkeypatch):
    monkeypatch.setenv(RATIONALE_GLOSSARY_ENV, "0")
    prompt = flow.reason_system_prompt("zh-TW")
    assert RATIONALE_GLOSSARY["zh-TW"] not in prompt
    assert prompt == f"{flow.SYSTEM_PROMPT}\n\n{LANGUAGE_NOTE['zh-TW']}"


def test_the_english_prompt_is_unchanged_character_for_character(glossary_on):
    """English wrote the clause the right way round on 120 of 120; nothing is added."""
    assert RATIONALE_GLOSSARY["en"] == ""
    assert flow.reason_system_prompt("en") == f"{flow.SYSTEM_PROMPT}\n\n{LANGUAGE_NOTE['en']}"


def test_the_planner_prompt_does_not_carry_the_note(glossary_on):
    messages = build_planning_messages("M11 的 open 多嗎？", DOMAINS, "zh-TW")
    text = "\n".join(m["content"] for m in messages)
    assert RATIONALE_GLOSSARY["zh-TW"] not in text
