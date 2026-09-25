"""The language note has two measurable arms, and the eval can pin its batch.

Four planner readings of the independent seventy straddle the 2026-09-02
change to `LANGUAGE_NOTE`, and the 09-19 entry read the spread as drift on
unchanged code. The switch here is what lets the two sentences be compared
inside one session. What would silently spoil that comparison:

* a misspelled arm name running the current sentence under the old label;
* the planner reading the note through a path the switch does not reach;
* the eval running on whatever batch the server chose, when the two arms
  have to be read on one load.
"""

from __future__ import annotations

import analysis_eval
import pytest

from aoi_agent import i18n
from aoi_agent.analysis import prompts
from aoi_agent.graph import flow

_DOMAINS = {"line_id": ["L1"], "machine_id": ["M11"], "shift": ["A"],
            "defect_type": ["open"], "event_kinds": ["parameter_change"], "max_days": 9}


def test_the_default_arm_is_the_current_sentence(monkeypatch):
    monkeypatch.delenv(i18n.LANGUAGE_NOTE_ENV, raising=False)
    assert i18n.language_note_arm() == "current"
    assert i18n.language_note("zh-TW") == i18n.LANGUAGE_NOTE["zh-TW"]
    assert i18n.language_note("en") == i18n.LANGUAGE_NOTE["en"]


def test_the_2026_08_30_arm_is_the_sentence_before_the_class_list(monkeypatch):
    monkeypatch.setenv(i18n.LANGUAGE_NOTE_ENV, "2026-08-30")
    note = i18n.language_note("zh-TW")
    assert note == i18n.LANGUAGE_NOTE_2026_08_30["zh-TW"]
    assert i18n.LANGUAGE_NOTE["zh-TW"].startswith(note)      # a prefix: the list was appended
    assert "開路" not in note and "`open`" not in note


def test_an_unknown_arm_is_refused_not_read_as_the_default(monkeypatch):
    monkeypatch.setenv(i18n.LANGUAGE_NOTE_ENV, "2026-8-30")
    with pytest.raises(ValueError, match="names no language-note arm"):
        i18n.language_note("zh-TW")


def test_the_planner_prompt_follows_the_switch(monkeypatch):
    monkeypatch.setenv(i18n.LANGUAGE_NOTE_ENV, "2026-08-30")
    old = prompts.build_planning_messages("最近 short 有比較多嗎?", _DOMAINS)[0]["content"]
    monkeypatch.setenv(i18n.LANGUAGE_NOTE_ENV, "current")
    new = prompts.build_planning_messages("最近 short 有比較多嗎?", _DOMAINS)[0]["content"]
    assert i18n.LANGUAGE_NOTE_2026_08_30["zh-TW"] in old and "not 開路" not in old
    assert i18n.LANGUAGE_NOTE["zh-TW"] in new
    assert old != new


def test_the_reason_prompt_reads_the_same_switch(monkeypatch):
    monkeypatch.setenv(i18n.LANGUAGE_NOTE_ENV, "2026-08-30")
    monkeypatch.setenv(i18n.RATIONALE_GLOSSARY_ENV, "0")
    assert flow.reason_system_prompt("zh-TW") == (
        f"{flow.SYSTEM_PROMPT}\n\n{i18n.LANGUAGE_NOTE_2026_08_30['zh-TW']}")


def test_the_eval_pins_the_batch_by_default_and_frees_it_only_on_zero():
    assert analysis_eval.DEFAULT_NUM_BATCH == 512
    pinned = analysis_eval.eval_client("gpt-oss:20b", analysis_eval.DEFAULT_NUM_BATCH)
    assert pinned.load_options == {"num_batch": 512}
    free = analysis_eval.eval_client("gpt-oss:20b", 0)
    assert free.load_options == {}
