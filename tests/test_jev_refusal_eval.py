"""The Jev refusal eval scores what it says it scores, and shows Jev only what it should.

What would spoil the reading without raising: the planner's system prompt
leaking into the state (then the score is of the prompt, not the question),
the blind arm still carrying the question, a wording drift in the fixed
questions, a gold scheme that removes the wrong five, a metric that counts
the answer class as the positive one, an ECE read against zero, or a verdict
that reads the full arm without the blind one.
"""

from __future__ import annotations

import json

import jev_refusal_eval as ev
import pytest

from aoi_agent.analysis.prompts import SYSTEM_PROMPT

TOOLS = ["query_defect_history — Count defects.", "query_machine_stats — Rank machines.",
         "run_sql — One SELECT."]
FACTS = "The store holds 9 days of inspections, 2026-08-01 to 2026-08-09."


# ---- the state ------------------------------------------------------------------


def test_the_state_carries_the_question_the_tools_and_the_facts_and_nothing_of_the_prompt():
    state = ev.build_state("M22 這兩天 open 多不多？", "full", TOOLS, FACTS)
    assert "M22 這兩天 open 多不多？" in state and "query_machine_stats" in state and "9 days" in state
    for sentence in SYSTEM_PROMPT.splitlines():
        if len(sentence.strip()) > 30:
            assert sentence.strip() not in state


def test_the_blind_arm_has_no_question_and_tools_only_has_no_facts():
    blind = ev.build_state("M22 這兩天 open 多不多？", "blind", TOOLS, FACTS)
    assert "M22" not in blind and ev.BLIND_QUESTION in blind and "9 days" in blind
    tools_only = ev.build_state("M22 這兩天 open 多不多？", "tools_only", TOOLS, FACTS)
    assert "M22" in tools_only and "9 days" not in tools_only
    with pytest.raises(ValueError):
        ev.build_state("x", "half", TOOLS, FACTS)


def test_the_fixed_wording_is_the_fixed_wording():
    """A changed word is a new experiment; this pins the one that was run."""
    assert ev.ANSWERABLE["instructions"] == (
        "Can this question be answered by calling one or more of the listed tools, "
        "with arguments the question itself supplies?")
    assert set(ev.ANSWERABLE["criteria"]) == {"true", "false"}
    q = ev.questions_for(["query_defect_history", "run_sql"])
    assert list(q) == ["answerable", "needs__query_defect_history", "needs__run_sql"]
    assert q["needs__run_sql"]["instructions"] == "Does answering this question need a call to `run_sql`?"


def test_the_facts_come_from_the_domains_and_name_the_absences():
    text = ev.facts({"max_days": 9, "date_span": ("2026-08-01", "2026-08-09"),
                     "line_id": {"L2", "L1"}, "machine_id": {"M22"}, "defect_type": {"open"},
                     "relative_to": {"maintenance"}})
    assert "9 days" in text and "L1, L2" in text and "maintenance" in text
    assert "who reviewed a board" in text


# ---- gold -------------------------------------------------------------------------


def _rows():
    rows = [{"id": f"A{i:02d}", "question": f"q{i}", "expect_refusal": False,
             "expect_tools": ["query_defect_history"]} for i in range(1, 5)]
    rows += [{"id": q, "question": q, "expect_refusal": True} for q in ("S02", "S06", "S11", "S12", "A04", "S20")]
    rows[0]["expect_any_of"] = [{"expect_tools": ["query_machine_stats"], "why": "ranking"}]
    return rows


def test_the_adjudicated_gold_moves_exactly_the_five_gains():
    rows = _rows()
    fixture, adjudicated = ev.gold(rows, "fixture"), ev.gold(rows, "adjudicated")
    assert sum(fixture.values()) == 6 and sum(adjudicated.values()) == 1
    assert {q for q in fixture if fixture[q] != adjudicated[q]} == ev.ADJUDICATED_GAINS
    with pytest.raises(ValueError):
        ev.gold(rows, "lenient")


def test_accepted_tool_sets_include_the_alternatives():
    assert ev.expected_tool_sets(_rows()[0]) == [{"query_defect_history"}, {"query_machine_stats"}]
    assert ev.expected_tool_sets(_rows()[1]) == [{"query_defect_history"}]


# ---- metrics -----------------------------------------------------------------------


def test_refusal_metrics_count_refuse_as_the_positive_class():
    truth = {"a": True, "b": True, "c": False, "d": False}
    pred = {"a": True, "b": False, "c": True, "d": False}
    m = ev.refusal_metrics(pred, truth)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5


def test_ece_is_zero_for_a_perfect_model_and_large_for_a_confident_wrong_one():
    assert ev.ece({"a": 1.0, "b": 0.0}, {"a": True, "b": False}) == 0.0
    assert ev.ece({"a": 0.95, "b": 0.95}, {"a": False, "b": False}) == pytest.approx(0.95)


def test_the_noise_floor_is_above_zero_for_uncertain_probabilities_and_zero_for_certain_ones():
    certain = {f"q{i}": float(i % 2) for i in range(70)}
    assert ev.noise_floor_ece(certain, 50) == 0.0
    uncertain = {f"q{i}": 0.5 for i in range(70)}
    assert ev.noise_floor_ece(uncertain, 50) > 0.02


def test_bootstrap_ci_brackets_the_point_estimate():
    ids = [f"q{i}" for i in range(40)]
    lo, hi = ev.bootstrap_ci(lambda s: sum(1 for q in s if int(q[1:]) < 20) / len(s), ids, 200)
    assert lo <= 0.5 <= hi and lo > 0.2 and hi < 0.8


def test_toolset_hit_is_recall_over_accepted_sets_and_precision_is_over_the_prediction():
    rows = _rows()
    predicted = {"A01": {"query_machine_stats", "run_sql"}, "A02": {"run_sql"}, "A03": {"query_defect_history"}}
    m = ev.toolset_metrics(predicted, rows, ["A01", "A02", "A03"])
    assert m["hits"] == 2 and m["hit"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx((0.5 + 0.0 + 1.0) / 3)


# ---- reading gpt-oss --------------------------------------------------------------


def test_gptoss_runs_are_read_from_raw_files_and_from_logs(tmp_path):
    raw = tmp_path / "language_note_2_current.json"
    raw.write_text(json.dumps([{"id": "A01", "planned": "query_defect_history({\"days\": 7}) + run_sql({\"sql\": \"x\"})"},
                               {"id": "S20", "planned": "(refused)"}]))
    r = ev.gptoss_from_raw(raw)
    assert r["A01"] == {"refused": False, "tools": {"query_defect_history", "run_sql"}}
    assert r["S20"]["refused"] is True
    log = tmp_path / "planner_drift_2026-09-22_run4.log"
    log.write_text("noise\n  [  1/70] ok   stable  q\n        planned query_machine_stats(top_n=5)\n"
                   "  [  2/70] MISS stable  q\n        planned (refused)\n        refused a question\n")
    r = ev.gptoss_from_log(log, ["A01", "S20"])
    assert r == {"A01": {"refused": False, "tools": {"query_machine_stats"}},
                 "S20": {"refused": True, "tools": set()}}


# ---- the run and the verdict -------------------------------------------------------


class _StubClient:
    """Answers P(answerable) from a table and every tool 0.9 for the first tool only."""

    door = "gateway"

    def __init__(self, table):
        self.table, self.calls = table, []

    def ask(self, state, questions):
        self.calls.append(state)
        from aoi_agent.llm.jev import JevAnswer
        key = next((q for q in self.table if q in state), "blind")
        answers = {"answerable": {"noul": self.table.get(key, 0.4)}}
        for name in questions:
            if name.startswith("needs__"):
                answers[name] = {"noul": 0.9 if name.endswith("query_defect_history") else 0.1}
        return JevAnswer(answers=answers, model="jev-1.13.0", input_tokens=300, output_tokens=20, latency_s=0.2)


def test_run_asks_every_arm_repeat_and_question_once_and_resumes():
    rows = _rows()
    client = _StubClient({r["question"]: 0.9 for r in rows})
    first = ev.run(client, rows, ("full",), 2, TOOLS, FACTS, log=lambda *_: None)
    assert len(first) == 2 * len(rows) and len(client.calls) == 2 * len(rows)
    again = ev.run(client, rows, ("full", "blind"), 2, TOOLS, FACTS, log=lambda *_: None, existing=first)
    assert len(again) == 4 * len(rows) and len(client.calls) == 4 * len(rows)
    assert {r["model"] for r in again} == {"jev-1.13.0"}


def _analysis(full_f1_table, blind_p, gpt_refused):
    rows = _rows()
    table = {r["question"]: full_f1_table(r) for r in rows}
    client = _StubClient(table)
    records = ev.run(client, rows, ("full",), 3, TOOLS, FACTS, log=lambda *_: None)
    for r in rows:
        for rep in range(3):
            records.append({"arm": "blind", "repeat": rep, "id": r["id"], "p_answerable": blind_p,
                            "p_tools": {}, "model": "jev-1.13.0", "latency_s": 0.2,
                            "input_tokens": 1, "output_tokens": 1, "asked_at": ""})
    gptoss = {"today": {r["id"]: {"refused": gpt_refused(r), "tools": {"query_defect_history"}} for r in rows}}
    return ev.analyse(records, rows, gptoss, n_boot=50)


def test_a_perfect_full_arm_over_a_blind_one_is_useful_as_a_gate():
    truth = ev.gold(_rows(), "adjudicated")
    a = _analysis(lambda r: 0.1 if truth[r["id"]] else 0.9, 0.6, lambda r: r["id"] == "S20" or r["id"] == "A01")
    assert a["arms"]["full"]["adjudicated"]["f1"] == 1.0
    assert ev.verdict(a)[0] == "useful_as_gate"


def test_a_full_arm_that_matches_the_blind_arm_is_not_reading():
    a = _analysis(lambda r: 0.6, 0.6, lambda r: r["id"] == "S20")
    assert ev.verdict(a)[0] == "not_reading"


def test_the_verdict_needs_both_arms_and_a_comparator():
    rows = _rows()
    records = ev.run(_StubClient({}), rows, ("full",), 1, TOOLS, FACTS, log=lambda *_: None)
    assert ev.verdict(ev.analyse(records, rows, {}, n_boot=10))[0] == "incomplete"


def test_render_names_the_model_version_both_golds_and_the_verdict():
    truth = ev.gold(_rows(), "adjudicated")
    a = _analysis(lambda r: 0.1 if truth[r["id"]] else 0.9, 0.6, lambda r: r["id"] == "S20")
    label, sentence = ev.verdict(a)
    record = {"date": "2026-09-25", "commit": "abc1234", "door": "gateway", "repeats": 3, "requests": 60,
              "analysis": a, "verdict": label, "verdict_sentence": sentence, "zh_en": {},
              "per_question": {"A01": {"refuse": False, "p": 0.9, "gptoss": "A", "tools": {"query_defect_history"}}}}
    body = ev.render(record)
    assert "jev-1.13.0" in body and "**fixture gold**" in body and "**adjudicated gold**" in body
    assert f"**Verdict: {label}.**" in body and "| Jev full |" in body and "| gpt-oss today |" in body
