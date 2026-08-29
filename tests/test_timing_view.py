"""The stage table under an answer: wait beside inference, nothing as zero."""

from __future__ import annotations

from aoi_agent.station.timing_view import rows


def test_each_stage_that_ran_gets_a_row_and_the_wall_column_is_totalled():
    table = rows({
        "plan": 16200.0, "plan_eval": 14900.0, "plan_prompt_eval": 800.0, "plan_load": 170.0,
        "tools_wall": 210.0, "tools_longest_branch": 180.0, "tools_sequential": 400.0,
        "chart": 0.4,
        "synthesise": 18400.0, "synthesise_eval": 17100.0,
    })
    by_key = {row["key"]: row for row in table}

    assert [row["key"] for row in table] == [
        "analysis.timing.plan", "analysis.timing.tools", "analysis.timing.chart",
        "analysis.timing.synthesise", "analysis.timing.total",
    ]
    assert by_key["analysis.timing.plan"]["wall_s"] == 16.2
    assert by_key["analysis.timing.plan"]["model_s"] == 14.9
    assert by_key["analysis.timing.tools"]["model_s"] is None
    assert by_key["analysis.timing.tools"]["model_unrecorded"] is False, "nothing infers there"
    assert by_key["analysis.timing.total"]["wall_s"] == round(16.2 + 0.21 + 0.0 + 18.4, 2)


def test_a_run_stored_before_the_model_column_existed_says_unrecorded_not_zero():
    table = rows({"plan": 9000.0, "tools_wall": 100.0, "synthesise": 8000.0})
    plan = next(row for row in table if row["key"] == "analysis.timing.plan")
    assert plan["model_s"] is None
    assert plan["model_unrecorded"] is True


def test_a_refusal_has_a_planning_row_and_nothing_else_before_the_total():
    table = rows({"plan": 7000.0, "plan_eval": 6500.0})
    assert [row["key"] for row in table] == ["analysis.timing.plan", "analysis.timing.total"]


def test_no_timings_is_no_table():
    assert rows({}) == []
    assert rows(None) == []
