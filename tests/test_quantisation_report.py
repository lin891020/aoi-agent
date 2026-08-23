"""What the quantisation report says, and in what order it says it.

The measurement loop needs a checkpoint, an ONNX runtime and a quiet machine.
The argument the report makes needs none of those, and it is the part that can
go wrong silently: a quantisation report that leads with milliseconds has
already answered the wrong question, and would leave this project with two
headline metrics that disagree with each other.

So what is guarded here is the shape of the argument. The escape-budget table
comes before the latency table. A conversion that loses review reduction is
called a bad trade in words, not left as two numbers a reader might average.
And the deployment paragraph is allowed to conclude "not worth it", because the
alternative -- a script that can only report a win -- is a script whose output
carries no information.
"""

from __future__ import annotations

from quantisation_report import (
    REVIEW_REDUCTION_TOLERANCE,
    conclusion,
    latency_note,
    line_rate_implication,
    operating_row,
    render,
    review_reduction_delta,
    trade_verdict,
)
from reverifier_latency import summarise, throttle_verdict


class FakePoint:
    """The fields `operating_row` reads off an `OperatingPoint`."""

    def __init__(self, threshold, escape_rate, review_reduction, escapes):
        self.threshold = threshold
        self.escape_rate = escape_rate
        self.review_reduction = review_reduction
        self.escapes = escapes
        self.defects_total = 2997


def row(review_reduction: float, budget: float = 0.005) -> dict:
    return operating_row(FakePoint(0.915, 0.0047, review_reduction, 14), budget)


UNREACHABLE = {"budget": 0.005, "reachable": False}


# --------------------------------------------------------------------------
# the curve, which is what decides
# --------------------------------------------------------------------------


def test_operating_row_keeps_the_escape_count_behind_the_rate():
    # A rate without its numerator is unreadable at ≤0.10%, where the whole
    # column turns on two candidates.
    flattened = row(0.562)
    assert flattened["escapes"] == 14
    assert flattened["defects_total"] == 2997


def test_an_unreachable_budget_is_not_a_zero():
    assert operating_row(None, 0.001) == {"budget": 0.001, "reachable": False}
    assert review_reduction_delta(row(0.562), UNREACHABLE) is None
    assert review_reduction_delta(UNREACHABLE, row(0.562)) is None


def test_review_reduction_delta_is_signed_the_way_a_reader_expects():
    assert review_reduction_delta(row(0.562), row(0.549)) < 0
    assert review_reduction_delta(row(0.562), row(0.570)) > 0


def test_a_conversion_that_loses_the_curve_is_called_a_bad_trade():
    verdict = trade_verdict(
        row(0.562), row(0.549), "INT8 dynamic",
        baseline_p50=2.50, candidate_p50=1.20,
        baseline_mb=42.7, candidate_mb=10.7,
    )
    assert "not worth taking" in verdict
    # And it is refused *while* being faster and smaller, which is the whole
    # point: the speed-up is not allowed to buy its way past the curve.
    assert "1.3 points" in verdict
    assert "4.0x smaller" in verdict


def test_a_conversion_that_holds_the_curve_is_allowed_through():
    verdict = trade_verdict(
        row(0.562), row(0.560), "INT8 static",
        baseline_p50=2.50, candidate_p50=1.20,
        baseline_mb=42.7, candidate_mb=10.8,
    )
    assert "the curve holds" in verdict
    assert "not worth taking" not in verdict


def test_an_identical_curve_is_reported_as_unchanged_not_as_zero_point_zero():
    verdict = trade_verdict(
        row(0.562), row(0.562), "FP32 ONNX",
        baseline_p50=2.50, candidate_p50=2.40,
        baseline_mb=42.7, candidate_mb=42.6,
    )
    assert "unchanged" in verdict


def test_a_conversion_with_no_operating_point_is_refused_outright():
    verdict = trade_verdict(
        row(0.562), UNREACHABLE, "INT8 dynamic",
        baseline_p50=2.50, candidate_p50=1.20,
        baseline_mb=42.7, candidate_mb=10.7,
    )
    assert "refused" in verdict
    assert "no operating point" in verdict


def test_the_verdict_leads_with_the_curve_not_the_clock():
    verdict = trade_verdict(
        row(0.562), row(0.549), "INT8 dynamic",
        baseline_p50=2.50, candidate_p50=1.20,
        baseline_mb=42.7, candidate_mb=10.7,
    )
    assert verdict.index("review reduction") < verdict.index("smaller on disk")


def test_the_tolerance_is_the_only_thing_separating_the_two_verdicts():
    # Pinned so nobody widens the tolerance to let a conversion through. One
    # point of review reduction is roughly eighty regions per eight-thousand
    # candidate shift back in front of an operator; that is the price this
    # project is willing to pay for a smaller file, and no more.
    assert REVIEW_REDUCTION_TOLERANCE == 0.01
    inside = trade_verdict(
        row(0.562), row(0.562 - REVIEW_REDUCTION_TOLERANCE / 2), "engine",
        2.5, 1.2, 42.7, 10.7,
    )
    outside = trade_verdict(
        row(0.562), row(0.562 - REVIEW_REDUCTION_TOLERANCE * 2), "engine",
        2.5, 1.2, 42.7, 10.7,
    )
    assert "the curve holds" in inside
    assert "not worth taking" in outside


# --------------------------------------------------------------------------
# the deployment question
# --------------------------------------------------------------------------


def test_the_line_rate_paragraph_answers_in_boards_not_milliseconds():
    text = line_rate_implication(16.3, 499, 2.50, 1.20)
    assert "No" in text
    assert "499 boards" in text
    # 16.3 candidates x 2.5ms is 41ms of a 10-second cycle.
    assert "41ms" in text
    assert "0.41%" in text


def test_the_line_rate_paragraph_states_its_assumption_rather_than_hiding_it():
    text = line_rate_implication(16.3, 499, 2.50, 1.20, seconds_per_board=4.0)
    assert "every 4 seconds" in text


def test_the_conclusion_can_say_nothing_changed():
    text = conclusion(None, 0.0, 42.7)
    assert "nothing, and that is the result" in text
    assert "keeps the float32 checkpoint" in text


def test_the_conclusion_names_the_survivor_when_there_is_one():
    text = conclusion("INT8 static", 31.9, 42.7)
    assert "INT8 static" in text
    assert "10.8MB" in text
    # Even a survivor does not silently move the deployed threshold.
    assert "DEFAULT_DISMISS_THRESHOLD" in text


def test_the_latency_note_says_slower_when_it_is_slower():
    engines = [
        {"key": "fp32_torch", "label": "FP32 torch", "single": summarise([2.5] * 5)},
        {"key": "int8", "label": "INT8", "single": summarise([5.0] * 5)},
    ]
    text = latency_note(engines, "fp32_torch")
    assert "2.00x slower" in text
    assert "faster" not in text.split("INT8")[1].split(";")[0]


def test_the_latency_note_refuses_to_be_read_as_a_reason_to_ship():
    engines = [
        {"key": "fp32_torch", "label": "FP32 torch", "single": summarise([2.5] * 5)},
        {"key": "int8", "label": "INT8", "single": summarise([1.25] * 5)},
    ]
    text = latency_note(engines, "fp32_torch")
    assert "2.00x faster" in text
    assert "rather than as a reason to ship it" in text


# --------------------------------------------------------------------------
# the rendered section
# --------------------------------------------------------------------------


def engine(key: str, label: str, p50: float, size_mb: float) -> dict:
    return {
        "key": key,
        "label": label,
        "size_mb": size_mb,
        "load_ms": 90.0,
        "cold_ms": p50 * 4,
        "single": summarise([p50, p50 * 1.1, p50 * 1.2, p50 * 2]),
        "batches": {
            1: {"p50": p50, "per_candidate_ms": p50, "throughput": 1000 / p50},
            8: {"p50": p50 * 4, "per_candidate_ms": p50 / 2,
                "throughput": 2000 / p50},
        },
        "soak_early": summarise([p50] * 10),
        "soak_late": summarise([p50 * 1.05] * 10),
        "throttle": throttle_verdict([p50] * 10, [p50 * 1.05] * 10),
        "rss_mb": 900.0,
    }


ENGINES = [
    engine("fp32_torch", "FP32 torch", 2.50, 42.7),
    engine("fp32_onnx", "FP32 ONNX", 2.10, 42.6),
    engine("int8_dynamic", "INT8 dynamic", 1.20, 10.7),
    engine("int8_static", "INT8 static", 1.30, 10.8),
]


def results_fixture() -> dict:
    reductions = {
        "fp32_torch": 0.562, "fp32_onnx": 0.562,
        "int8_dynamic": 0.549, "int8_static": 0.560,
    }
    budgets = [0.001, 0.005]
    return {
        "baseline_key": "fp32_torch",
        "engines": ENGINES,
        "budgets": budgets,
        "batch_sizes": [1, 8],
        "operating_points": {
            key: {str(budget): row(value, budget) for budget in budgets}
            for key, value in reductions.items()
        },
        "agreement": {
            key: {"n": 8143, "class_agreement": 0.995, "disagreements": 40,
                  "mean_probability_delta": 1.7e-3, "max_probability_delta": 3.9e-1}
            for key in ("fp32_onnx", "int8_dynamic", "int8_static")
        },
        "parameters": 11_181_642,
        "patch_count": 8143,
        "defects_total": 2997,
        "cpu_threads": 4,
        "calibration_samples": 512,
        "calibration_seed": 20260823,
        "pipeline_batch": 8,
        "pipeline_batch_bucket": 8,
        "soak_s": 150.0,
        "thermal_split_s": 60.0,
        "ps_before": "NAME  ID  SIZE",
        "ps_after": "NAME  ID  SIZE",
        "busy_before": [],
        "busy_after": [],
        "verdicts": [
            trade_verdict(row(0.562), row(reductions[key]), key, 2.5, 1.2, 42.7, 10.7)
            for key in ("fp32_onnx", "int8_dynamic", "int8_static")
        ],
        "latency_note": latency_note(ENGINES, "fp32_torch"),
        "line_rate": line_rate_implication(16.3, 499, 2.50, 1.20),
        "conclusion": conclusion("INT8 static", 31.9, 42.7),
    }


def test_render_produces_one_section_with_every_required_measurement():
    lines = render(results_fixture())
    text = "\n".join(lines)
    assert sum(1 for line in lines if line.startswith("### ")) == 1
    for required in (
        "#### Manual review removed at an escape budget — FP32 against INT8",
        "#### How far each engine drifted from the float model",
        "#### Single candidate, warm — CPU",
        "#### Batched throughput — CPU",
        "#### Cold against warm",
        "#### Footprint",
    ):
        assert required in text, required


def test_render_leads_with_the_operating_point_not_the_latency():
    # The invariant this whole file exists for. If a future edit moves the
    # latency table up, this fails.
    lines = render(results_fixture())
    curve = next(i for i, line in enumerate(lines)
                 if line.startswith("#### Manual review removed"))
    latency = next(i for i, line in enumerate(lines)
                   if line.startswith("#### Single candidate"))
    assert curve < latency


def test_render_names_every_engine_in_the_curve_table():
    text = "\n".join(render(results_fixture()))
    for label in ("FP32 torch", "FP32 ONNX", "INT8 dynamic", "INT8 static"):
        assert label in text


def test_render_says_where_the_calibration_set_came_from():
    text = "\n".join(render(results_fixture()))
    assert "trainval" in text
    assert "never test" in text
    assert "512 patches" in text


def test_render_warns_off_the_tightest_budget_row():
    text = "\n".join(render(results_fixture()))
    assert "decided by two escapes" in text


def test_render_records_both_contention_checks():
    results = results_fixture()
    results["busy_before"] = ["95096 201% ffmpeg -y -i blk_zh2zh_0030_v.mp4"]
    text = "\n".join(render(results))
    assert "busy processes before the run" in text
    assert "ffmpeg" in text


def test_render_tables_have_a_separator_matching_their_columns():
    lines = render(results_fixture())
    for index, line in enumerate(lines):
        if not line.startswith("|") or index + 1 >= len(lines):
            continue
        following = lines[index + 1]
        if not following.startswith("|---"):
            continue
        if set(following.replace("|", "").replace("-", "").strip()) - {""}:
            continue
        assert line.count("|") == following.count("|"), line


def test_render_states_that_peak_rss_is_a_floor_not_an_isolated_figure():
    text = "\n".join(render(results_fixture()))
    assert "high-water mark" in text
    assert "not an isolated measurement" in text
