"""The re-verifier latency report's arithmetic and its prose.

No checkpoint is loaded and no device is touched. The measurement half of that
script needs a GPU and ninety seconds of sustained load; the half that decides
what the numbers *mean* -- which percentile, which thermal phase, whether the
README's claim survived -- is pure, and it is the half that can be wrong
quietly. A p90 computed off by one still prints.
"""

from __future__ import annotations

from reverifier_latency import (
    claim_verdict,
    crossover_note,
    competing_processes,
    edge_implication,
    format_ms,
    magnitude_phrase,
    nearest_bucket,
    percentile,
    render,
    resident_models,
    summarise,
    thermal_split,
    throttle_verdict,
)


def test_percentile_is_nearest_rank_so_every_figure_really_happened():
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == 50
    assert percentile(values, 90) == 90
    assert percentile(values, 99) == 99
    # No interpolation: the answer is always a value from the sample.
    assert percentile([1.0, 2.0, 3.0], 90) in {1.0, 2.0, 3.0}


def test_percentile_survives_the_short_sample():
    assert percentile([7.0], 99) == 7.0
    assert percentile([1.0, 9.0], 100) == 9.0


def test_percentile_refuses_an_empty_sample():
    try:
        percentile([], 50)
    except ValueError:
        return
    raise AssertionError("an empty sample must raise, not return a made-up number")


def test_summarise_reports_the_distribution_not_just_a_mean():
    stats = summarise([1.0, 2.0, 3.0, 100.0])
    assert stats["n"] == 4
    assert stats["min"] == 1.0
    assert stats["max"] == 100.0
    assert stats["p50"] == 2.0
    # The mean is dragged four-fold by one outlier the median never sees. That
    # gap is the reason the headline is a percentile.
    assert stats["mean"] > stats["p50"] * 10


def test_format_ms_keeps_sub_millisecond_precision():
    # 0.62ms rounded to "1ms" is wrong by half, which is the whole finding here.
    assert format_ms(0.62) == "0.62ms"
    assert format_ms(4.317) == "4.32ms"
    assert format_ms(43.17) == "43.2ms"
    assert format_ms(431.7) == "432ms"


def test_magnitude_phrase_buckets_by_order_of_magnitude():
    assert magnitude_phrase(0.5) == "under a millisecond"
    assert magnitude_phrase(4.0) == "single-digit milliseconds"
    assert magnitude_phrase(40.0) == "tens of milliseconds"
    assert magnitude_phrase(400.0) == "hundreds of milliseconds"
    assert magnitude_phrase(4000.0) == "seconds"


def test_claim_verdict_says_which_direction_the_readme_was_wrong():
    faster = claim_verdict(4.0)
    assert "wrong" in faster
    assert "faster than the README said" in faster
    assert "4.00ms" in faster

    slower = claim_verdict(400.0)
    assert "slower than the README said" in slower

    held = claim_verdict(40.0)
    assert "holds" in held
    assert "wrong" not in held


def test_thermal_split_cuts_at_sixty_seconds():
    samples = [(0.0, 1.0), (59.9, 2.0), (60.0, 3.0), (120.0, 4.0)]
    early, late = thermal_split(samples)
    assert early == [1.0, 2.0]
    assert late == [3.0, 4.0]


def test_thermal_split_takes_an_explicit_boundary():
    early, late = thermal_split([(0.0, 1.0), (10.0, 2.0)], split_s=5.0)
    assert early == [1.0]
    assert late == [2.0]


def test_throttle_verdict_calls_a_real_slowdown():
    verdict = throttle_verdict([10.0] * 20, [14.0] * 20)
    assert "Throttled" in verdict
    assert "+40%" in verdict
    assert "steady-state" in verdict


def test_throttle_verdict_calls_a_flat_run_flat():
    verdict = throttle_verdict([10.0] * 20, [10.4] * 20)
    assert "No throttle observed" in verdict


def test_throttle_verdict_flags_a_run_that_got_faster_as_suspect():
    # Speeding up under sustained load is not thermal. Something else on the
    # machine let go, which makes the whole run untrustworthy.
    verdict = throttle_verdict([20.0] * 20, [10.0] * 20)
    assert "suspect" in verdict


def test_throttle_verdict_admits_when_the_soak_was_too_short():
    verdict = throttle_verdict([10.0] * 20, [])
    assert "unanswered" in verdict


def test_resident_models_reads_ollama_ps():
    output = (
        "NAME           ID              SIZE     PROCESSOR    CONTEXT    UNTIL\n"
        "gpt-oss:20b    17052f91a42e    12 GB    100% GPU     32768      27 minutes from now"
    )
    assert resident_models(output) == ["gpt-oss:20b"]


def test_resident_models_is_empty_on_an_idle_machine():
    assert resident_models("") == []
    assert resident_models("NAME    ID    SIZE    PROCESSOR    UNTIL") == []


def test_resident_models_does_not_invent_a_model_from_an_error_string():
    # `ollama_ps` returns a parenthesised message when the binary is missing.
    # Reading that as a resident model would block every run on this machine.
    assert resident_models("(ollama ps unavailable: [Errno 2] no ollama)") == []


def test_nearest_bucket_picks_the_closest_swept_batch():
    assert nearest_bucket(16, [1, 2, 4, 8, 16, 32]) == 16
    assert nearest_bucket(18, [1, 2, 4, 8, 16, 32]) == 16
    assert nearest_bucket(25, [1, 2, 4, 8, 16, 32]) == 32


def test_edge_implication_leads_with_the_cpu_number():
    text = edge_implication(cpu_p50=6.0, cpu_throughput=900.0, checkpoint_mb=42.7)
    assert "CPU figure is the one to size on" in text
    assert "6.00ms" in text
    assert "900" in text
    assert "43MB" in text


def results_fixture() -> dict:
    def device(label, p50):
        return {
            "name": label.lower(),
            "label": label,
            "load_ms": 300.0,
            "cold_ms": p50 * 20,
            "single": summarise([p50, p50 * 1.1, p50 * 1.2, p50 * 3]),
            "batches": {
                1: {"p50": p50, "per_candidate_ms": p50, "throughput": 1000 / p50},
                16: {"p50": p50 * 4, "per_candidate_ms": p50 / 4,
                     "throughput": 4000 / p50},
            },
            "soak_early": summarise([p50] * 10),
            "soak_late": summarise([p50 * 1.05] * 10),
            "throttle": throttle_verdict([p50] * 10, [p50 * 1.05] * 10),
        }

    return {
        "checkpoint_mb": 42.7,
        "parameters": 11_181_642,
        "patch_count": 8143,
        "cpu_threads": 10,
        "ps_before": "NAME  ID  SIZE",
        "ps_after": "NAME  ID  SIZE",
        "busy_before": [],
        "busy_after": [],
        "devices": [device("MPS", 2.0), device("CPU", 6.0)],
        "batch_sizes": [1, 16],
        "pipeline_batch": 16,
        "pipeline_batch_mean": 16.3,
        "pipeline_batch_max": 61,
        "pipeline_batch_bucket": 16,
        "soak_s": 90.0,
        "peak_rss_mb": 812.0,
        "thread_sweep": {1: {8: 2.04, 16: 8.10}, 2: {8: 1.29, 16: 8.08},
                         4: {8: 1.00, 16: 8.05}, 8: {8: 1.02, 16: 8.10}},
        "patch_build_p50": 0.05,
        "verdict": claim_verdict(6.0),
        "crossover": "crossover placeholder",
        "implication": edge_implication(6.0, 666.0, 42.7),
    }


def test_render_produces_one_section_with_every_required_measurement():
    lines = render(results_fixture())
    text = "\n".join(lines)

    # Newest-last append: one `###` section, subsections beneath it.
    assert sum(1 for line in lines if line.startswith("### ")) == 1
    for required in (
        "#### Single candidate, warm",
        "#### Cold against warm",
        "#### Batched throughput",
        "#### Thermal — first 60s against steady state",
        "#### Footprint",
    ):
        assert required in text, required

    # Both devices, named, so nobody has to guess which column is the edge one.
    assert "| MPS |" in text
    assert "| CPU |" in text
    # The contention evidence travels with the number.
    assert "ollama ps before the run" in text
    assert "ollama ps after the run" in text


def test_render_marks_the_pipelines_real_batch_in_the_sweep():
    lines = render(results_fixture())
    marked = [line for line in lines if line.startswith("| 16 ←")]
    assert len(marked) == 1


def test_render_tables_have_a_separator_matching_their_columns():
    lines = render(results_fixture())
    for index, line in enumerate(lines):
        if not line.startswith("|") or index + 1 >= len(lines):
            continue
        following = lines[index + 1]
        if set(following.replace("|", "").replace("-", "").strip()) - {""}:
            continue
        if not following.startswith("|---"):
            continue
        assert line.count("|") == following.count("|"), line


def test_render_carries_the_verdict_on_the_old_claim():
    text = "\n".join(render(results_fixture()))
    assert "tens of milliseconds" in text
    assert "wrong" in text


PS_TABLE = """  PID  %CPU COMMAND
17103   0.6 /opt/homebrew/opt/ollama/bin/ollama serve
60535   2.1 /opt/homebrew/Cellar/ollama/.../llama-server --model /Users/x/blobs/sha256-e7b
93044  26.1 /Users/lin1020/Projects/aoi-agent/.venv/bin/python3 bench_tv.py --arch fcos
94100 170.6 /System/Library/PrivateFrameworks/MediaAnalysis.framework/.../mediaanalysisd
29197   0.0 /Users/lin1020/.vscode/extensions/.../python-env-tools/bin/pet server
  713   5.7 /System/Library/CoreServices/Finder.app/Contents/MacOS/Finder
"""


def test_competing_processes_catches_the_torch_job_ollama_ps_cannot_see():
    # The whole point of the second check. `ollama ps` reports only Ollama's own
    # models, so this FCOS benchmark saturating MPS is invisible to it.
    found = competing_processes(PS_TABLE, own_pid=1)
    assert any("bench_tv.py" in line for line in found)


def test_competing_processes_catches_macos_own_gpu_user():
    found = competing_processes(PS_TABLE, own_pid=1)
    assert any("mediaanalysisd" in line for line in found)


def test_competing_processes_ignores_idle_helpers_and_unrelated_apps():
    found = competing_processes(PS_TABLE, own_pid=1)
    # `pet server` matches no claimant marker and sits at 0% anyway.
    assert not any("pet server" in line for line in found)
    # Finder is busy but is not a GPU claimant by any of the markers.
    assert not any("Finder" in line for line in found)
    # `ollama serve` is the supervisor, not the runner; it holds no weights.
    assert not any("ollama serve" in line for line in found)


def test_competing_processes_does_not_flag_the_benchmark_itself():
    table = PS_TABLE + "99999  80.0 /usr/bin/python3 scripts/reverifier_latency.py\n"
    assert not any("reverifier_latency" in line
                   for line in competing_processes(table, own_pid=99999))


def test_competing_processes_is_quiet_on_a_quiet_machine():
    assert competing_processes("  PID  %CPU COMMAND\n", own_pid=1) == []


def test_render_records_both_contention_checks():
    results = results_fixture()
    results["busy_before"] = ["93044 26% bench_tv.py --arch fcos"]
    text = "\n".join(render(results))
    assert "busy processes before the run" in text
    assert "bench_tv.py" in text
    # And the reader is told why one check is not enough.
    assert "reports Ollama's own resident models and nothing else" in text


def test_render_shows_the_cpu_cliff_against_thread_count():
    text = "\n".join(render(results_fixture()))
    assert "#### Does the CPU cliff move with the core count?" in text
    # Every thread count measured gets a row.
    for threads in (1, 2, 4, 8):
        assert f"| {threads} | " in text
    assert "batch at 8 on any CPU box" in text


def devices_for_crossover(mps_single, cpu_single, mps_batch16, cpu_batch16):
    def device(name, single, b16):
        return {
            "name": name,
            "single": {"p50": single},
            "batches": {
                1: {"per_candidate_ms": single},
                16: {"per_candidate_ms": b16},
            },
        }

    return [device("mps", mps_single, mps_batch16),
            device("cpu", cpu_single, cpu_batch16)]


def test_crossover_note_says_the_gpu_loses_at_one_candidate():
    # The measured shape: MPS ~3x slower single-shot, far faster batched.
    note = crossover_note(devices_for_crossover(7.34, 2.50, 0.21, 4.18))
    assert "At one candidate the GPU is the slower device" in note
    assert "2.9x slower" in note
    assert "overtakes at batch 16" in note


def test_crossover_note_reports_a_gpu_that_never_overtakes():
    note = crossover_note(devices_for_crossover(7.34, 2.50, 9.0, 4.18))
    assert "CPU stayed ahead" in note


def test_crossover_note_reports_a_gpu_that_wins_outright():
    note = crossover_note(devices_for_crossover(1.0, 2.50, 0.21, 4.18))
    assert "faster than CPU even at one candidate" in note


def test_crossover_note_is_silent_without_both_devices():
    only_cpu = [{"name": "cpu", "single": {"p50": 2.5}, "batches": {}}]
    assert crossover_note(only_cpu) == ""


def test_render_carries_the_crossover_finding():
    results = results_fixture()
    results["crossover"] = crossover_note(
        devices_for_crossover(7.34, 2.50, 0.21, 4.18))
    text = "\n".join(render(results))
    assert "At one candidate the GPU is the slower device" in text
