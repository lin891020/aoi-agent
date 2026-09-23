"""The session probe compares plans, not prose, and its verdict was written first.

Four ways it could report something other than what it claims:

* it compares the whole response, so a reworded `interpretation` reads as a
  different plan -- which is exactly what the five-minute version did;
* the interposed workload is drawn from the fixture, so it warms the prefixes
  being measured;
* a partial run appends, and the entry's table has empty rows for conditions
  nobody measured;
* the verdict is read after the fact, on a threshold chosen to fit.

Nothing here calls a model: the plans are hand-written strings.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The shape `build_planning_messages` reads; the probe composes a real prompt
#: even in these tests, because a probe that cannot build one is broken.
DOMAINS = {"line_id": ["L1"], "machine_id": ["M11"], "defect_type": ["open"],
           "relative_to": [], "max_days": 9, "date_span": ["2026-08-01", "2026-08-09"]}


def _script(name: str):
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    pytest.importorskip("langgraph", reason="the probe imports the analysis graph's prompts")
    return _script("planner_session_probe")


# ---- what counts as a difference ------------------------------------------------


def test_only_the_calls_are_compared(probe):
    """Two plans that call the same tools with the same arguments are one plan."""
    baseline = {"S01": "query_defect_history({\"days\": 7})"}
    same = {"S01": "query_defect_history({\"days\": 7})"}
    other = {"S01": "query_defect_history({\"days\": 3})"}
    assert probe.differences(baseline, same) == []
    assert probe.differences(baseline, other) == ["S01"]


def test_a_question_missing_from_a_condition_counts_as_moved(probe):
    assert probe.differences({"S01": "a", "S02": "b"}, {"S01": "a"}) == ["S02"]


def test_a_failed_call_is_a_cell_and_not_a_crash(probe):
    class Refuses:
        def chat(self, *_a, **_k):
            raise TimeoutError("deadline")

    text = probe.plan_calls(Refuses(), "問題", DOMAINS)
    assert text.startswith("(no answer: TimeoutError")


def test_a_prompt_this_probe_cannot_build_raises_rather_than_reading_as_a_timeout(probe):
    class NeverCalled:
        def chat(self, *_a, **_k):                     # pragma: no cover - must not be reached
            raise AssertionError("the prompt should have raised first")

    with pytest.raises(KeyError):
        probe.plan_calls(NeverCalled(), "問題", {})


def test_an_unparseable_answer_is_its_own_cell(probe):
    class Garbles:
        def chat(self, *_a, **_k):
            return type("R", (), {"text": "not json"})()

    assert probe.plan_calls(Garbles(), "問題", DOMAINS) == "(unparsed)"


# ---- the interposition cannot drift ---------------------------------------------


def test_the_interposed_questions_are_fixed_and_not_the_fixtures(probe):
    fixture = {q["question"] for q in json.loads(probe.QUESTIONS.read_text())}
    assert len(probe.INTERPOSED) == 10 and probe.INTERPOSED_ROUNDS == 20
    assert not (set(probe.INTERPOSED) & fixture), (
        "interposing a measured question warms the prefix the measurement reads")


# ---- the verdict was written before the run -------------------------------------


@pytest.mark.parametrize(("moves", "label"), [
    ({"C2": 0, "C3": 5, "C4": 6}, "the_sequence"),
    ({"C2": 4, "C3": 7, "C4": 4}, "the_sequence"),    # the floor is C2, not zero
    ({"C2": 4, "C3": 5, "C4": 4}, "neither"),         # one past the floor is noise
    ({"C2": 0, "C3": 0, "C4": 4}, "the_reload"),
    ({"C2": 2, "C3": 3, "C4": 6}, "the_reload"),
    ({"C2": 0, "C3": 0, "C4": 0}, "neither"),
    ({"C2": 9, "C3": 9, "C4": 9}, "neither"),         # a noisy repeat is a wide floor
    ({"C2": 0, "C3": 4}, "incomplete"),
])
def test_the_verdict_is_the_pre_registered_ladder(probe, moves, label):
    assert probe.verdict(moves)[0] == label


def test_the_threshold_is_the_one_the_drift_set(probe):
    """Three questions past C2 is the bar, because three readings spanned 23-28 of 42."""
    assert probe.MOVED == 3
    benchmarks = (ROOT / "docs" / "benchmarks.md").read_text()
    assert "25 / 42 every time" in benchmarks


def test_only_the_full_four_condition_run_may_append(probe):
    assert probe.publishable(["C1", "C2", "C3", "C4"], 70)
    assert not probe.publishable(["C1", "C2"], 70)
    assert not probe.publishable(["C1", "C2", "C3", "C4"], 6)


# ---- the steps mode: the runner is forced and read back ----------------------------

SYNTHETIC_LOG = """
time=2026-09-23T15:10:50.674+08:00 level=INFO source=sched.go:613 msg="system memory" total="32.0 GiB" free="17.9 GiB" free_swap="0 B"
load_tensors: loading model tensors, this can take a while... (load_mode = mmap)
llama_context: n_batch               = 512
llama_context: n_ubatch              = 512
time=2026-09-23T15:10:52.509+08:00 level=INFO source=sched.go:728 msg="loaded runners" count=1
slot   operator(): id  0 | task 1 | created context checkpoint 1 of 8 (pos_min = 100, pos_max = 200)
slot   operator(): id  0 | task 1 | restored context checkpoint (pos_min = 100)
[GIN] 2026/09/23 - 15:11:12 | 200 |  9.9s |       127.0.0.1 | POST     "/api/chat"
slot   operator(): id  0 | task 2 | erased invalidated context checkpoint (pos_min = 100)
slot   operator(): id  0 | task 2 | created context checkpoint 2 of 8 (pos_min = 300, pos_max = 400)
[GIN] 2026/09/23 - 15:11:24 | 200 |  9.9s |       127.0.0.1 | POST     "/api/chat"
[GIN] 2026/09/23 - 15:40:00 | 200 |  9.9s |       127.0.0.1 | POST     "/api/chat"
""".strip().splitlines()


def test_window_facts_read_the_load_and_the_checkpoints_inside_the_window(probe):
    facts = probe.window_facts(SYNTHETIC_LOG, "2026/09/23", "15:10:00", "15:12:00")
    assert facts["n_batch"] == 512 and facts["load_mode"] == "mmap"
    assert facts["requests"] == 2
    assert (facts["created"], facts["erased"], facts["restored"]) == (2, 1, 1)


def test_window_facts_ignore_a_load_and_requests_outside_the_window(probe):
    facts = probe.window_facts(SYNTHETIC_LOG, "2026/09/23", "15:30:00", "15:50:00")
    assert facts["n_batch"] is None and facts["requests"] == 1
    assert facts["created"] == 0


@pytest.mark.parametrize(("spec", "num_batch", "use_mmap"), [
    ("512", 512, True), ("1024", 1024, True), ("mmap0", 512, False), ("mmap1", 512, True)])
def test_a_step_spec_names_one_runner(probe, spec, num_batch, use_mmap):
    step = probe.parse_step(spec)
    assert (step["num_batch"], step["use_mmap"]) == (num_batch, use_mmap)


def test_an_unknown_step_is_refused(probe):
    with pytest.raises(SystemExit, match="unknown step"):
        probe.parse_step("2048")


def _steps(*rows):
    out = []
    for spec, vs_base, vs_same in rows:
        out.append({"spec": spec, "moved_vs_baseline": vs_base, "moved_vs_same_spec": vs_same})
    return out


@pytest.mark.parametrize(("rows", "label"), [
    ([("512", 0, None), ("1024", 22, None), ("512", 0, 0), ("1024", 22, 0), ("mmap0", 1, None), ("512", 1, 1)],
     "batch_is_a_cause"),
    ([("512", 0, None), ("1024", 22, None), ("512", 1, 1), ("1024", 20, 9), ("512", 0, 0)],
     "batch_moves_but_not_reproducibly"),
    ([("512", 0, None), ("1024", 1, None), ("512", 0, 0), ("1024", 2, 1), ("512", 0, 0)],
     "batch_is_not_a_cause"),
    ([("512", 0, None), ("1024", 22, None), ("512", 7, 7), ("1024", 22, 0), ("512", 0, 0)],
     "baseline_unstable"),
    ([("512", 0, None), ("1024", 22, None)], "incomplete"),
])
def test_the_steps_verdict_is_the_pre_registered_ladder(probe, rows, label):
    assert probe.step_verdict(_steps(*rows))[0] == label


def test_the_mmap_step_is_read_on_its_own(probe):
    rows = _steps(("512", 0, None), ("1024", 22, None), ("512", 0, 0), ("1024", 22, 0),
                  ("mmap0", 15, None), ("512", 0, 0))
    assert "a second cause" in probe.step_verdict(rows)[1]


def test_a_step_whose_batch_the_server_did_not_honour_is_refused(probe, monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "unload", lambda *_a, **_k: None)
    monkeypatch.setattr(probe, "measure", lambda *_a, **_k: {"S01": "x"})
    monkeypatch.setattr(probe, "OllamaClient", lambda *_a, **_k: object())
    monkeypatch.setattr(probe.time, "sleep", lambda *_: None)
    log = tmp_path / "ollama.log"
    monkeypatch.setattr(probe, "LOG_PATH", log)
    # the server loaded batch 512 although 1024 was asked for
    day = probe.datetime.now(probe.LOCAL).strftime("%Y-%m-%d")
    log.write_text(f"time={day}T00:00:00.000+08:00 msg=\"system memory\" free=\"1 GiB\"\n"
                   "llama_context: n_batch               = 512\n"
                   f"[GIN] {day.replace('-', '/')} - 23:59:59 | 200 | 1s | 127.0.0.1 | POST \"/api/chat\"\n")
    with pytest.raises(SystemExit, match="refusing to count"):
        probe.run_steps(["1024"], [{"id": "S01", "question": "q"}], {}, "m", 0, log=lambda *_: None)
