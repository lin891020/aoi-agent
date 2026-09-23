"""The two-sentence comparison reads its verdict off a rule written first.

What would let it say the wrong thing without raising: a reload between arms
read as a sentence effect; the same arm disagreeing with itself and the
difference still being called; a run order that is not ABAB; an entry that
omits the correction it exists to make.
"""

from __future__ import annotations

import language_note_arms as arms
import pytest


def _rows(ok_ids: set[str], plans: dict[str, str] | None = None) -> list[dict]:
    plans = plans or {}
    rows = []
    for i in range(1, 43):
        q = f"A{i:02d}"
        rows.append({"id": q, "ok": q in ok_ids, "expect_refusal": False, "no_plan": False,
                     "planned": plans.get(q, f"plan-{q}")})
    for i in range(1, 29):
        q = f"R{i:02d}"
        rows.append({"id": q, "ok": True, "expect_refusal": True, "no_plan": False,
                     "planned": "(refused)"})
    return rows


def _run(arm: str, answered: int, plans: dict[str, str] | None = None) -> dict:
    ok = {f"A{i:02d}" for i in range(1, answered + 1)}
    return {"arm": arm, **arms.summarise(_rows(ok, plans))}


def test_summarise_counts_the_evals_two_figures():
    s = arms.summarise(_rows({"A01", "A02", "A03"}))
    assert (s["answered"], s["answerable"], s["refused"], s["refusable"]) == (3, 42, 28, 28)
    assert s["no_plan"] == 0 and s["no_plan_ids"] == [] and len(s["planned"]) == 70


def test_moved_compares_the_scored_calls_only():
    a = _run("current", 25)
    b = _run("current", 25, {"A07": "other", "A09": "other"})
    assert arms.moved(a, b) == ["A07", "A09"]


@pytest.mark.parametrize(("old", "new", "label"), [
    ((28, 28), (25, 25), "the_sentence"),
    ((28, 27), (25, 24), "the_sentence"),
    ((25, 25), (25, 25), "not_the_sentence"),
    ((26, 26), (25, 25), "not_the_sentence"),
    ((27, 27), (25, 25), "unresolved"),
])
def test_the_ladder_is_the_pre_registered_one(old, new, label):
    runs = [_run("2026-08-30", old[0]), _run("current", new[0]),
            _run("2026-08-30", old[1]), _run("current", new[1])]
    assert arms.verdict(runs, [512])[0] == label


def test_an_arm_that_disagrees_with_itself_is_not_read():
    runs = [_run("2026-08-30", 28), _run("current", 25),
            _run("2026-08-30", 26), _run("current", 25)]
    assert arms.verdict(runs, [512])[0] == "baseline_unstable"
    drifted = _run("2026-08-30", 28, {"A30": "x", "A31": "x", "A32": "x"})
    runs = [_run("2026-08-30", 28), _run("current", 25), drifted, _run("current", 25)]
    assert arms.verdict(runs, [512])[0] == "baseline_unstable"


@pytest.mark.parametrize("loads", [[], [1024], [512, 512], [512, None]])
def test_anything_but_one_512_load_is_invalid(loads):
    runs = [_run("2026-08-30", 28), _run("current", 25),
            _run("2026-08-30", 28), _run("current", 25)]
    label, sentence = arms.verdict(runs, loads)
    assert label == "invalid" and "exactly one" in sentence


_LOG = """\
time=2026-09-23T22:10:01.000+08:00 level=INFO source=server.go msg="system memory" total="32.0 GiB" free="20.1 GiB"
llama_context: n_batch    = 512
time=2026-09-23T22:10:09.000+08:00 level=INFO source=server.go msg="loaded runners"
[GIN] 2026/09/23 - 22:10:20 | 200 | 9s | 127.0.0.1 | POST     "/api/chat"
time=2026-09-23T22:40:01.000+08:00 level=INFO source=server.go msg="system memory" total="32.0 GiB" free="19.0 GiB"
llama_context: n_batch    = 1024
time=2026-09-23T22:40:09.000+08:00 level=INFO source=server.go msg="loaded runners"
time=2026-09-23T23:59:01.000+08:00 level=INFO source=server.go msg="system memory" total="32.0 GiB" free="19.0 GiB"
llama_context: n_batch    = 512
""".splitlines()


def test_loads_are_read_from_the_window_with_their_batch():
    assert arms.loads_in(_LOG, "2026-09-23", "22:00:00", "22:30:00") == [512]
    assert arms.loads_in(_LOG, "2026-09-23", "22:00:00", "23:00:00") == [512, 1024]
    assert arms.loads_in(_LOG, "2026-09-23", "23:00:00", "23:30:00") == []
    assert arms.loads_in(_LOG, "2026-09-22", "22:00:00", "23:00:00") == []


def _record(label_runs) -> dict:
    runs = [_run(arm, n) for arm, n in label_runs]
    label, sentence = arms.verdict(runs, [512])
    return {"date": "2026-09-23", "commit": "abc1234",
            "window": {"day": "2026-09-23", "start": "22:10:00", "end": "23:10:00"},
            "loads": [512], "log_facts": {},
            "prompt_hash": {"2026-08-30": "af31604385e8", "current": "b519db949d60"},
            "runs": runs, "verdict": label, "verdict_sentence": sentence}


def test_the_entry_carries_the_correction_and_both_hashes():
    body = arms.render(_record([("2026-08-30", 28), ("current", 25),
                                ("2026-08-30", 28), ("current", 25)]))
    assert "Correction to the 2026-09-19 entry" in body
    assert "af31604385e8" in body and "b519db949d60" in body and "bbe1df8" in body
    assert "**Verdict: the_sentence.**" in body
    assert "| 1 | `2026-08-30` | 28 | 28 | 0 | — |" in body
    assert "A26, A27, A28" in body            # right under the old sentence, wrong under the new
    assert "read 25 on the same sentence" in body and "25 and 25 of 42" in body


def test_an_invalid_run_renders_no_per_question_reading():
    record = _record([("2026-08-30", 28), ("current", 25), ("2026-08-30", 28), ("current", 25)])
    record["loads"] = [512, 512]
    record["verdict"], record["verdict_sentence"] = arms.verdict(record["runs"], [512, 512])
    body = arms.render(record)
    assert "**Verdict: invalid.**" in body and "Questions right under" not in body
