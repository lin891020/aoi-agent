"""The latency report's sweep must not refuse a run for the model it measured.

`competing_processes` lists every claimant above the CPU floor, which is right
for the MPS benchmark: an Ollama runner beside a torch job *is* contention
there. For the explanation-latency report the runner is the subject. Two
English measurements on 2026-08-30 were refused because the sweep taken the
instant the run ended found that runner at 6%, finishing the last call.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys


def _load(name: str):
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PS = """  PID %CPU COMMAND
71503   6.0 /opt/homebrew/Cellar/ollama/0.32.13/libexec/lib/ollama/llama-server --model /Users/x/.ollama/models/blobs/sha256-abc
80001  95.0 /Users/x/.venv/bin/python scripts/train.py
80002   0.1 /Applications/Cursor.app/Contents/MacOS/Cursor
"""


def test_ollamas_own_runner_is_not_a_competitor_but_a_torch_job_is():
    rl = _load("reverifier_latency")
    report = _load("latency_report")
    raw = rl.competing_processes(PS, own_pid=1)
    assert any("llama-server" in line for line in raw), "the shared sweep still sees the runner"
    kept = [line for line in raw if not report.is_ollama_runner(line)]
    assert len(kept) == 1 and "train.py" in kept[0]
