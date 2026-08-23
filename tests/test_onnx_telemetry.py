"""onnxruntime must not start its telemetry uploader in this process.

The suite used to pass and then abort: ``650 passed``, then

    libc++abi: terminating due to uncaught exception of type
      std::__1::system_error: recursive_mutex lock failed: Invalid argument

and exit 134. Six full runs in ten, and every one of sixteen macOS crash reports
named the same frames -- ``Microsoft::Applications::Events::HttpClientManager::
onHttpResponse`` on a ``PlatformAbstraction::WorkerThread``, calling
``DebugEventSource::DispatchEvent``, which locks a ``recursive_mutex`` that the
C++ static destructors had already destroyed. That is Microsoft's 1DS
"OneCollector" client, statically linked into onnxruntime's wheel and started
when the module is imported. Nothing joins it at exit, so whether the process
aborts depends on whether an HTTP response happens to land during teardown.

It reached this process through exactly one door: Chroma's default embedding
function is ONNX MiniLM, ``aoi_agent.store.standards`` is the only module that
imports chromadb, and ``tests/test_standards_retrieval.py`` is the only test
that builds a real index. Removing that one file took ten runs to zero aborts;
it is also the test that catches a shipped retrieval defect, so it stays and the
uploader goes.

What is guarded here is the fix's one fragile property: ``ORT_DISABLE_TELEMETRY``
is read while onnxruntime constructs its ``Env``, so the assignment only works
if it runs *before* the import that triggers it. Move it below ``import
chromadb``, or delete it as a stray line nobody could explain, and the suite
goes back to aborting at random -- with every test still passing, which is the
whole reason this is worth a test of its own.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aoi_agent.store import standards

REPO = Path(__file__).resolve().parents[1]
CHILD = Path(__file__).parent / "onnx_threads_in_another_process.py"
SWITCH = "ORT_DISABLE_TELEMETRY"


def test_importing_standards_sets_the_switch():
    """Importing the module is what sets it -- no fixture, no conftest."""
    assert os.environ.get(SWITCH) == "1"


def test_the_switch_is_set_above_the_import_it_has_to_beat():
    """Order, not presence. Below ``import chromadb`` the line does nothing."""
    source = Path(standards.__file__).read_text()
    assignment = source.index(f'os.environ.setdefault("{SWITCH}"')
    chroma = source.index("\nimport chromadb")
    assert assignment < chroma, (
        f"{SWITCH} is set after chromadb is imported, which is after "
        "onnxruntime has already built its Env and started the uploader. "
        "The line has to come first to do anything."
    )


def test_onnxruntime_starts_no_thread_of_its_own():
    """The uploader, measured: import onnxruntime and count native threads.

    In a child, with the switch stripped from its environment -- this process
    already has it set as a side effect of its own imports, and a child that
    inherited it would pass this check with the source line deleted.
    """
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO / "tests")]),
    }
    environment.pop(SWITCH, None)

    finished = subprocess.run(
        [sys.executable, str(CHILD)],
        capture_output=True, text=True, env=environment, cwd=REPO, timeout=180,
    )
    assert finished.returncode == 0, finished.stderr
    counts = json.loads(finished.stdout.strip().splitlines()[-1])

    if counts["before"] < 0:
        pytest.skip(f"no thread count on {sys.platform}")

    assert counts["switch"] == "1", (
        "the child did not set the switch itself, so this measurement says "
        "nothing about the fix"
    )
    assert counts["after"] == counts["before"], (
        f"onnxruntime started {counts['after'] - counts['before']} thread(s) at "
        f"import ({counts['before']} -> {counts['after']}). The telemetry "
        "uploader is back, and with it a race between its HTTP callback and the "
        "static destructors that aborts this suite after every test passes."
    )
