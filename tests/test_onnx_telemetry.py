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

How the measurement is taken, and why it is taken this way
----------------------------------------------------------

The first version of this file counted native threads across ``import
standards; import onnxruntime`` together and demanded the total not move. That
passed on macOS and failed on its first Linux CI run at ``1 -> 2``, which read
like the uploader surviving on manylinux. It was not. Measured in a Linux
container, staging the imports one at a time:

    import numpy         +1   (2 CPUs)   +9 (10 CPUs)
    import chromadb      +0               +0
    import onnxruntime   +1 telemetry on, +0 telemetry off, on x86_64 and arm64

The thread was numpy's OpenBLAS worker pool -- ``nproc - 1`` threads started
when numpy is imported, which on a two-core GitHub runner is exactly one. macOS
never showed it because numpy there links Accelerate, which starts nothing at
import. ``ORT_DISABLE_TELEMETRY`` works on manylinux; the guard's proxy was
wrong, charging onnxruntime for every thread any transitive import had started.

So the delta is now taken across ``import onnxruntime`` alone, with everything
else already imported and OpenBLAS's pool already up, and it is calibrated
against a control arm that leaves telemetry on. The uploader is worth 2 threads
on macOS and 1 on manylinux; if a future wheel makes that 0 the control says so
and the test skips rather than passing vacuously.

Two sharper-sounding assertions were tried and rejected on evidence. Naming the
thread: every thread in ``/proc/self/task/*/comm`` reads ``python``, ORT's
uploader included, so there is no name to match. Watching for the socket: no
outbound connection exists at import in either arm -- ``/proc/self/net/tcp`` is
empty and the process holds no socket fd -- because the uploader connects on a
timer, not when it is constructed. The thread's existence is the earliest
honest signal, and it is the one thing that has to be true at teardown.
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


def _threads_started_by_importing_onnxruntime(arm: str) -> dict:
    """Run one arm of the child and hand back what it measured.

    The switch is stripped from the child's environment either way: this
    process has it set as a side effect of its own imports, and a child that
    inherited it would report a clean measurement with the source line deleted.
    """
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO / "tests")]),
    }
    environment.pop(SWITCH, None)

    finished = subprocess.run(
        [sys.executable, str(CHILD), arm],
        capture_output=True, text=True, env=environment, cwd=REPO, timeout=180,
    )
    assert finished.returncode == 0, finished.stderr
    return json.loads(finished.stdout.strip().splitlines()[-1])


def test_onnxruntime_starts_no_thread_of_its_own():
    """The uploader, measured against a control that leaves it running."""
    control = _threads_started_by_importing_onnxruntime("baseline")

    if control["before"] < 0:
        pytest.skip(f"no native thread count on {sys.platform}")

    assert control["switch"] is None, (
        "the control arm had the switch set, so it is not a control: it "
        "measures the fix twice and would agree with itself with the fix gone"
    )

    uploader = control["after"] - control["before"]
    if uploader == 0:
        pytest.skip(
            f"onnxruntime {control['version']} starts no thread at import on "
            f"{sys.platform} even with telemetry left on, so this measurement "
            "cannot tell the fix from its absence. Either the wheel stopped "
            "shipping 1DS -- in which case delete the switch and this file "
            "together -- or the uploader moved somewhere this no longer looks."
        )

    fixed = _threads_started_by_importing_onnxruntime("fixed")

    # The measurement leads, and carries the switch's value in its message,
    # because "the uploader is running" is the finding and "the switch is
    # unset" is only the likeliest reason for it.
    assert fixed["after"] == fixed["before"], (
        f"importing aoi_agent.store.standards first left onnxruntime starting "
        f"{fixed['after'] - fixed['before']} thread(s) at import "
        f"({fixed['before']} -> {fixed['after']}), against {uploader} with "
        f"telemetry left on -- and the child had {SWITCH}="
        f"{fixed['switch']!r}. The uploader is back, and with it a race "
        "between its HTTP callback and the static destructors that aborts "
        "this suite after every test passes."
    )
    assert fixed["switch"] == "1", (
        "no thread was started, but the child did not set the switch either, "
        "so this measurement says nothing about the fix"
    )
