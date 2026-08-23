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
when onnxruntime builds its ``Env`` -- at import on this laptop, at the first
inference session on CI. Nothing joins it at exit, so whether the process aborts
depends on whether an HTTP response happens to land during teardown.

It reached this process through exactly one door: Chroma's default embedding
function is ONNX MiniLM, ``aoi_agent.store.standards`` is the only module that
imports chromadb, and ``tests/test_standards_retrieval.py`` is the only test
that builds a real index. Removing that one file took ten runs to zero aborts;
it is also the test that catches a shipped retrieval defect, so it stays and the
uploader goes.

What is guarded here is the fix's one fragile property: ``ORT_DISABLE_TELEMETRY``
is read while onnxruntime constructs its ``Env``, so the assignment only works
if it runs *before* the import that leads to it. Move it below ``import
chromadb``, or delete it as a stray line nobody could explain, and the suite
goes back to aborting at random -- with every test still passing, which is the
whole reason this is worth a test of its own.

How the measurement is taken, and why it is taken this way
----------------------------------------------------------

The first version of this file counted native threads across ``import
standards; import onnxruntime`` together and demanded the total not move. That
passed on macOS and failed on its first Linux CI run at ``1 -> 2``, which read
like the uploader surviving on manylinux. It was not. Staged one import at a
time in a Linux container:

    import numpy         +1  (2 CPUs)   +9  (10 CPUs)
    import chromadb      +0             +0
    import onnxruntime   +1 telemetry on, +0 telemetry off, x86_64 and aarch64

The thread was numpy's OpenBLAS worker pool -- ``nproc - 1`` threads started
when numpy is imported, which on a two-core GitHub runner is exactly one. macOS
never showed it because numpy links Accelerate there, which starts nothing at
import. The switch was doing its job; the guard was charging onnxruntime for
every thread any transitive import had started.

Fixing the attribution was not enough, because *when* the uploader starts is
also not the same everywhere:

    threads started        macOS    manylinux in Docker    GitHub runner
    by the import            2              1                    0
    by the first embed      +1             +0                   +1

On the runner onnxruntime starts nothing at import; the uploader arrives with
the first inference session. An import-time guard is therefore blind on exactly
the machine CI runs -- it skipped there, green and useless, which is the same
disease in a new place. So the count is taken after one real embedding, through
the door this project actually uses, and a thread alive then is a thread alive
at static destruction.

1DS is statically linked into all three wheels, checked by byte-grep rather than
inferred: ``OneCollector``, ``mobile.events.data.microsoft.com``,
``ORT_DISABLE_TELEMETRY`` and the offline store's ``onnxruntime.db`` all appear
in the macOS, manylinux x86_64 and manylinux aarch64 binaries. The runner is not
a wheel without telemetry, it is a wheel that starts it later.

There is no platform constant here on purpose. The uploader is worth 3 threads
on macOS and 1 on Linux, so what is asserted is the difference between the two
arms rather than a number: with the fix, strictly fewer threads survive than
without it. Delete the switch and the arms become the same program, the
difference goes to zero, and this fails.

Two sharper-sounding assertions were tried and rejected on evidence. Naming the
thread: every entry in ``/proc/self/task/*/comm`` reads ``python``, the
uploader's included, so there is no name to match. Watching for the socket: 1DS
uploads on a timer, the runner never even creates the offline store, and a test
that needs an outbound connection to prove a negative passes for free on any
machine without one.
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


def _threads_after_one_embedding(arm: str) -> dict:
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
        capture_output=True, text=True, env=environment, cwd=REPO, timeout=600,
    )
    assert finished.returncode == 0, finished.stderr
    return json.loads(finished.stdout.strip().splitlines()[-1])


def test_onnxruntime_leaves_no_thread_of_its_own_running():
    """The uploader, measured against a control that leaves it running."""
    control = _threads_after_one_embedding("baseline")

    if control["base"] < 0:
        pytest.skip(f"no native thread count on {sys.platform}")

    assert control["switch"] is None, (
        "the control arm had the switch set, so it is not a control: it "
        "measures the fix twice and would agree with itself with the fix gone"
    )

    fixed = _threads_after_one_embedding("fixed")

    control_threads = control["after_embed"] - control["base"]
    fixed_threads = fixed["after_embed"] - fixed["base"]
    uploader = control_threads - fixed_threads

    assert uploader > 0, (
        f"onnxruntime {control['version']} left {fixed_threads} thread(s) "
        f"running after one embedding with aoi_agent.store.standards imported "
        f"first, and {control_threads} without it -- the same, so the switch "
        f"removed nothing. The child had {SWITCH}={fixed['switch']!r}. Either "
        f"the line above `import chromadb` in {Path(standards.__file__).name} "
        "is gone and the uploader is back -- and with it the race between its "
        "HTTP callback and the static destructors that aborts this suite after "
        "every test passes -- or onnxruntime stopped shipping 1DS, in which "
        "case the byte-grep in this file's docstring is the thing to re-run "
        "before deleting the switch and this test together.\n"
        f"  control: {control}\n  fixed:   {fixed}"
    )

    assert fixed["switch"] == "1", (
        f"the switch removed {uploader} thread(s), but the child reports "
        f"{SWITCH}={fixed['switch']!r} -- so something other than this "
        "repository's line turned telemetry off, and the guard is measuring "
        "someone else's fix"
    )
