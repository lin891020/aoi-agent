"""Count the native threads that ``import onnxruntime`` -- and nothing else --
starts, in a process that has done everything else first.

A child process because the question is what one import does, and the parent has
already done it. It also has ``ORT_DISABLE_TELEMETRY`` in its own environment by
then -- ``standards`` puts it there -- and a child inheriting that would pass the
check whether or not the line still exists. The caller strips it; this script
asserts nothing and only reports, so that the assertion and its message live
with the test.

Two arms, identical but for whether the repository's fix has run:

``baseline``  imports ``chromadb`` directly, so nothing sets the switch and
              onnxruntime builds its Env with telemetry on. This is the control:
              it says how many threads the uploader is worth on this platform,
              and the test refuses to draw a conclusion if the answer is zero.
``fixed``     imports ``aoi_agent.store.standards``, whose first statement is the
              switch. Same modules, one bit different.

Both arms take the ``before`` count *after* their imports and a settle, so that
the delta belongs to onnxruntime alone. The first version of this bracketed the
whole compound import and attributed every thread in it to onnxruntime, which is
what broke on CI: numpy's OpenBLAS starts ``nproc - 1`` workers when it is
imported, so on a two-core runner the count went 1 -> 2 with the fix working
perfectly. The matmul below is there to make sure that pool is up before the
measurement starts rather than during it.

Prints one line of JSON.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

#: Long enough for a thread started by the import to appear, short enough that
#: two arms cost four seconds. The uploader shows up immediately; this is only
#: insurance against measuring a thread that is still being created.
SETTLE = 1.0


def native_threads() -> int:
    """Every thread in this process, including the ones C++ started.

    ``threading.enumerate()`` cannot see these: the uploader is a ``std::thread``
    inside onnxruntime and was never registered with Python. Returns -1 where
    the platform has no cheap way to ask, and the test skips.
    """
    if sys.platform.startswith("linux"):
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
        return -1
    if sys.platform == "darwin":
        listing = subprocess.run(
            ["ps", "-M", str(os.getpid())], capture_output=True, text=True
        ).stdout
        return len(listing.strip().splitlines()) - 1  # minus the header
    return -1


arm = sys.argv[1] if len(sys.argv) > 1 else ""
if arm == "fixed":
    from aoi_agent.store import standards  # noqa: F401  (the import is the subject)
elif arm == "baseline":
    import chromadb  # noqa: F401  (the same modules, without the switch)
else:
    raise SystemExit(f"usage: {sys.argv[0]} baseline|fixed (got {arm!r})")

import numpy  # noqa: E402

numpy.zeros((64, 64)) @ numpy.zeros((64, 64))  # OpenBLAS's pool, up before we look
time.sleep(SETTLE)

before = native_threads()
import onnxruntime  # noqa: E402,F401  (this line is the whole measurement)

time.sleep(SETTLE)

print(json.dumps({
    "arm": arm,
    "before": before,
    "after": native_threads(),
    "switch": os.environ.get("ORT_DISABLE_TELEMETRY"),
    "version": onnxruntime.__version__,
}))
