"""Import the standards module, then onnxruntime, and report the thread count.

A child process because the question is what importing does, and the parent has
already done it. It also has ``ORT_DISABLE_TELEMETRY`` in its own environment by
then -- ``standards`` puts it there -- and a child inheriting that would pass
this check whether or not the line still exists. The caller strips it; this
script asserts nothing and only reports, so that the assertion and its message
live with the test.

Prints one line of JSON: the thread count before either import and after both.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


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


before = native_threads()
from aoi_agent.store import standards  # noqa: E402,F401  (the import is the subject)
import onnxruntime  # noqa: E402,F401

print(json.dumps({
    "before": before,
    "after": native_threads(),
    "switch": os.environ.get("ORT_DISABLE_TELEMETRY"),
}))
