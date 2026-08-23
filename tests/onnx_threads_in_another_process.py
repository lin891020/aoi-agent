"""Count the native threads onnxruntime leaves running after one real embedding.

A child process because the question is what the imports do, and the parent has
already done them. It also has ``ORT_DISABLE_TELEMETRY`` in its own environment
by then -- ``standards`` puts it there -- and a child inheriting that would
report a clean measurement with the source line deleted. The caller strips it;
this script asserts nothing and only reports, so that the assertion and its
message live with the test.

Two arms, identical but for whether the repository's fix has run:

``baseline``  imports ``chromadb`` directly, so nothing sets the switch and
              onnxruntime keeps its telemetry. This is the control: it says what
              the uploader is worth on this machine, which is the only honest
              way to know the measurement can still see it.
``fixed``     imports ``aoi_agent.store.standards``, whose first statement is
              the switch. Same modules, one bit different.

Both arms then do what this project actually does -- build Chroma's default
embedding function and embed one string -- and count threads afterwards. The
measurement is taken *there*, and not at ``import onnxruntime``, because where
the uploader starts is not the same on every platform:

    threads started        macOS    manylinux in Docker    GitHub runner
    by the import            2              1                    0
    by the first embed      +1             +0                   +1

On the runner onnxruntime starts nothing at all at import and the uploader
appears with the first inference session, so an import-time measurement is blind
on exactly the machine CI runs. After one embedding every platform shows it, and
a thread that exists then is a thread that is still there at static destruction,
which is the whole problem.

Prints one line of JSON.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

#: Long enough for a thread the session started to appear and for 1DS to get
#: past its own start-up, short enough that two arms stay cheap.
SETTLE = 3.0

PHRASE = "a printed circuit board with an open trace"


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

# numpy's OpenBLAS starts `nproc - 1` workers when it is imported, and nothing
# to do with onnxruntime. Get them up, and get them counted in the baseline,
# before anything is attributed to anyone -- the first version of this file
# charged them to onnxruntime and failed CI on a two-core runner.
import numpy  # noqa: E402

numpy.zeros((64, 64)) @ numpy.zeros((64, 64))
time.sleep(1.0)
base = native_threads()

import onnxruntime  # noqa: E402

time.sleep(1.0)
after_import = native_threads()

from chromadb.utils import embedding_functions  # noqa: E402

embed = embedding_functions.DefaultEmbeddingFunction()
embed([PHRASE])
time.sleep(SETTLE)

print(json.dumps({
    "arm": arm,
    "base": base,
    "after_import": after_import,
    "after_embed": native_threads(),
    "switch": os.environ.get("ORT_DISABLE_TELEMETRY"),
    "version": onnxruntime.__version__,
}))
