"""Eight concurrent first searches in a fresh interpreter, as `/ask` fans out.

Run as a script by `tests/test_standards_retrieval.py`: the race is in opening
the index, which a process does once, so it can only be reached from a process
that has not opened it yet. Prints one line per search and exits non-zero if
any came back empty or raised.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from aoi_agent.store import standards

standards.CHROMA_DIR = Path(sys.argv[1])
CLASSES = ["open", "short", "mousebite", "spur", "copper", "pin-hole", "open", "short"]
barrier = threading.Barrier(len(CLASSES))
outcomes: list[str] = [""] * len(CLASSES)


def search(position: int, defect_class: str) -> None:
    barrier.wait()
    try:
        passages = standards.search("acceptance criteria", top_k=1, defect_class=defect_class)
        outcomes[position] = "ok" if passages else "empty"
    except Exception as error:  # noqa: BLE001 -- any failure is the outcome being measured
        outcomes[position] = f"{type(error).__name__}: {error}"


threads = [threading.Thread(target=search, args=(i, c)) for i, c in enumerate(CLASSES)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
for outcome in outcomes:
    print(outcome)
sys.exit(0 if all(o == "ok" for o in outcomes) else 1)
