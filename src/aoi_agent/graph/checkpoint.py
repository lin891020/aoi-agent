"""Where suspended runs live.

The flow's whole reason for being a graph is that an escalated candidate
suspends and resumes later -- possibly in another process, possibly the next
day, when an operator finally opens the review station. An in-memory
checkpointer cannot honour that: it makes the hand-off look right in a single
CLI run and quietly loses the queue the moment the process exits.

So the default is a SQLite file, alongside the store. Swapping it for Postgres
is one function.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

CHECKPOINT_PATH = os.getenv("AOI_AGENT_CHECKPOINT_PATH", "data/checkpoints.db")

_saver: SqliteSaver | None = None


def make_checkpointer(path: str | None = None) -> SqliteSaver:
    """A process-wide checkpointer over one SQLite file.

    ``check_same_thread=False`` because the web station serves requests from a
    thread pool; SQLite's own locking serialises the writes, and checkpoint
    writes are small and infrequent next to a 20B model's inference.
    """
    global _saver
    if path is None and _saver is not None:
        return _saver

    target = Path(path or CHECKPOINT_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target, check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()

    if path is None:
        _saver = saver
    return saver
