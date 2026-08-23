"""Does an escalation actually outlive the process that raised it?

Every other test in this suite hands ``build_graph`` an ``InMemorySaver``,
which is the right thing when the subject is routing: one run, one process, no
file. But the invariant in CLAUDE.md is specifically that the checkpointer is
*not* that -- and a suite that only ever uses the forbidden object cannot fail
when someone makes it the default.

So this file exercises the durable default, and it does it across a process
boundary. Suspending and resuming inside one interpreter would prove less than
the invariant claims: the saver object never leaves memory, so a run could
resume off nothing but its own live state and the test would still pass. Here
the interpreter that raised the escalation has exited before anything tries to
answer it. What the second process reads, it reads off the file.

Both processes go through the *default* checkpointer -- neither passes one in
-- so the thing under test is the wiring the CLI and the station actually use.
The file is a temporary one, named through ``AOI_AGENT_CHECKPOINT_PATH`` in
the child and by patching ``CHECKPOINT_PATH`` in the parent; ``data/`` is never
touched and nothing is left behind.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from aoi_agent.graph import flow
from aoi_agent.graph import checkpoint

from test_graph import StubClient, stub_tools  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parents[1]
RAISER = Path(__file__).parent / "escalation_in_another_process.py"
REFERENCE = "20085294#3"


def raise_escalation_in_another_process(db: Path) -> dict:
    """Run the flow to its ``interrupt`` in a separate interpreter, and let it die."""
    env = {
        **os.environ,
        "AOI_AGENT_CHECKPOINT_PATH": str(db),
        "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO / "tests")]),
    }
    finished = subprocess.run(
        [sys.executable, str(RAISER), REFERENCE],
        capture_output=True, text=True, env=env, cwd=REPO, timeout=180,
    )
    assert finished.returncode == 0, finished.stderr
    return json.loads(finished.stdout.strip().splitlines()[-1])


@pytest.fixture
def resuming_process(tmp_path, monkeypatch, stub_tools):  # noqa: F811
    """This interpreter, made to look like one that has just started.

    Fresh checkpointer over the same file: the module cache is cleared, so
    ``make_checkpointer`` opens its own connection rather than handing back one
    some earlier test warmed up.
    """
    monkeypatch.setattr(checkpoint, "CHECKPOINT_PATH", str(tmp_path / "checkpoints.db"))
    monkeypatch.setattr(checkpoint, "_saver", None)
    return tmp_path / "checkpoints.db"


def test_an_escalation_survives_the_process_that_raised_it(resuming_process):
    """The claim in full: raised by one interpreter, answered by another.

    The child exits before this process builds anything. If the checkpointer
    were an ``InMemorySaver``, the state would have died with it and the resume
    below would have nothing to resume -- so this test is what stands between
    the invariant and the word "durable" in a docstring.
    """
    raised = raise_escalation_in_another_process(resuming_process)

    assert raised["payload"]["candidate_ref"] == REFERENCE
    assert raised["trace"] == ["classify", "gather_context", "reason"]

    # Fresh graph, fresh checkpointer, same file. Nothing is shared with the
    # process above but the bytes on disk.
    graph = flow.build_graph(StubClient(confident=False))
    config = {"configurable": {"thread_id": REFERENCE}}

    suspended = dict(graph.get_state(config).values or {})
    assert suspended, (
        "a second process found nothing for this thread: the first process's "
        "state died with it, which is what an in-memory checkpointer does"
    )
    assert suspended["trace"] == ["classify", "gather_context", "reason"], (
        "the evidence was gathered in the other process and must not be re-gathered"
    )
    assert suspended["board_context"]["machine_id"] == "M22"
    assert suspended["agent_rationale"] == "raised elsewhere"

    resumed = graph.invoke(
        Command(resume={"verdict": "mousebite", "reviewer": "mike"}), config=config
    )

    assert "__interrupt__" not in resumed, (
        "the run restarted instead of resuming: the other process's state was lost"
    )
    assert resumed["verdict"] == "mousebite"
    assert resumed["decided_by"] == "human"
    assert resumed["trace"] == [
        "classify", "gather_context", "reason", "escalate", "record_human"
    ], "resuming replays no node the first process already ran"


def test_the_default_checkpointer_is_a_file_and_not_a_dictionary(resuming_process):
    """The narrow version of the same invariant, so a swap fails loudly here too."""
    saver = checkpoint.make_checkpointer()

    assert isinstance(saver, SqliteSaver)
    assert resuming_process.exists()


def test_a_checkpoint_path_is_created_under_a_directory_that_does_not_exist_yet(tmp_path):
    """First run on a clean clone: `data/` is gitignored and may not be there."""
    target = tmp_path / "nested" / "deeper" / "checkpoints.db"
    checkpoint.make_checkpointer(str(target))

    assert target.exists()
