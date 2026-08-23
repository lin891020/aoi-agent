"""Every threshold must state where it came from, and be that value.

CLAUDE.md's invariant is that thresholds come from the sweep or from the work
instructions. `tests/test_response_budget.py` holds exactly one of them to it.
The rest were held to it by nothing, and it showed: an outside review found
`CONFIDENT` cited to "WI-300 decision authority" when WI-300 states no such
number, and `ESCALATE_BELOW` cited to a sweep in docs/benchmarks.md that had
never been run. Both citations had been true-looking for as long as anyone
cared to check, because checking meant reading four files and believing a
table.

So the table in docs/architecture.md is the contract, and this file is the
enforcement:

- every row states the value the code actually carries
- every row cites a file a reader can open -- a script they can run, or a
  document that states the number
- a source that names a work instruction is checked against the work
  instruction, in the direction the invariant requires: the document states
  the number and the code reads it, not the reverse
- a threshold added to the code and not to the table fails the suite

The last one is the point. The others catch drift in a number; that one catches
a number arriving with no provenance at all, which is how both of the wrong
citations got in.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from aoi_agent.aoi.matching import FRAGMENT_GAP_PX, IOU_THRESHOLD
from aoi_agent.graph.flow import CONFIDENT, ESCALATE_BELOW
from aoi_agent.llm.ollama import RESPONSE_BUDGET_S
from aoi_agent.vision.inference import DEFAULT_DISMISS_THRESHOLD, LOW_CONFIDENCE

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = ROOT / "docs/architecture.md"
WI_300 = ROOT / "data/standards/reverification-procedure.md"

#: The constants the table is a contract over, and where they live. Adding a
#: numeric module-level constant to one of these modules without adding a table
#: row fails ``test_no_threshold_reaches_the_code_without_a_row``.
CONSTANTS = {
    "DEFAULT_DISMISS_THRESHOLD": DEFAULT_DISMISS_THRESHOLD,
    "LOW_CONFIDENCE": LOW_CONFIDENCE,
    "ESCALATE_BELOW": ESCALATE_BELOW,
    "CONFIDENT": CONFIDENT,
    "RESPONSE_BUDGET_S": RESPONSE_BUDGET_S,
    "IOU_THRESHOLD": IOU_THRESHOLD,
    "FRAGMENT_GAP_PX": FRAGMENT_GAP_PX,
}

THRESHOLD_MODULES = (
    "src/aoi_agent/graph/flow.py",
    "src/aoi_agent/vision/inference.py",
    "src/aoi_agent/aoi/matching.py",
    "src/aoi_agent/llm/ollama.py",
)

#: Paths under the dataset are gitignored -- it is a 231MB clone, rebuilt by
#: script. A citation into it is checkable by anyone who has the data, which is
#: what the ``dataset`` marker is for; the always-on tests check the shape of
#: the citation rather than its content.
DATASET_PREFIX = "data/DeepPCB/"


def table_rows() -> list[tuple[str, str, str]]:
    """``(constant, value, source)`` from the thresholds table."""
    text = ARCHITECTURE.read_text()
    section = text.split("## Thresholds and where they come from", 1)
    assert len(section) == 2, "docs/architecture.md no longer has a thresholds table"
    body = section[1].split("\n## ", 1)[0]

    rows = []
    for line in body.splitlines():
        match = re.match(r"\|\s*`(\w+)`\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*\|\s*$", line)
        if match:
            rows.append((match.group(1), match.group(2), match.group(3)))
    assert rows, "the thresholds table parsed to nothing"
    return rows


def cited_paths(source: str) -> list[str]:
    """Every repository path the source cell names."""
    return [
        token
        for token in re.findall(r"`([^`]+)`", source)
        if "/" in token and token.endswith((".py", ".md"))
    ]


def test_the_table_covers_every_constant():
    named = {constant for constant, _, _ in table_rows()}
    assert named == set(CONSTANTS), (
        "docs/architecture.md's thresholds table and this test disagree about "
        f"which constants exist: {named ^ set(CONSTANTS)}"
    )


@pytest.mark.parametrize("constant,stated,_source", table_rows())
def test_every_row_states_the_value_the_code_carries(constant, stated, _source):
    """A table that describes last month's constant is worse than no table."""
    assert float(stated) == pytest.approx(float(CONSTANTS[constant])), (
        f"docs/architecture.md says {constant} is {stated}, the code says "
        f"{CONSTANTS[constant]}"
    )


@pytest.mark.parametrize("constant,_value,source", table_rows())
def test_every_row_cites_something_a_reader_can_open(constant, _value, source):
    """Prose is not a citation.

    "the operating-point sweep" and "WI-300 decision authority" both read like
    sources and neither can be opened. The row has to name a file.
    """
    paths = cited_paths(source)
    assert paths, f"{constant}'s source cites no file: {source!r}"
    for path in paths:
        if path.startswith(DATASET_PREFIX):
            continue
        assert (ROOT / path).exists(), (
            f"{constant} cites {path}, which does not exist"
        )


@pytest.mark.parametrize("constant,value,source", table_rows())
def test_a_cited_work_instruction_states_the_number(constant, value, source):
    """The direction of the dependency, held in place.

    A work instruction that does not contain the number is not the source of
    the number, whatever the table says -- and editing the document to match
    the code afterwards is the failure this whole file exists to catch, so the
    documents cite values only where they genuinely fix them.
    """
    for path in cited_paths(source):
        if not path.startswith("data/standards/"):
            continue
        text = (ROOT / path).read_text()
        # A work instruction writes "10 seconds", not "10.0". Accept any
        # spelling of the same number rather than forcing the document to
        # adopt the code's formatting -- that is the reverse dependency again,
        # in miniature.
        spellings = {value, value.rstrip("0").rstrip("."), f"{float(value):g}"}
        assert any(
            re.search(rf"(?<![\d.]){re.escape(s)}(?![\d])", text) for s in spellings
        ), f"{constant} is cited to {path}, which does not state {value}"


def test_no_threshold_reaches_the_code_without_a_row():
    """A number arriving with no provenance is how the wrong citations got in."""
    undocumented = []
    for relative in THRESHOLD_MODULES:
        tree = ast.parse((ROOT / relative).read_text())
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or not target.id.isupper():
                continue
            numeric = (
                isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, (int, float))
                and not isinstance(node.value.value, bool)
            ) or (
                # ``ESCALATE_BELOW = DEFAULT_DISMISS_THRESHOLD`` -- a threshold
                # derived from another threshold is still a threshold, and the
                # derivation is exactly what the table has to record.
                isinstance(node.value, ast.Name)
                and node.value.id in CONSTANTS
            )
            if numeric and target.id not in CONSTANTS:
                undocumented.append(f"{relative}:{target.id}")

    assert not undocumented, (
        "these thresholds are in the code and not in docs/architecture.md's "
        f"table: {undocumented}"
    )


def wi_300_escalation_floor() -> float:
    match = re.search(
        r"top class carries confidence below (\d\.\d+)", WI_300.read_text()
    )
    assert match, "WI-300 no longer states an escalation floor"
    return float(match.group(1))


def test_the_floor_in_the_code_is_the_floor_in_wi_300():
    assert LOW_CONFIDENCE == wi_300_escalation_floor()


def test_the_escalation_threshold_clears_the_floor():
    """WI-300's 0.70 is a floor, and the flow is allowed to be stricter.

    It is not allowed to be looser. Measured on the test split, 0.70 used as
    the operating threshold dismisses eight real defects and puts the line at
    0.767% against QP-110's 0.5% -- see scripts/threshold_sweep.py.
    """
    assert ESCALATE_BELOW >= wi_300_escalation_floor()


def test_wi_300_still_calls_its_number_a_floor():
    """If the document stops saying it, the code is stricter than the rule for
    no stated reason and the next reader is entitled to lower it."""
    assert re.search(r"floor,\s+not the\s+operating threshold", WI_300.read_text())


def test_the_cost_gate_never_drops_below_the_escalation_threshold():
    """``CONFIDENT`` below ``ESCALATE_BELOW`` confirms, unreviewed, regions the
    flow would have handed to a person -- 66 of them at 0.70. It is the one
    thing the sweep says this constant must not do."""
    assert CONFIDENT >= ESCALATE_BELOW


def test_wi_300_reserves_dismissal_to_the_dismissal_threshold():
    """The rule ``ESCALATE_BELOW``'s value implements, stated where rules live.

    Without this clause the equality in flow.py is a coincidence someone
    tidies away.
    """
    text = WI_300.read_text()
    assert re.search(r"Dismissal is reserved to this threshold", text)
    assert re.search(r"confirms a defect or\s+escalates", text)


@pytest.mark.dataset
def test_the_sweep_still_agrees_with_the_shipped_values():
    """The citation is a script, so run it.

    Everything above checks that the table and the code say the same thing. A
    table and a constant can agree perfectly about a number that stopped being
    true when the model was retrained -- which is the failure the
    `retraining-the-reverifier` skill exists to catch, one threshold over. This
    is the same check for the two graph constants: it re-derives them from the
    stored predictions rather than from anything either document claims.
    """
    from threshold_sweep import load_rows, outcome, tally

    rows = load_rows()
    if not rows:
        pytest.skip("the store is empty; run scripts/seed_store.py")

    counts = tally(rows, CONFIDENT, ESCALATE_BELOW)
    assert counts["agent_escapes"] == 0, (
        f"the agent branch dismissed {counts['agent_escapes']} real defects at "
        f"ESCALATE_BELOW={ESCALATE_BELOW}"
    )
    assert counts["agent_dismissed"] == 0, (
        "the agent branch dismissed regions at all, which the routers are set "
        "to make impossible -- the escape count above is then luck, not structure"
    )

    # And the claim that ``CONFIDENT`` is a cost gate: moving it anywhere at or
    # above ``ESCALATE_BELOW`` must not change a single disposition.
    at_floor = [outcome(*row[:3], ESCALATE_BELOW, ESCALATE_BELOW)[1:] for row in rows]
    shipped = [outcome(*row[:3], CONFIDENT, ESCALATE_BELOW)[1:] for row in rows]
    assert at_floor == shipped, (
        "CONFIDENT changed a disposition, so it is not the cost gate "
        "docs/architecture.md says it is"
    )
