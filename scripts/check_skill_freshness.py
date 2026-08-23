"""Assert that the project skills still describe the code they cite.

A skill in `.claude/skills/` states facts about the codebase -- a constant's
value, a helper's existence, the threshold it tests. When the code moves, the
skill is silently wrong: nothing imports it, nothing type-checks it, and an
agent following it produces a confident, stale answer.

This has already happened once. `measuring-llm-latency` shipped saying
"discard the run when `load_ms > 0`" while `Timing.was_reloaded` was already in
the tree testing `load_ms > 100`, and `RESPONSE_BUDGET_S` cut the call timeout
from 600 s to 10 s an hour later.

The claims below mirror the "What This Skill Reads From the Code" table in each
SKILL.md. Change one, change the other. Parsing is static (`ast`) so that
running this never imports the package or touches the GPU.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module(relpath: str) -> ast.Module:
    return ast.parse((ROOT / relpath).read_text(), filename=relpath)


def _assign(tree: ast.Module, name: str):
    """Value of a module-level `name = <literal>`, or None."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    return None


def _class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    return next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name), None
    )


def _has_member(tree: ast.Module, cls: str, member: str) -> bool:
    node = _class(tree, cls)
    if node is None:
        return False
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and getattr(item.target, "id", None) == member:
            return True
        if isinstance(item, ast.FunctionDef) and item.name == member:
            return True
    return False


def _class_assign(tree: ast.Module, cls: str, name: str):
    """Value of a class-level `name = <literal>`, or None.

    `_has_member` only sees annotated fields and methods, which is right for a
    dataclass's columns but blind to a plain constant hung off the class.
    """
    node = _class(tree, cls)
    if node is None:
        return None
    for item in node.body:
        if isinstance(item, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in item.targets
        ):
            return ast.literal_eval(item.value)
    return None


def _source_of(tree: ast.Module, cls: str, member: str) -> str:
    node = _class(tree, cls)
    for item in node.body if node else []:
        if isinstance(item, ast.FunctionDef) and item.name == member:
            return ast.unparse(item)
    return ""


def check_measuring_llm_latency() -> list[str]:
    """Claims made by .claude/skills/measuring-llm-latency/SKILL.md."""
    rel = "src/aoi_agent/llm/ollama.py"
    tree = _module(rel)
    bad: list[str] = []

    keep_alive = _assign(tree, "KEEP_ALIVE")
    if keep_alive != "30m":
        bad.append(f"{rel}: KEEP_ALIVE is {keep_alive!r}, skill says '30m'")

    deadline = _assign(tree, "EXPLANATION_DEADLINE_S")
    if deadline != 60.0:
        bad.append(
            f"{rel}: EXPLANATION_DEADLINE_S is {deadline!r}, skill says 60.0"
        )

    # The two were one constant until 2026-08-23, and the skill's whole third
    # section is about their being two. A ``RESPONSE_BUDGET_S`` back in the
    # client module is that merge happening again.
    if _assign(tree, "RESPONSE_BUDGET_S") is not None:
        bad.append(
            f"{rel}: RESPONSE_BUDGET_S is back in the client module -- the skill "
            "says it lives on the disposition path in graph/flow.py, and the "
            "defect it names is exactly the client timeout wearing that name"
        )

    flow_rel = "src/aoi_agent/graph/flow.py"
    budget = _assign(_module(flow_rel), "RESPONSE_BUDGET_S")
    if budget != 10.0:
        bad.append(f"{flow_rel}: RESPONSE_BUDGET_S is {budget!r}, skill says 10.0")

    for member in ("wall_ms", "load_ms", "prompt_eval_ms", "eval_ms", "was_reloaded"):
        if not _has_member(tree, "Timing", member):
            bad.append(f"{rel}: Timing.{member} is gone, skill still cites it")

    reloaded = _source_of(tree, "Timing", "was_reloaded")
    if reloaded and "self.load_ms > self.RELOAD_MS" not in reloaded:
        bad.append(
            f"{rel}: Timing.was_reloaded no longer tests `load_ms > RELOAD_MS`, "
            "skill states that threshold"
        )

    reload_ms = _class_assign(tree, "Timing", "RELOAD_MS")
    if reload_ms != 2000.0:
        bad.append(f"{rel}: Timing.RELOAD_MS is {reload_ms!r}, skill says 2000.0")

    # The reasoning token trap turns on the reason node actually thinking. If
    # the flow stops passing `think`, the skill's central warning describes a
    # problem the code no longer has, and the latency method it prescribes is
    # heavier than it needs to be.
    flow = _module("src/aoi_agent/graph/flow.py")
    if 'think="low"' not in (ROOT / "src/aoi_agent/graph/flow.py").read_text():
        bad.append(
            "src/aoi_agent/graph/flow.py: the reason node no longer requests "
            "thinking, so the skill's reasoning token trap no longer applies"
        )
    del flow

    for member in ("warm_up", "resident_models"):
        if not _has_member(tree, "OllamaClient", member):
            bad.append(f"{rel}: OllamaClient.{member} is gone, skill still cites it")

    return bad


def check_retraining_the_reverifier() -> list[str]:
    """Claims made by .claude/skills/retraining-the-reverifier/SKILL.md."""
    bad: list[str] = []

    rel = "src/aoi_agent/vision/inference.py"
    tree = _module(rel)
    if _assign(tree, "DEFAULT_DISMISS_THRESHOLD") is None:
        bad.append(f"{rel}: DEFAULT_DISMISS_THRESHOLD is gone; the threshold gate names it")

    tree = _module("scripts/train.py")
    if not any(
        isinstance(n, ast.FunctionDef) and n.name == "split_by_image" for n in tree.body
    ):
        bad.append(
            "scripts/train.py: split_by_image is gone; the skill cites it as the "
            "enforcement point for the split-by-image invariant"
        )

    for rel in (
        "scripts/build_patches.py",
        "scripts/report.py",
        "scripts/routing_report.py",
        "scripts/gate_check.py",
    ):
        if not (ROOT / rel).exists():
            bad.append(f"{rel}: missing; the retraining order names it")

    return bad


CHECKS = {
    "measuring-llm-latency": check_measuring_llm_latency,
    "retraining-the-reverifier": check_retraining_the_reverifier,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", choices=sorted(CHECKS), default=None)
    args = parser.parse_args()

    names = [args.skill] if args.skill else sorted(CHECKS)
    findings = 0
    for name in names:
        bad = CHECKS[name]()
        findings += len(bad)
        if bad:
            print(f"STALE  {name}")
            for line in bad:
                print(f"    {line}")
        else:
            print(f"ok     {name}")

    if findings:
        print(f"\n{findings} stale claim(s). Update the SKILL.md, not this script.")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
