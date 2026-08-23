"""What the model is allowed to ask for, and how that is checked.

The model proposes; this module disposes. Every plan is validated against the
real tool signatures and the real value domains before a single tool runs, and
a plan that fails is shown to the user rather than retried.

Three layers, and the third is the one that earns its keep. A bad tool name or
a bad argument name would raise on its own. `line_id="L4"` raises nothing: it is
a valid string in a valid parameter that happens to match no row, so the query
succeeds, the series is missing from the chart, and the gap reads as a finding.
That is the same failure the project's no-SQL invariant exists to prevent, one
level up.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, TypedDict

from sqlalchemy import func, select

from aoi_agent.mcp_servers.production import (
    query_board_context,
    query_defect_history,
    query_machine_stats,
)
from aoi_agent.mcp_servers.standards import search_standards
from aoi_agent.mcp_servers.classify import list_candidates
from aoi_agent.store.standards import SCOPES


class ToolCall(TypedDict):
    tool: str
    args: dict[str, Any]
    why: str


class Plan(TypedDict):
    interpretation: str
    assumptions: list[str]
    calls: list[ToolCall]


class Domains(TypedDict):
    line_id: set[str]
    machine_id: set[str]
    defect_type: set[str]
    defect_class: set[str]
    max_days: int


#: `classify_defect` is deliberately absent: it loads a torch model onto MPS,
#: and a plan fanning out to ten of them is ten GPU contentions.
PLANNABLE_TOOLS: dict[str, Callable] = {
    "query_defect_history": query_defect_history,
    "query_machine_stats": query_machine_stats,
    "query_board_context": query_board_context,
    "search_standards": search_standards,
    "list_candidates": list_candidates,
}

#: Arguments whose values are checked against a domain rather than a type.
DOMAIN_OF = {
    "line_id": "line_id",
    "machine_id": "machine_id",
    "defect_type": "defect_type",
    #: `search_standards`'s scope, and a wider set than `defect_type`: the
    #: criteria are asked about a class the *classifier* emitted, which
    #: includes `false_call`, whereas `defect_type` filters production records
    #: over the six real classes. Same vocabulary plus one, different question,
    #: so they are two domains rather than one shared name.
    "defect_class": "defect_class",
}

PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "interpretation": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": sorted(PLANNABLE_TOOLS)},
                    "args": {"type": "object"},
                    "why": {"type": "string"},
                },
                "required": ["tool", "args", "why"],
            },
        },
    },
    "required": ["interpretation", "assumptions", "calls"],
}


def store_domains() -> Domains:
    """The values that actually exist, read from the store.

    Hard-coding these would let the store and the validator drift, and the
    drift would show up as a plan rejected for naming a machine that exists.

    Degrades to empty domains rather than raising when the store is missing or
    unseeded -- a validator that cannot reach its store should reject every
    plan on domain grounds, not crash the caller.
    """
    empty: Domains = {
        "line_id": set(),
        "machine_id": set(),
        "defect_type": {"open", "short", "mousebite", "spur", "copper", "pin-hole"},
        "defect_class": set(SCOPES),
        "max_days": 1,
    }

    try:
        from aoi_agent.store.boards import session_factory
        from aoi_agent.store.models import Board

        with session_factory()() as session:
            lines = set(session.execute(select(Board.line_id).distinct()).scalars())
            machines = set(session.execute(select(Board.machine_id).distinct()).scalars())
            lo, hi = session.execute(
                select(func.min(Board.inspected_at), func.max(Board.inspected_at))
            ).first()
    except Exception:
        return empty

    span = max(1, (hi - lo).days + 1) if lo and hi else 1
    return {
        "line_id": lines,
        "machine_id": machines,
        "defect_type": empty["defect_type"],
        "defect_class": empty["defect_class"],
        "max_days": span,
    }


#: `*args` and `**kwargs` are not arguments a plan can name or omit. Read as
#: though they were, a catch-all signature rejects every plan that names it
#: twice over: once for an argument it does accept, once for a parameter that
#: does not exist to be passed.
_VARIADIC = (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)


def _signature_errors(name: str, args: dict, position: int) -> list[str]:
    parameters = inspect.signature(PLANNABLE_TOOLS[name]).parameters
    named = {
        key: parameter
        for key, parameter in parameters.items()
        if parameter.kind not in _VARIADIC
    }
    takes_anything = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    errors = []

    if not takes_anything:
        for given in args:
            if given not in named:
                errors.append(
                    f"call {position}: {name} has no argument {given!r} "
                    f"(it takes {', '.join(named)})"
                )

    for required, parameter in named.items():
        if parameter.default is inspect.Parameter.empty and required not in args:
            errors.append(f"call {position}: {name} requires {required!r}")

    return errors


def _domain_errors(name: str, args: dict, domains: Domains, position: int) -> list[str]:
    errors = []
    for key, value in args.items():
        domain_key = DOMAIN_OF.get(key)
        if domain_key and value is not None and value not in domains[domain_key]:
            allowed = ", ".join(sorted(domains[domain_key]))
            errors.append(
                f"call {position}: {key}={value!r} does not exist "
                f"(known values: {allowed})"
            )
        if key == "days" and isinstance(value, int) and value > domains["max_days"]:
            errors.append(
                f"call {position}: days={value} exceeds the {domains['max_days']} "
                f"days of data held, which would silently return the whole span"
            )
    return errors


def validate_plan(plan: dict, domains: Domains) -> list[str]:
    """Every reason this plan must not run. Empty means it may.

    All errors are collected rather than raised on the first, because the plan
    is shown to a person: fixing one problem per model round trip is worse than
    seeing the whole list at once.
    """
    calls = plan.get("calls") or []
    if not calls:
        errors = ["the plan has no calls, so it cannot answer anything"]
        return errors

    errors: list[str] = []
    for position, call in enumerate(calls, 1):
        name = call.get("tool")
        if name not in PLANNABLE_TOOLS:
            errors.append(
                f"call {position}: {name!r} is not a tool this system exposes "
                f"(available: {', '.join(sorted(PLANNABLE_TOOLS))})"
            )
            continue

        args = call.get("args") or {}
        errors += _signature_errors(name, args, position)
        errors += _domain_errors(name, args, domains, position)

    return errors
