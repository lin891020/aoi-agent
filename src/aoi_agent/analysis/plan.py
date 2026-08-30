"""What the model is allowed to ask for, and how that is checked.

The model proposes; this module disposes. Every plan is validated against the
real tool signatures and the real value domains before a single tool runs, and
a plan that fails is shown to the user rather than retried.

Three layers on the plan, and the third is the one that earns its keep. A bad
tool name or a bad argument name would raise on its own. `line_id="L4"` raises
nothing: it is a valid string in a valid parameter that happens to match no
row, so the query succeeds, the series is missing from the chart, and the gap
reads as a finding. That is the same failure the project's no-SQL invariant
exists to prevent, one level up.

There is a fourth, and it runs before any of them, at import: `Registration`
accounts for every parameter each plannable tool exposes, and a tool the
account does not cover cannot be registered at all. Validating arguments says
nothing about the surface they are passed to -- `run_query(sql: str)` passed
every test this module had, because `sql` was a known argument of a known tool
holding a value with no domain to be outside of.
"""

from __future__ import annotations

import re

import ast
import importlib.util
import inspect
import os
import textwrap
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, TypedDict

from sqlalchemy import func, select

from aoi_agent.mcp_servers.production import (
    GROUP_BY,
    SIDES,
    query_board_context,
    query_defect_history,
    query_false_call_rate,
    query_machine_events,
    query_machine_stats,
)
from aoi_agent.mcp_servers.sql_readonly import run_sql
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
    group_by: set[str]
    relative_to: set[str]
    side: set[str]
    max_days: int
    #: The first and last calendar day the store holds, ISO, or None when it
    #: is empty. `date_from`/`date_to` are checked against it the way `days`
    #: is checked against `max_days`: a date outside it returns nothing, and
    #: an empty chart reads as a finding about that day.
    date_span: tuple[str, str] | None


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
    #: `query_false_call_rate`'s axis. The tool owns the vocabulary (it is a
    #: fact about the tool, not the store) and the validator reads it from the
    #: tool's module, so the two cannot drift.
    "group_by": "group_by",
    #: `query_defect_history`'s anchor and which side of it. `relative_to` is
    #: read off the `machine_events` table -- the kinds actually recorded, not
    #: the seeder's tuple -- so an event kind nobody has recorded is refused
    #: before it runs, the same way an unknown machine is. `side` is the
    #: tool's own two-word vocabulary. Both share `query_machine_events`'s
    #: `kind` with `relative_to`, which is the same domain under the name the
    #: tool's signature uses.
    "relative_to": "relative_to",
    "kind": "relative_to",
    "side": "side",
    #: A calendar day, checked for being one and for being inside the span
    #: held. Not a closed vocabulary like the others, so `_domain_errors`
    #: reads it specially; it is in this table because that is the account
    #: the registry keeps of every parameter that can carry text.
    "date_from": "date_span",
    "date_to": "date_span",
}


# ---------------------------------------------------------------------------
# What may be registered
# ---------------------------------------------------------------------------

#: The corpora a free-text parameter may be executed against: a name, and the
#: module that owns the index. `aoi_agent.store.standards` is Chroma over the
#: markdown in `data/standards/` -- no engine, no schema, no query language.
#: The board store is not here and cannot be put here while the module check
#: below stands.
RETRIEVAL_CORPORA: dict[str, str] = {"standards": "aoi_agent.store.standards"}

#: The one way model-written SQL may reach the store: a guard module and the
#: function in it that every declared `sql` parameter has to be passed to,
#: and to nothing else. What the guard does -- a copy of the store without
#: the answer key, one SELECT, a row cap, a time cap -- is held by
#: `tests/test_sql_guard.py`; what this table holds is that the declaration
#: names a guard that exists.
SQL_GUARDS: dict[str, tuple[str, str]] = {
    "production_readonly": ("aoi_agent.analysis.sql_guard", "guarded_select"),
}

#: `AOI_SQL_TOOL=0` leaves `run_sql` out of the registry. That is the control
#: arm of the measurement that decides whether the tool stays: the same
#: question set planned with and without it. Default on, because the store
#: it can reach is a read-only copy with nothing in it that is not already on
#: the page.
SQL_TOOL_ENV = "AOI_SQL_TOOL"


def sql_tool_enabled() -> bool:
    return os.environ.get(SQL_TOOL_ENV, "1").strip().lower() not in ("0", "off", "false", "no")


#: Importing any of these is what makes a module able to run a query language.
#: A module that reaches one of them is not a document corpus, whatever its
#: registration says it is.
STORE_IMPORTS = ("sqlalchemy", "aoi_agent.store.boards", "aoi_agent.store.models")

#: Calls that turn a string into something a database will execute. SQLAlchemy
#: 2.0 refuses a bare string, so `text()` is the door; the DBAPI ones are the
#: door underneath it.
_SQL_FROM_STRING = {"text", "exec_driver_sql", "executescript", "raw_connection"}

#: Where a string becomes syntax rather than a value.
_EXECUTORS = {
    "execute",
    "executemany",
    "executescript",
    "exec_driver_sql",
    "scalar",
    "scalars",
}


#: `*args` and `**kwargs` are not arguments a plan can name or omit. Read as
#: though they were, a catch-all signature rejects every plan that names it
#: twice over: once for an argument it does accept, once for a parameter that
#: does not exist to be passed.
_VARIADIC = (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)


class UnregistrableTool(Exception):
    """A tool whose parameter surface the planner cannot account for."""


@dataclass(frozen=True)
class Registration:
    """One plannable tool, and an account of every parameter it exposes.

    The no-SQL invariant is about a *surface*, not about a call. `validate_plan`
    checks the arguments a plan passes; it has nothing to say about what the
    registry offers to be passed, so `run_query(sql: str)` was refused by none
    of it -- `sql` is a known argument of a known tool holding a value with no
    domain to be outside of. The defence has to sit here, at registration.

    What a signature cannot do is tell a retrieval query from a query language.
    `search_standards(query: str)` and `run_query(sql: str)` are the same
    signature; renaming `sql` to `query` costs nothing, so a rule about
    parameter names or types would ban a working feature and stop nothing. So
    the discrimination is declared, per parameter, and every parameter that
    could carry text has to fall into one of three accounts:

    ``DOMAIN_OF``   a closed vocabulary, checked per call against the store.
    ``identifiers`` a string used as a *value* -- an equality filter or a
                    lookup key -- never parsed. A wrong one returns nothing and
                    says so; it cannot mean something else.
    ``retrieval``   free text, executed against a named document corpus. The
                    declaration names which, and both the tool's module and the
                    corpus module are checked for a route to the store, so free
                    text is admissible only on a path with no query engine on
                    it. The tool's own body is checked for SQL built from a
                    string as well, which is what stops the declaration from
                    being a promise.
    ``sql``         a query language, admissible since 2026-08-29 on one
                    condition: the parameter is passed to the named guard's
                    entry function and to nothing else, checked on the tool's
                    body. The guard is where a wrong query is made harmless;
                    this account is where a tool that sidesteps the guard is
                    refused at import.

    The last two are declarations, and a determined author can write a false
    one. That is the point of putting them here: the lie is a line of source in
    the registry with a reviewer's name on the commit, rather than a signature
    that slid through five green tests.

    What the corpus buys the operator, and the reason this asymmetry is real
    rather than bookkeeping: retrieval hands back passages with their document,
    heading and text, and a wrong passage is visibly the wrong passage. A query
    language hands back a number whose derivation is gone. In a disposition
    context a plausible wrong number is worse than a crash, because it is acted
    on.
    """

    tool: Callable
    #: Parameter names whose string value is used as a value, never as syntax.
    identifiers: frozenset[str] = frozenset()
    #: Parameter name -> the corpus in ``RETRIEVAL_CORPORA`` its text reaches.
    retrieval: Mapping[str, str] = field(default_factory=dict)
    #: Parameter name -> the guard in ``SQL_GUARDS`` that is the only thing
    #: allowed to receive it.
    sql: Mapping[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.tool.__name__


def _admits_text(parameter: inspect.Parameter) -> bool:
    """Could a string be passed here?

    Annotations are strings under `from __future__ import annotations`, so this
    reads them rather than the types. Anything it cannot read counts as text:
    an unreadable annotation is the case to be careful about, not the case to
    wave through.
    """
    annotation = parameter.annotation
    if annotation is inspect.Parameter.empty:
        return True
    if not isinstance(annotation, str):
        annotation = getattr(annotation, "__name__", str(annotation))
    parts = {
        part.strip().strip("'\"").removeprefix("Optional[").removesuffix("]")
        for part in annotation.split("|")
    }
    known = {"int", "float", "bool", "None", "bytes"}
    return not parts <= known


def _sql_from_a_string(registration: Registration) -> list[str]:
    """Places in this tool's own body where a string becomes SQL."""
    try:
        source = textwrap.dedent(inspect.getsource(registration.tool))
    except (OSError, TypeError) as error:  # a lambda, a builtin, a C function
        return [f"{registration.name}: its source cannot be read ({error}), so "
                "nothing here can say whether it builds SQL from a string"]

    tree = ast.parse(source)
    function = next(
        (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if function is None:
        return []
    parameters = {a.arg for a in function.args.args + function.args.kwonlyargs}

    problems = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        called = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(
            node.func, "id", ""
        )
        if called in _SQL_FROM_STRING:
            problems.append(
                f"{registration.name} calls {called}(), which turns a string into "
                "SQL. Typed parameters over a fixed query set, not a query language"
            )
        elif called in _EXECUTORS and node.args:
            argument = node.args[0]
            if isinstance(argument, (ast.Constant, ast.JoinedStr, ast.BinOp)):
                problems.append(
                    f"{registration.name} executes a string it composed, "
                    f"{ast.dump(argument)[:60]}..., rather than a query built "
                    "from typed parameters"
                )
            elif isinstance(argument, ast.Name) and argument.id in parameters:
                problems.append(
                    f"{registration.name} executes its own {argument.id!r} "
                    "parameter. That is text-to-SQL with the model one step back"
                )
    return problems


def _reaches_only_the_guard(registration: Registration, parameter: str, entry: str) -> list[str]:
    """Every use of `parameter` in the tool's body is `entry(parameter)`.

    A declared query-language parameter may be handed to the guard and to
    nothing else -- not printed, not concatenated, not passed to a helper
    that might be the guard under another name. Read off the body, so the
    declaration is checked against what the tool does rather than trusted.
    """
    try:
        source = textwrap.dedent(inspect.getsource(registration.tool))
    except (OSError, TypeError) as error:
        return [f"{registration.name}: its source cannot be read ({error}), so nothing "
                f"can say where {parameter!r} goes"]
    tree = ast.parse(source)
    function = next(
        (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if function is None:
        return []

    guarded: set[int] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            called = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            if called == entry:
                guarded.update(id(a) for a in node.args if isinstance(a, ast.Name))
    problems = []
    for node in ast.walk(function.body[0] if False else function):
        if isinstance(node, ast.Name) and node.id == parameter and isinstance(node.ctx, ast.Load):
            if id(node) not in guarded:
                problems.append(
                    f"{registration.name} uses its {parameter!r} parameter somewhere "
                    f"other than as the argument of {entry}(): a query language may "
                    "reach the guard and nothing else"
                )
    if not guarded:
        problems.append(
            f"{registration.name} never passes {parameter!r} to {entry}(), so the "
            "declaration that it is guarded SQL describes nothing the tool does"
        )
    return problems


@lru_cache(maxsize=None)
def _imports_of(module_name: str) -> frozenset[str]:
    """Every module a module imports, read statically -- nothing is executed."""
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return frozenset()

    imported: set[str] = set()
    for node in ast.walk(ast.parse(Path(spec.origin).read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    return frozenset(imported)


def _reaches_the_store(module_name: str) -> str | None:
    for imported in sorted(_imports_of(module_name)):
        if any(imported.startswith(forbidden) for forbidden in STORE_IMPORTS):
            return imported
    return None


def registration_errors(registrations: tuple[Registration, ...]) -> list[str]:
    """Every reason this registry must not be offered to the planner.

    Injectable, like `audit()` in scripts/invariant_audit.py, so the suite can
    hand it a registry that should be refused and check that it is. A gate that
    has never been shown a rejection is not known to have one.
    """
    errors: list[str] = []
    for registration in registrations:
        name = registration.name
        parameters = inspect.signature(registration.tool).parameters

        for parameter in parameters.values():
            if parameter.kind in _VARIADIC:
                errors.append(
                    f"{name} takes {parameter}, so its parameter surface is open. "
                    "A plannable tool has to state what it accepts"
                )
                continue
            if not _admits_text(parameter):
                continue
            if parameter.name in DOMAIN_OF or parameter.name in registration.identifiers:
                continue
            if parameter.name in registration.retrieval:
                continue
            if parameter.name in registration.sql:
                continue
            errors.append(
                f"{name}({parameter.name}: {parameter.annotation}) can carry "
                "arbitrary text and the registration does not say what that text "
                "is. Give it a domain in DOMAIN_OF, declare it an identifier, or "
                "declare the document corpus it is a retrieval query over"
            )

        for parameter_name, corpus in registration.retrieval.items():
            if parameter_name not in parameters:
                errors.append(f"{name} has no parameter {parameter_name!r} to declare")
                continue
            if corpus not in RETRIEVAL_CORPORA:
                errors.append(
                    f"{name}({parameter_name}) is declared free text over "
                    f"{corpus!r}, which is not a document corpus this system has "
                    f"(known: {', '.join(sorted(RETRIEVAL_CORPORA))})"
                )
                continue
            for module_name, what in (
                (registration.tool.__module__, f"{name}'s module"),
                (RETRIEVAL_CORPORA[corpus], f"the {corpus!r} corpus"),
            ):
                reached = _reaches_the_store(module_name)
                if reached:
                    errors.append(
                        f"{name}({parameter_name}) is declared a retrieval query, "
                        f"but {what} ({module_name}) imports {reached}. Free text "
                        "is only admissible where there is no query engine to "
                        "reach"
                    )

        for parameter_name, guard in registration.sql.items():
            if parameter_name not in parameters:
                errors.append(f"{name} has no parameter {parameter_name!r} to declare")
                continue
            if guard not in SQL_GUARDS:
                errors.append(
                    f"{name}({parameter_name}) is declared a query language over "
                    f"{guard!r}, which is not a guard this system has "
                    f"(known: {', '.join(sorted(SQL_GUARDS))})"
                )
                continue
            module_name, entry = SQL_GUARDS[guard]
            if importlib.util.find_spec(module_name) is None:
                errors.append(f"{name}({parameter_name}): the guard module {module_name} does not exist")
                continue
            errors += _reaches_only_the_guard(registration, parameter_name, entry)

        errors += _sql_from_a_string(registration)

    return errors


#: `classify_defect` is deliberately absent: it loads a torch model onto MPS,
#: and a plan fanning out to ten of them is ten GPU contentions.
REGISTRATIONS: tuple[Registration, ...] = (
    #: `lot_id` is matched against `Board.lot_id` and nothing else.
    Registration(query_defect_history, identifiers=frozenset({"lot_id"})),
    Registration(query_machine_stats),
    Registration(query_false_call_rate),
    Registration(query_machine_events),
    Registration(query_board_context, identifiers=frozenset({"board"})),
    #: The one free-text parameter in the system, and the reason this check is
    #: a declaration rather than a rule about types: it is the same `str` as a
    #: SQL string and it is legitimate.
    Registration(search_standards, retrieval={"query": "standards"}),
    Registration(list_candidates, identifiers=frozenset({"board"})),
    #: The experiment. One SELECT, and the only thing it may be handed to is
    #: the guard; whether it stays is `analysis_eval.py`'s to say.
    *((Registration(run_sql, sql={"sql": "production_readonly"}),) if sql_tool_enabled() else ()),
)

def registry(registrations: tuple[Registration, ...]) -> dict[str, Callable]:
    """The tools a plan may name, or nothing at all.

    Called at import, so a tool the registry cannot account for does not
    become unavailable to the planner -- it stops the module loading, and the
    suite fails on every test that touches the analysis path. There is no
    degraded mode here: a registry that offers a query language is not a
    smaller feature, it is the failure the invariant is about.
    """
    refused = registration_errors(registrations)
    if refused:
        raise UnregistrableTool(
            "the planner's registry offers a surface it cannot account for:\n  "
            + "\n  ".join(refused)
        )
    return {registration.name: registration.tool for registration in registrations}


PLANNABLE_TOOLS: dict[str, Callable] = registry(REGISTRATIONS)


def capability_summary() -> list[str]:
    """What this system can be asked, one line per tool.

    Read off `REGISTRATIONS` at call time rather than written out beside it. A
    hand-kept list of features rots, and it reads as correct the whole time it
    is rotting -- the same argument that made a decision's model identity the
    checkpoint's digest rather than a version string somebody remembers to
    bump. Derived, never declared.

    The sentence for each tool is the first line of its own docstring, which is
    already written for a person: "Compare every machine's rate for one defect
    class." A tool with no docstring gets its name and nothing else, because
    inventing a description for it here is the declared list coming back in.
    """
    lines = []
    for registration in REGISTRATIONS:
        doc = inspect.getdoc(registration.tool) or ""
        first = doc.splitlines()[0].strip() if doc else ""
        lines.append(f"{registration.name} — {first}" if first else registration.name)
    return lines

#: Ways of asking this page what it can be asked. Matched on the question
#: before any model is called, because the answer is the registry and the
#: registry does not need a model to be read. The first question typed at
#: the station was «你能查什麼», and it went to the planner, which refused
#: it as a lookup it could not plan -- restating the question twice and
#: never listing what could be asked. Deliberately narrow: a real question
#: that happens to contain "功能" («M22 的功能異常嗎») must still be planned,
#: so the whole question has to be about the page.
_CAPABILITY_PATTERNS = (
    r"^(你們|你们|你|妳|您|這個系統|这个系统|這一頁|這裡|这里|這頁|这页|這個|这个|這|这|它)?(可以|能|會|会)?(查|問|问|做|回答|處理|处理)(什麼|什么|哪些|啥)",
    r"^(有|你有|這裡有|这里有)?(什麼|什么|哪些|啥)(功能|能力|可以查的|可以問的|可以问的)",
    r"^(what|which)\b.*\b(can|could|do|does)\b.*\b(ask|query|answer|do|look ?up|help)\b",
    r"^(help|capabilities|\?)\s*[?？]*$",
)
_CAPABILITY = tuple(__import__("re").compile(pattern, __import__("re").IGNORECASE) for pattern in _CAPABILITY_PATTERNS)


def is_capability_question(question: str) -> bool:
    """Is this the page being asked what it can be asked?"""
    text = (question or "").strip()
    if not text or len(text) > 40:
        return False
    return any(pattern.search(text) for pattern in _CAPABILITY)


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
        "group_by": set(GROUP_BY),
        "relative_to": set(),
        "side": set(SIDES),
        "max_days": 1,
        "date_span": None,
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
        from aoi_agent.store.events import kinds_present

        kinds = kinds_present()
    except Exception:
        return empty

    span = max(1, (hi - lo).days + 1) if lo and hi else 1
    return {
        "line_id": lines,
        "machine_id": machines,
        "defect_type": empty["defect_type"],
        "defect_class": empty["defect_class"],
        "group_by": empty["group_by"],
        "relative_to": kinds,
        "side": empty["side"],
        "max_days": span,
        "date_span": (lo.date().isoformat(), hi.date().isoformat()) if lo and hi else None,
    }


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


def _date_error(key: str, value: object, span: tuple[str, str] | None, position: int) -> str | None:
    """Why this date must not run, or None."""
    from datetime import date

    try:
        day = date.fromisoformat(str(value))
    except ValueError:
        return f"call {position}: {key}={value!r} is not a date; write it as YYYY-MM-DD"
    if span is None:
        return f"call {position}: {key}={value!r} names a day, and the store holds no days"
    first, last = span
    if not first <= day.isoformat() <= last:
        return (f"call {position}: {key}={value!r} is outside the days held "
                f"({first} to {last}); there is nothing on that day to count")
    return None


_PLACEHOLDER = re.compile(r"<[^<>]{1,40}>|\{[a-z_]{1,40}\}|\.\.\.")


def _domain_errors(name: str, args: dict, domains: Domains, position: int) -> list[str]:
    errors = []
    for key, value in args.items():
        if isinstance(value, str) and _PLACEHOLDER.search(value):
            # `board='<board_id>'` ran on 2026-08-30: the question named no
            # board, the model wrote the slot instead of a value, the tool
            # answered "no board '<board_id>'" and the page rendered a
            # complete-looking answer around a lookup that could not have
            # found anything. A placeholder means the question did not
            # supply the value; that is a refusal, not a lookup.
            errors.append(
                f"call {position}: {key}={value!r} is a placeholder, not a value -- "
                f"the question does not say which {key} is meant"
            )
            continue
        domain_key = DOMAIN_OF.get(key)
        if domain_key == "date_span":
            if value is not None:
                error = _date_error(key, value, domains.get("date_span"), position)
                if error:
                    errors.append(error)
            continue
        if key == "top_n" and value is not None and (not isinstance(value, int) or value < 1):
            errors.append(f"call {position}: top_n={value!r} must be a whole number of at least 1")
            continue
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
