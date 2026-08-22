# Natural-language analysis interface — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a shift supervisor ask a production question in plain language and get a chart, a written answer, and a visible account of how it was obtained.

**Architecture:** A second LangGraph flow, separate from the disposition flow. One model call produces a typed plan; validation runs before any tool does; `Send` expands the plan into N parallel branches over the existing MCP tools; an `operator.add` reducer accumulates their results; deterministic code builds the chart spec and a second model call writes the prose. No checkpointer — nothing suspends. Every run is persisted so its chart can be redrawn without re-running the model.

**Tech Stack:** LangGraph 1.2 (`Send`, `Annotated` reducers), SQLAlchemy 2.0, FastAPI + Jinja2, Ollama (`gpt-oss:20b`), pytest.

**Spec:** `docs/superpowers/specs/2026-08-22-analysis-interface-design.md`

**Verified before writing this plan** (do not re-derive):
- `Send(node, payload)` hands the payload to the node as its entire state; the
  node's return merges into the parent through the reducer.
- Three 0.2s branches complete in 0.21s, not 0.6s. Four real tools against the
  live store: 183ms parallel, 462ms sequential, no threading errors from
  concurrent SQLAlchemy sessions.
- Two parallel writes to a channel with no reducer raise `InvalidUpdateError`
  rather than racing; with `Annotated[list, operator.add]` both land.
- `stream_mode="updates"` yields one update per `Send` branch, and a node
  returning an empty dict streams as `None`.

## Global Constraints

- **No free-form SQL, ever.** Tools are called by name from a fixed registry with typed parameters. Copied from the project invariant in `CLAUDE.md`.
- **`classify_defect` is not plannable.** It loads a torch model onto MPS; a plan fanning out to ten of them is ten GPU contentions.
- **Validation runs before any tool executes.** A plan that fails validation is shown to the user with its errors and nothing runs. No retry.
- **The model explains; it does not decide.** Chart type is derived from result shape. This mirrors `route_after_reason` in the disposition flow, which was changed for measured reasons — see `docs/benchmarks.md`.
- **Ground truth is never read outside evaluation scripts.**
- **Measured, before you claim it:** four tools run in 183ms parallel against 462ms sequential, while the two model calls cost ~25s. The fan-out is the correct structure for independent work; it is not a latency optimisation, and the plan must not describe it as one.
- Python 3.12. `uv run pytest` must stay green at every commit.

---

### Task 1: Plan types, tool registry, and validation

Pure functions. No model, no database, no graph. This is the task that carries the project's central invariant, so it comes first and is tested hardest.

**Files:**
- Create: `src/aoi_agent/analysis/__init__.py`
- Create: `src/aoi_agent/analysis/plan.py`
- Test: `tests/test_analysis_plan.py`

**Interfaces:**
- Consumes: the five MCP tool functions from `aoi_agent.mcp_servers.*`
- Produces:
  - `ToolCall = TypedDict("ToolCall", {"tool": str, "args": dict, "why": str})`
  - `Plan = TypedDict("Plan", {"interpretation": str, "assumptions": list[str], "calls": list[ToolCall]})`
  - `PLANNABLE_TOOLS: dict[str, Callable]`
  - `Domains = TypedDict(...)` with keys `line_id`, `machine_id`, `defect_type`, `max_days`
  - `validate_plan(plan: dict, domains: Domains) -> list[str]` — empty list means valid
  - `store_domains() -> Domains`
  - `PLAN_SCHEMA: dict` — the JSON schema handed to the model

- [ ] **Step 1: Write the failing tests**

```python
"""Validating a plan before anything runs.

The layer that matters is the third. A tool name that does not exist raises;
an argument name that does not exist raises; but `line_id="L4"` raises nothing
at all -- it returns an empty result, the chart comes back with one fewer line,
and nobody notices. That is the failure this whole file exists to prevent, and
it is the same argument as the project's no-SQL invariant.
"""

from __future__ import annotations

import pytest

from aoi_agent.analysis.plan import (
    PLANNABLE_TOOLS,
    store_domains,
    validate_plan,
)

DOMAINS = {
    "line_id": {"L1", "L2", "L3"},
    "machine_id": {"M11", "M12", "M21", "M22", "M31", "M32"},
    "defect_type": {"open", "short", "mousebite", "spur", "copper", "pin-hole"},
    "max_days": 9,
}


def plan(*calls) -> dict:
    return {
        "interpretation": "how the question was read",
        "assumptions": ["compared against the fleet average"],
        "calls": [
            {"tool": t, "args": a, "why": "because"} for t, a in calls
        ],
    }


def test_a_well_formed_plan_validates():
    errors = validate_plan(
        plan(("query_machine_stats", {"defect_type": "open", "days": 7})), DOMAINS
    )
    assert errors == []


def test_an_unknown_tool_is_rejected():
    errors = validate_plan(plan(("drop_tables", {})), DOMAINS)
    assert len(errors) == 1
    assert "drop_tables" in errors[0]


def test_classify_defect_is_not_plannable():
    """It loads a torch model onto MPS. Ten of them in one fan-out is ten
    contentions, so it is kept out of the registry rather than rate-limited."""
    assert "classify_defect" not in PLANNABLE_TOOLS
    errors = validate_plan(plan(("classify_defect", {"candidate_ref": "1#0"})), DOMAINS)
    assert errors


def test_an_unknown_argument_name_is_rejected():
    errors = validate_plan(
        plan(("query_machine_stats", {"defect_type": "open", "weeks": 3})), DOMAINS
    )
    assert any("weeks" in e for e in errors)


def test_a_missing_required_argument_is_rejected():
    errors = validate_plan(plan(("query_machine_stats", {"days": 7})), DOMAINS)
    assert any("defect_type" in e for e in errors)


def test_an_optional_argument_may_be_omitted():
    errors = validate_plan(plan(("query_defect_history", {"days": 7})), DOMAINS)
    assert errors == []


@pytest.mark.parametrize(
    "args,bad",
    [
        ({"defect_type": "open", "days": 7, "line_id": "L4"}, "L4"),
        ({"defect_type": "open", "days": 7, "machine_id": "M99"}, "M99"),
        ({"defect_type": "scratch", "days": 7}, "scratch"),
    ],
)
def test_a_legal_looking_value_outside_its_domain_is_rejected(args, bad):
    """The quiet failure. `line_id="L4"` is a valid string and a valid
    parameter; it simply matches nothing, and a chart with a missing series
    reads as a finding rather than as a bug."""
    errors = validate_plan(plan(("query_defect_history", args)), DOMAINS)
    assert any(bad in e for e in errors)


def test_a_window_longer_than_the_data_is_rejected():
    """Asking for 30 days of an 8-day store does not error, it silently
    returns the same 8 days -- so a month-on-month comparison would report the
    two windows as identical."""
    errors = validate_plan(
        plan(("query_defect_history", {"days": 30})), DOMAINS
    )
    assert any("30" in e for e in errors)


def test_every_error_is_reported_not_just_the_first():
    """The plan is shown to the user. Fixing one error at a time across
    several model round trips is worse than seeing them all at once."""
    errors = validate_plan(
        plan(
            ("nope", {}),
            ("query_defect_history", {"line_id": "L9", "days": 999}),
        ),
        DOMAINS,
    )
    assert len(errors) >= 3


def test_an_empty_plan_is_rejected():
    errors = validate_plan(plan(), DOMAINS)
    assert any("no calls" in e.lower() for e in errors)


@pytest.mark.dataset
def test_store_domains_reads_the_real_store():
    domains = store_domains()
    assert domains["line_id"] == {"L1", "L2", "L3"}
    assert len(domains["machine_id"]) == 6
    assert 1 <= domains["max_days"] <= 400
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_analysis_plan.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi_agent.analysis'`

- [ ] **Step 3: Write the implementation**

Create `src/aoi_agent/analysis/__init__.py` as an empty file, then `src/aoi_agent/analysis/plan.py`:

```python
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
    """
    from aoi_agent.store.boards import session_factory
    from aoi_agent.store.models import Board

    with session_factory()() as session:
        lines = set(session.execute(select(Board.line_id).distinct()).scalars())
        machines = set(session.execute(select(Board.machine_id).distinct()).scalars())
        lo, hi = session.execute(
            select(func.min(Board.inspected_at), func.max(Board.inspected_at))
        ).first()

    span = max(1, (hi - lo).days + 1) if lo and hi else 1
    return {
        "line_id": lines,
        "machine_id": machines,
        "defect_type": {"open", "short", "mousebite", "spur", "copper", "pin-hole"},
        "max_days": span,
    }


def _signature_errors(name: str, args: dict, position: int) -> list[str]:
    parameters = inspect.signature(PLANNABLE_TOOLS[name]).parameters
    errors = []

    for given in args:
        if given not in parameters:
            errors.append(
                f"call {position}: {name} has no argument {given!r} "
                f"(it takes {', '.join(parameters)})"
            )

    for required, parameter in parameters.items():
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_analysis_plan.py -q`
Expected: PASS, 12 tests (11 plus the dataset-marked one)

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS, no regression in the existing 94

- [ ] **Step 6: Commit**

```bash
git add src/aoi_agent/analysis tests/test_analysis_plan.py
git commit -m "Validate an analysis plan before any of it runs

Three layers: the tool exists, its arguments exist, and their values do. The
third is the one that matters. A bad tool name raises and a bad argument name
raises, but line_id=\"L4\" is a valid string in a valid parameter that matches
no row -- the query succeeds, the chart comes back with a series missing, and
the gap reads as a finding rather than as a bug. Same argument as the no-SQL
invariant, one level up.

Errors are collected rather than raised on the first, because the plan is shown
to a person and fixing one problem per model round trip is worse than seeing
the list. classify_defect is kept out of the registry: it loads a torch model
onto MPS and a fan-out of ten is ten contentions."
```

---

### Task 2: Running one call, where failure is data

A branch that raises would abort the fan-out and lose the three siblings that
succeeded. The branches are independent by construction, so a failure becomes a
result with `ok=False` and the join reports what is missing.

**Files:**
- Create: `src/aoi_agent/analysis/tools.py`
- Test: `tests/test_analysis_tools.py`

**Interfaces:**
- Consumes: `PLANNABLE_TOOLS`, `ToolCall` from `aoi_agent.analysis.plan`
- Produces:
  - `ToolResult = TypedDict("ToolResult", {"tool": str, "args": dict, "ok": bool, "data": dict | None, "error": str | None, "elapsed_ms": float})`
  - `run_call(call: ToolCall) -> ToolResult`

- [ ] **Step 1: Write the failing tests**

```python
"""One tool call, and what happens when it goes wrong.

The fan-out's branches are independent, so one of them failing must not take
the others with it. A raised exception would do exactly that, so `run_call`
catches its own failures and returns them as data.
"""

from __future__ import annotations

import pytest

from aoi_agent.analysis import tools


def call(tool: str, **args) -> dict:
    return {"tool": tool, "args": args, "why": "because"}


def test_a_successful_call_carries_its_data_and_timing(monkeypatch):
    monkeypatch.setitem(
        tools.PLANNABLE_TOOLS, "query_machine_stats", lambda **kw: {"machines": []}
    )
    result = tools.run_call(call("query_machine_stats", defect_type="open", days=7))

    assert result["ok"] is True
    assert result["data"] == {"machines": []}
    assert result["error"] is None
    assert result["elapsed_ms"] >= 0
    assert result["tool"] == "query_machine_stats"
    assert result["args"] == {"defect_type": "open", "days": 7}


def test_a_raising_tool_becomes_a_failed_result_not_an_exception():
    """The whole point. One branch dying must not abort its siblings."""
    def boom(**kw):
        raise RuntimeError("the store is unreachable")

    tools.PLANNABLE_TOOLS["_boom"] = boom
    try:
        result = tools.run_call(call("_boom"))
    finally:
        del tools.PLANNABLE_TOOLS["_boom"]

    assert result["ok"] is False
    assert result["data"] is None
    assert "unreachable" in result["error"]
    assert "RuntimeError" in result["error"]


def test_a_tool_that_is_not_in_the_registry_fails_closed():
    """Validation should have caught this. If it did not, the call must still
    not reach `getattr` on something arbitrary."""
    result = tools.run_call(call("drop_tables"))

    assert result["ok"] is False
    assert "drop_tables" in result["error"]


def test_the_elapsed_time_is_recorded_even_on_failure():
    """A slow failure and a fast one are different operational problems."""
    def boom(**kw):
        raise ValueError("no")

    tools.PLANNABLE_TOOLS["_boom"] = boom
    try:
        result = tools.run_call(call("_boom"))
    finally:
        del tools.PLANNABLE_TOOLS["_boom"]

    assert result["elapsed_ms"] >= 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_analysis_tools.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi_agent.analysis.tools'`

- [ ] **Step 3: Write the implementation**

```python
"""Calling one tool, safely.

A fan-out branch that raises takes the whole superstep with it, and the three
siblings that succeeded are lost along with it. The branches are independent by
construction; letting one abort the others would be a defect, not caution. So a
failure here is a value: `ok=False` with the reason, which the join reports and
the answer names.
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

from aoi_agent.analysis.plan import PLANNABLE_TOOLS, ToolCall


class ToolResult(TypedDict):
    tool: str
    args: dict[str, Any]
    ok: bool
    data: dict | None
    error: str | None
    elapsed_ms: float


def run_call(call: ToolCall) -> ToolResult:
    """Run one planned call and return its outcome, successful or not."""
    name = call.get("tool", "")
    args = call.get("args") or {}
    started = time.perf_counter()

    def finish(ok: bool, data: dict | None, error: str | None) -> ToolResult:
        return {
            "tool": name,
            "args": args,
            "ok": ok,
            "data": data,
            "error": error,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    function = PLANNABLE_TOOLS.get(name)
    if function is None:
        # Validation should have caught this. Failing closed anyway means a
        # gap in validation costs an error message rather than an arbitrary
        # call.
        return finish(False, None, f"{name!r} is not a tool this system exposes")

    try:
        return finish(True, function(**args), None)
    except Exception as error:  # noqa: BLE001 -- the branch must not raise
        return finish(False, None, f"{type(error).__name__}: {error}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_analysis_tools.py -q`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add src/aoi_agent/analysis/tools.py tests/test_analysis_tools.py
git commit -m "Make a failed tool call data rather than an exception

A raising branch takes its whole superstep down and loses the siblings that
succeeded with it. The branches are independent by construction, so run_call
catches its own failures and returns ok=False with the reason. The join then
reports three succeeded and one did not, and the answer says which."
```

---

### Task 3: Chart specs derived from result shape

The model does not choose the chart. Same principle as `decide_node` in the
disposition flow, and for the same measured reason.

**Files:**
- Create: `src/aoi_agent/analysis/charts.py`
- Test: `tests/test_analysis_charts.py`

**Interfaces:**
- Consumes: `ToolResult` from `aoi_agent.analysis.tools`
- Produces: `chart_spec_for(results: list[ToolResult]) -> dict | None` returning
  `{"kind": "bar" | "line" | "none", "title": str, "x_label": str, "y_label": str, "series": [{"name": str, "points": [{"x": str, "y": float}]}]}`

- [ ] **Step 1: Write the failing tests**

```python
"""Choosing a chart from the shape of the data, not from the model's opinion.

`chart_spec` is data rather than an image: the page renders it, which is what
lets a stored answer be redrawn months later without re-running a model that
would not produce the same plan twice.
"""

from __future__ import annotations

from aoi_agent.analysis.charts import chart_spec_for


def ok(tool: str, data: dict) -> dict:
    return {"tool": tool, "args": {}, "ok": True, "data": data,
            "error": None, "elapsed_ms": 1.0}


MACHINE_STATS = {
    "defect_type": "open",
    "fleet_share_of_defects": 0.225,
    "machines": [
        {"machine": "L2-M22", "share_of_defects": 0.321, "per_board": 2.3},
        {"machine": "L1-M11", "share_of_defects": 0.190, "per_board": 1.4},
    ],
}


def test_a_comparison_across_entities_becomes_bars():
    spec = chart_spec_for([ok("query_machine_stats", MACHINE_STATS)])

    assert spec["kind"] == "bar"
    assert spec["series"][0]["points"][0]["x"] == "L2-M22"
    assert spec["series"][0]["points"][0]["y"] == 0.321


def test_the_fleet_average_is_carried_as_its_own_series():
    """A bar chart of machine shares with no baseline invites the reader to
    compare machines against each other and miss that all of them are high."""
    spec = chart_spec_for([ok("query_machine_stats", MACHINE_STATS)])
    names = [s["name"] for s in spec["series"]]

    assert any("fleet" in n.lower() for n in names)


def test_a_defect_breakdown_becomes_bars():
    spec = chart_spec_for(
        [ok("query_defect_history", {"counts": {"open": 12, "short": 3}})]
    )

    assert spec["kind"] == "bar"
    assert {p["x"] for p in spec["series"][0]["points"]} == {"open", "short"}


def test_results_with_nothing_plottable_produce_no_chart():
    """Retrieved criteria are prose. Forcing a chart onto them would be
    decoration, and a chart that means nothing is worse than none."""
    spec = chart_spec_for(
        [ok("search_standards", {"passages": [{"document": "WI-201", "text": "..."}]})]
    )

    assert spec is None


def test_failed_results_are_skipped_rather_than_plotted_as_zero():
    """A missing bar and a zero bar read very differently, and only one of them
    is true."""
    failed = {"tool": "query_machine_stats", "args": {}, "ok": False,
              "data": None, "error": "boom", "elapsed_ms": 1.0}
    spec = chart_spec_for([failed])

    assert spec is None


def test_the_first_plottable_result_wins_when_several_are_present():
    """One question, one chart. Stacking several unrelated charts under one
    answer makes the reader do the joining."""
    spec = chart_spec_for(
        [
            ok("search_standards", {"passages": []}),
            ok("query_machine_stats", MACHINE_STATS),
            ok("query_defect_history", {"counts": {"open": 1}}),
        ]
    )

    assert spec["kind"] == "bar"
    assert spec["series"][0]["points"][0]["x"] == "L2-M22"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_analysis_charts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi_agent.analysis.charts'`

- [ ] **Step 3: Write the implementation**

```python
"""Turning results into a chart specification.

The model does not pick the chart. It picks nothing in this system: measured on
the disposition path, its judgement lost to the classifier's, and there is no
reason to expect it to do better at choosing an axis. The shape of the data
decides.

What comes out is a specification, not an image -- axis labels, series, points.
The page renders it. That is what lets a run stored today be redrawn next
quarter without re-running a plan that would not regenerate identically.
"""

from __future__ import annotations

from typing import Any

from aoi_agent.analysis.tools import ToolResult


def _machine_comparison(data: dict) -> dict | None:
    machines = data.get("machines")
    if not machines:
        return None

    defect = data.get("defect_type", "defects")
    series: list[dict[str, Any]] = [
        {
            "name": f"share of {defect}",
            "points": [
                {"x": m["machine"], "y": round(m["share_of_defects"], 4)}
                for m in machines
            ],
        }
    ]

    fleet = data.get("fleet_share_of_defects")
    if fleet is not None:
        # Carried as a series rather than folded into the bars: without it the
        # reader compares machines against each other and cannot see that every
        # one of them sits above the fleet.
        series.append(
            {
                "name": "fleet average",
                "points": [
                    {"x": m["machine"], "y": round(fleet, 4)} for m in machines
                ],
            }
        )

    return {
        "kind": "bar",
        "title": f"{defect} share by machine",
        "x_label": "machine",
        "y_label": "share of that machine's defects",
        "series": series,
    }


def _defect_breakdown(data: dict) -> dict | None:
    counts = data.get("counts")
    if not counts:
        return None
    return {
        "kind": "bar",
        "title": "defects by class",
        "x_label": "class",
        "y_label": "count",
        "series": [
            {
                "name": "count",
                "points": [
                    {"x": name, "y": float(value)}
                    for name, value in sorted(counts.items(), key=lambda kv: -kv[1])
                ],
            }
        ],
    }


#: Tried in order. The first result that yields a spec is the chart, because
#: one question gets one chart -- stacking several unrelated ones under a single
#: answer leaves the reader to do the joining.
BUILDERS = {
    "query_machine_stats": _machine_comparison,
    "query_defect_history": _defect_breakdown,
}


def chart_spec_for(results: list[ToolResult]) -> dict | None:
    """The chart for these results, or None when nothing is plottable.

    Failed results are skipped rather than plotted as zero: a missing bar and a
    zero bar read very differently and only one of them is true.
    """
    for result in results:
        if not result.get("ok") or not result.get("data"):
            continue
        builder = BUILDERS.get(result["tool"])
        if builder is None:
            continue
        spec = builder(result["data"])
        if spec is not None:
            return spec
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_analysis_charts.py -q`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add src/aoi_agent/analysis/charts.py tests/test_analysis_charts.py
git commit -m "Derive the chart from the data's shape, not the model's opinion

Measured on the disposition path, the model's judgement lost to the
classifier's; nothing suggests it would do better at choosing an axis. The
shape of the result decides, and what comes out is a specification rather than
an image so a run stored today can be redrawn without re-running a plan that
would not regenerate identically.

The fleet average rides as its own series. Without it a reader compares the
machines against each other and cannot see that all of them are above the
fleet. Failed results are skipped rather than drawn as zero, because a missing
bar and a zero bar read very differently and only one is true."
```

---

### Task 4: The prompt, its five boundary examples, and the planning node

**Files:**
- Create: `src/aoi_agent/analysis/prompts.py`
- Test: `tests/test_analysis_prompts.py`

**Interfaces:**
- Consumes: `PLAN_SCHEMA`, `PLANNABLE_TOOLS`, `Domains` from `aoi_agent.analysis.plan`
- Produces:
  - `SYSTEM_PROMPT: str`
  - `FEW_SHOT: list[dict]` — five `{"question": str, "plan": dict}` entries
  - `build_planning_messages(question: str, domains: Domains) -> list[dict]`
  - `SYNTHESIS_PROMPT: str`
  - `build_synthesis_messages(question, plan, results) -> list[dict]`

- [ ] **Step 1: Write the failing tests**

```python
"""The prompt, and the five examples that shape what comes back.

Diversity beats count: five examples that show the edges teach more than five
that show the happy path. Four of these five are refusals or hedges, which is
deliberate -- a system that answers everything is more dangerous on a factory
floor than one that says it cannot.
"""

from __future__ import annotations

from aoi_agent.analysis.plan import PLANNABLE_TOOLS, validate_plan
from aoi_agent.analysis.prompts import (
    FEW_SHOT,
    build_planning_messages,
    build_synthesis_messages,
)

DOMAINS = {
    "line_id": {"L1", "L2", "L3"},
    "machine_id": {"M11", "M12", "M21", "M22", "M31", "M32"},
    "defect_type": {"open", "short", "mousebite", "spur", "copper", "pin-hole"},
    "max_days": 9,
}


def test_there_are_five_examples():
    assert len(FEW_SHOT) == 5


def test_every_example_plan_would_pass_validation():
    """An example that the validator rejects teaches the model to produce
    plans the validator rejects."""
    for example in FEW_SHOT:
        plan = example["plan"]
        if not plan["calls"]:
            continue  # a refusal example; nothing to validate
        assert validate_plan(plan, DOMAINS) == [], example["question"]


def test_the_examples_cover_the_five_shapes():
    shapes = {example["shape"] for example in FEW_SHOT}
    assert shapes == {
        "cross_tool",
        "unstated_baseline",
        "causal",
        "out_of_range",
        "too_vague",
    }


def test_refusals_are_expressed_as_an_empty_plan_with_a_reason():
    """A refusal is not an error. It is a plan with no calls and an
    interpretation that says why, so it renders through the same path."""
    refusals = [e for e in FEW_SHOT if e["shape"] in ("out_of_range", "too_vague")]
    assert refusals
    for example in refusals:
        assert example["plan"]["calls"] == []
        assert example["plan"]["interpretation"]


def test_the_baseline_example_states_its_baseline_in_assumptions():
    example = next(e for e in FEW_SHOT if e["shape"] == "unstated_baseline")
    assert example["plan"]["assumptions"]


def test_the_planning_messages_carry_the_domains_and_the_examples():
    messages = build_planning_messages("L2 的 open 正常嗎", DOMAINS)
    blob = "\n".join(m["content"] for m in messages)

    assert "L2" in blob
    assert "9" in blob, "the data span has to be in the prompt to be respected"
    for name in PLANNABLE_TOOLS:
        assert name in blob
    assert FEW_SHOT[0]["question"] in blob
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].endswith("L2 的 open 正常嗎")


def test_the_synthesis_prompt_carries_failures_so_the_answer_can_name_them():
    results = [
        {"tool": "query_machine_stats", "args": {}, "ok": True,
         "data": {"machines": []}, "error": None, "elapsed_ms": 1.0},
        {"tool": "search_standards", "args": {}, "ok": False,
         "data": None, "error": "TimeoutError: index unreachable",
         "elapsed_ms": 2.0},
    ]
    messages = build_synthesis_messages(
        "q", {"interpretation": "i", "assumptions": ["a"], "calls": []}, results
    )
    blob = "\n".join(m["content"] for m in messages)

    assert "index unreachable" in blob
    assert "search_standards" in blob


def test_the_synthesis_prompt_forbids_inventing_causes():
    """The tools carry no causal data. A plausible causal story from
    correlation is the failure mode that gets a machine stopped."""
    messages = build_synthesis_messages("why is M22 bad", {"interpretation": "i",
                                        "assumptions": [], "calls": []}, [])
    system = messages[0]["content"].lower()

    assert "cause" in system or "causal" in system
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_analysis_prompts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi_agent.analysis.prompts'`

- [ ] **Step 3: Write the implementation**

```python
"""What the model is told, and the five examples that show it the edges.

Diversity matters more than count in few-shot, and the useful examples are the
boundaries rather than the happy path: a question with no stated baseline, a
causal question the data cannot answer, a window outside the data, a question
too vague to act on. Four of the five here are refusals or hedges, which is the
intended lesson. A system that answers everything is more dangerous on a factory
floor than one that says it cannot.

Whether this actually helps is a measurement, not an assumption -- see
`scripts/analysis_eval.py`.
"""

from __future__ import annotations

import json
from typing import Any

from aoi_agent.analysis.plan import PLANNABLE_TOOLS, Domains

SYSTEM_PROMPT = """You plan data lookups for a PCB production line's review station.

A supervisor asks a question in plain language. You turn it into a plan: which
of the available tools to call, with which arguments, and why. You do not answer
the question yourself and you do not run anything -- the plan is validated and
executed by the system, and the results come back to you separately.

Rules that matter more than completeness:

- State every assumption you make in "assumptions", in plain language. If the
  question says "is it high" without saying high compared to what, you choose a
  baseline, and the person reading the answer must be able to see which one you
  chose. A hidden baseline is how a correct query produces a wrong conclusion.
- If the question needs data outside what is held, return no calls and say so in
  "interpretation". Do not substitute the nearest window you do have.
- If the question asks *why* something happened, gather what exists and say in
  "assumptions" that the data shows association and not cause. Nothing available
  to you establishes cause.
- If the question is too vague to turn into arguments, return no calls and say
  in "interpretation" what you would need to know.

A plan with no calls is a valid answer. Guessing arguments to avoid returning
one is not."""

SYNTHESIS_PROMPT = """You write the answer a PCB line supervisor reads.

You are given their question, the plan that was run, and what each tool
returned. Describe what the results show, in their terms, in a short paragraph.

Constraints:

- Describe only what is in the results. Do not add figures, trends or context
  that are not there.
- Never state or imply a cause. The tools carry association, not causation, and
  a plausible causal sentence here is what gets a machine stopped for the wrong
  reason. If the question asked why, say what the data associates and what would
  be needed to establish cause.
- If a tool failed, say plainly which one and what is therefore missing from
  your answer. Do not work around it silently.
- Repeat the assumptions from the plan in your own words, so the baseline is
  visible in the prose and not only in a side panel."""


FEW_SHOT: list[dict[str, Any]] = [
    {
        "shape": "cross_tool",
        "question": "L2-M22 最近的 open 是不是不尋常？該停機嗎？",
        "plan": {
            "interpretation": "Whether machine M22 on line L2 shows an unusual "
            "rate of open-circuit defects, and what the criteria say about it.",
            "assumptions": [
                "Unusual means compared with the other machines and the fleet "
                "average over the same window.",
                "The full span of data held is used, since it is short.",
            ],
            "calls": [
                {
                    "tool": "query_machine_stats",
                    "args": {"defect_type": "open", "days": 7},
                    "why": "M22's open rate against every other machine",
                },
                {
                    "tool": "query_defect_history",
                    "args": {"machine_id": "M22", "days": 7},
                    "why": "M22's own defect mix, to see if opens dominate it",
                },
                {
                    "tool": "search_standards",
                    "args": {"query": "open circuit acceptance and disposition",
                             "top_k": 2},
                    "why": "what the criteria require once an open is confirmed",
                },
            ],
        },
    },
    {
        "shape": "unstated_baseline",
        "question": "L1 的缺陷是不是變多了？",
        "plan": {
            "interpretation": "Whether line L1's defect count is high.",
            "assumptions": [
                "The question does not say compared with what. Comparing L1 "
                "against the other two lines over the same window, not against "
                "an earlier period, because the data held is too short to "
                "separate two periods.",
            ],
            "calls": [
                {"tool": "query_defect_history", "args": {"line_id": "L1", "days": 7},
                 "why": "L1's defect counts"},
                {"tool": "query_defect_history", "args": {"line_id": "L2", "days": 7},
                 "why": "L2, as one of the two comparison lines"},
                {"tool": "query_defect_history", "args": {"line_id": "L3", "days": 7},
                 "why": "L3, as the other comparison line"},
            ],
        },
    },
    {
        "shape": "causal",
        "question": "為什麼 M22 最近怪怪的？",
        "plan": {
            "interpretation": "What the data shows about M22 that differs from "
            "the other machines. The question asks for a cause.",
            "assumptions": [
                "Nothing available establishes cause. These results show what "
                "M22 is associated with, not why. Establishing cause would need "
                "maintenance, programme-change or material-lot records, which "
                "this system does not hold.",
            ],
            "calls": [
                {"tool": "query_defect_history", "args": {"machine_id": "M22", "days": 7},
                 "why": "M22's defect mix"},
                {"tool": "query_machine_stats", "args": {"defect_type": "open", "days": 7},
                 "why": "whether M22 stands out from the fleet on its dominant class"},
            ],
        },
    },
    {
        "shape": "out_of_range",
        "question": "去年這個時候的 open 缺陷率是多少？",
        "plan": {
            "interpretation": "This asks for a period the store does not cover. "
            "The data held spans a single short window; there is no last year to "
            "compare against. Returning the window that does exist would answer "
            "a different question than the one asked.",
            "assumptions": [],
            "calls": [],
        },
    },
    {
        "shape": "too_vague",
        "question": "產線最近怎麼樣？",
        "plan": {
            "interpretation": "Too broad to turn into arguments. Which line, and "
            "which aspect -- defect counts, a particular defect class, one "
            "machine against the others, or how much review the station is "
            "generating?",
            "assumptions": [],
            "calls": [],
        },
    },
]


def _tool_catalogue() -> str:
    import inspect

    lines = []
    for name, function in sorted(PLANNABLE_TOOLS.items()):
        signature = inspect.signature(function)
        summary = (function.__doc__ or "").strip().splitlines()[0]
        lines.append(f"- {name}{signature}\n    {summary}")
    return "\n".join(lines)


def _domain_note(domains: Domains) -> str:
    return (
        f"Lines: {', '.join(sorted(domains['line_id']))}\n"
        f"Machines: {', '.join(sorted(domains['machine_id']))}\n"
        f"Defect classes: {', '.join(sorted(domains['defect_type']))}\n"
        f"The store holds {domains['max_days']} days of inspection data. "
        f"`days` must not exceed that; a larger window silently returns the "
        f"same span and would report two different periods as identical."
    )


def build_planning_messages(question: str, domains: Domains) -> list[dict]:
    """System prompt, catalogue, domains, five examples, then the question."""
    examples = "\n\n".join(
        f"Question: {e['question']}\nPlan: {json.dumps(e['plan'], ensure_ascii=False)}"
        for e in FEW_SHOT
    )
    context = (
        f"Tools available:\n{_tool_catalogue()}\n\n"
        f"Values that exist:\n{_domain_note(domains)}\n\n"
        f"Examples:\n\n{examples}"
    )
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{context}"},
        {"role": "user", "content": f"Question: {question}"},
    ]


def build_synthesis_messages(question: str, plan: dict, results: list[dict]) -> list[dict]:
    """The question, what was assumed, and everything the tools returned."""
    rendered = []
    for result in results:
        if result["ok"]:
            rendered.append(
                f"[{result['tool']} {json.dumps(result['args'], ensure_ascii=False)}]\n"
                f"{json.dumps(result['data'], ensure_ascii=False)}"
            )
        else:
            rendered.append(
                f"[{result['tool']} {json.dumps(result['args'], ensure_ascii=False)}]\n"
                f"FAILED: {result['error']}"
            )

    assumptions = "\n".join(f"- {a}" for a in plan.get("assumptions") or []) or "- none"
    body = (
        f"Question: {question}\n\n"
        f"How it was read: {plan.get('interpretation', '')}\n\n"
        f"Assumptions made:\n{assumptions}\n\n"
        f"Results:\n\n" + "\n\n".join(rendered)
    )
    return [
        {"role": "system", "content": SYNTHESIS_PROMPT},
        {"role": "user", "content": body},
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_analysis_prompts.py -q`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add src/aoi_agent/analysis/prompts.py tests/test_analysis_prompts.py
git commit -m "Teach the planner with five examples, four of them edges

Diversity beats count in few-shot, and the useful examples are boundaries: a
comparison with no stated baseline, a causal question the data cannot answer, a
window outside the store, a question too vague to turn into arguments. A refusal
is expressed as a plan with no calls and an interpretation saying why, so it
renders through the same path as an answer rather than as an error.

A test asserts every example plan passes validation, because an example the
validator rejects teaches the model to produce plans the validator rejects.

The synthesis prompt forbids stating a cause. The tools carry association only,
and a plausible causal sentence is what gets a machine stopped for the wrong
reason."
```

---

### Task 5: The graph

Verified before this plan was written: `Send("run_tool", payload)` hands the
payload to the node as its whole state, the node's return value merges into the
parent through the reducer, and three 0.2s branches complete in 0.21s rather
than 0.6s. Against the real store, four tools take 183ms in parallel and 462ms
in sequence.

**Files:**
- Create: `src/aoi_agent/analysis/graph.py`
- Test: `tests/test_analysis_graph.py`

**Interfaces:**
- Consumes: `validate_plan`, `store_domains`, `PLAN_SCHEMA` (task 1); `run_call`, `ToolResult` (task 2); `chart_spec_for` (task 3); `build_planning_messages`, `build_synthesis_messages` (task 4); `OllamaClient`
- Produces:
  - `AnalysisState` TypedDict with `results: Annotated[list[ToolResult], operator.add]`
  - `build_analysis_graph(client, domains=None) -> CompiledStateGraph`

- [ ] **Step 1: Write the failing tests**

```python
"""The analysis flow, with a stubbed model so it needs neither GPU nor Ollama."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import pytest

from aoi_agent.analysis import graph as analysis
from aoi_agent.llm.ollama import ChatResult, Timing

DOMAINS = {
    "line_id": {"L1", "L2", "L3"},
    "machine_id": {"M11", "M12", "M21", "M22", "M31", "M32"},
    "defect_type": {"open", "short", "mousebite", "spur", "copper", "pin-hole"},
    "max_days": 9,
}

GOOD_PLAN = {
    "interpretation": "M22 against the fleet",
    "assumptions": ["compared with the fleet average"],
    "calls": [
        {"tool": "query_machine_stats", "args": {"defect_type": "open", "days": 7},
         "why": "fleet comparison"},
        {"tool": "search_standards", "args": {"query": "open", "top_k": 2},
         "why": "criteria"},
    ],
}


@dataclass
class StubClient:
    """Returns the plan first and the prose second, recording both prompts."""

    plan: dict = field(default_factory=lambda: GOOD_PLAN)
    answer: str = "M22 runs above the fleet on opens."
    calls: list = field(default_factory=list)

    def chat(self, messages, **kwargs) -> ChatResult:
        self.calls.append(messages)
        text = json.dumps(self.plan) if kwargs.get("response_format") else self.answer
        return ChatResult(text=text, tool_calls=[], thinking="",
                          timing=Timing(1.0, 0.0, 1.0, 1.0, 10, 10))


@pytest.fixture
def stub_tools(monkeypatch):
    """Every tool answers instantly and records that it ran."""
    ran = []

    def make(name, payload):
        def tool(**kwargs):
            ran.append(name)
            time.sleep(0.05)
            return payload
        return tool

    monkeypatch.setitem(
        analysis.PLANNABLE_TOOLS, "query_machine_stats",
        make("query_machine_stats",
             {"defect_type": "open", "fleet_share_of_defects": 0.2,
              "machines": [{"machine": "L2-M22", "share_of_defects": 0.32,
                            "per_board": 2.3}]}),
    )
    monkeypatch.setitem(
        analysis.PLANNABLE_TOOLS, "search_standards",
        make("search_standards", {"passages": [{"document": "WI-201", "text": "x"}]}),
    )
    return ran


def run(client, question="M22 正常嗎", domains=DOMAINS):
    return analysis.build_analysis_graph(client, domains).invoke(
        {"question": question, "results": [], "timings_ms": {}}
    )


def test_a_valid_plan_runs_every_call_and_produces_an_answer(stub_tools):
    state = run(StubClient())

    assert sorted(stub_tools) == ["query_machine_stats", "search_standards"]
    assert len(state["results"]) == 2
    assert state["answer"]
    assert state["plan_errors"] == []


def test_the_results_arrive_through_the_reducer_not_by_overwriting(stub_tools):
    """Two parallel branches both write `results`. Without the reducer this is
    an InvalidUpdateError; with it, both land."""
    state = run(StubClient())
    tools = {r["tool"] for r in state["results"]}

    assert tools == {"query_machine_stats", "search_standards"}


def test_the_branches_run_concurrently(stub_tools):
    """Each stub sleeps 50ms. Sequential would be 100ms plus overhead."""
    started = time.perf_counter()
    run(StubClient())
    elapsed = (time.perf_counter() - started) * 1000

    assert elapsed < 90, f"branches look sequential: {elapsed:.0f}ms"


def test_an_invalid_plan_runs_nothing_and_reports_every_error(stub_tools):
    bad = {
        "interpretation": "i", "assumptions": [],
        "calls": [{"tool": "query_defect_history",
                   "args": {"line_id": "L9", "days": 999}, "why": "w"}],
    }
    state = run(StubClient(plan=bad))

    assert stub_tools == [], "no tool may run when validation fails"
    assert len(state["plan_errors"]) >= 2
    assert state["results"] == []
    assert state["chart_spec"] is None


def test_a_refusal_is_not_an_error(stub_tools):
    """A plan with no calls is the model declining, and it renders as an answer
    rather than as a failure."""
    refusal = {"interpretation": "the store does not cover last year",
               "assumptions": [], "calls": []}
    state = run(StubClient(plan=refusal))

    assert stub_tools == []
    assert state["plan"]["interpretation"]
    assert state["refused"] is True


def test_one_failing_branch_does_not_take_the_others_with_it(monkeypatch, stub_tools):
    def boom(**kwargs):
        raise RuntimeError("index unreachable")

    monkeypatch.setitem(analysis.PLANNABLE_TOOLS, "search_standards", boom)
    state = run(StubClient())

    assert len(state["results"]) == 2
    ok = {r["tool"]: r["ok"] for r in state["results"]}
    assert ok == {"query_machine_stats": True, "search_standards": False}
    assert state["answer"], "a partial answer is still an answer"


def test_the_failure_reaches_the_synthesis_prompt(monkeypatch, stub_tools):
    """The answer can only name what is missing if it is told."""
    def boom(**kwargs):
        raise RuntimeError("index unreachable")

    monkeypatch.setitem(analysis.PLANNABLE_TOOLS, "search_standards", boom)
    client = StubClient()
    run(client)

    synthesis = "\n".join(m["content"] for m in client.calls[-1])
    assert "index unreachable" in synthesis


def test_an_unparseable_plan_is_reported_rather_than_guessed_at(stub_tools):
    class Garbage(StubClient):
        def chat(self, messages, **kwargs):
            self.calls.append(messages)
            return ChatResult(text="I think you want stats?", tool_calls=[],
                              thinking="", timing=Timing(1.0, 0.0, 1.0, 1.0, 10, 10))

    state = run(Garbage())

    assert stub_tools == []
    assert any("parse" in e.lower() for e in state["plan_errors"])


def test_an_unreachable_model_reports_instead_of_crashing(stub_tools):
    import httpx

    class Dead:
        def chat(self, messages, **kwargs):
            raise httpx.ReadTimeout("timed out")

    state = run(Dead())

    assert stub_tools == []
    assert any("ReadTimeout" in e for e in state["plan_errors"])


def test_timings_are_recorded_per_tool_and_for_the_phases(stub_tools):
    state = run(StubClient())

    assert "plan" in state["timings_ms"]
    assert "tools_wall" in state["timings_ms"]
    assert "synthesise" in state["timings_ms"]
    assert state["timings_ms"]["tools_sequential"] >= state["timings_ms"]["tools_wall"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_analysis_graph.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi_agent.analysis.graph'`

- [ ] **Step 3: Write the implementation**

```python
"""The analysis flow.

A second graph, separate from the disposition flow. It has no checkpointer:
nothing suspends, nobody is in the loop, and a question is answered in one
invocation. The disposition flow needs one because of `interrupt`; adopting a
framework feature that has no work to do is what this project spent a day
removing from the other graph.

What the graph is for here is the fan-out. `Send` expands a plan of N calls into
N branches whose count is not known until the plan exists, and an `operator.add`
reducer merges what they return. Without the reducer two branches writing
`results` in one superstep is an `InvalidUpdateError` -- which is the framework
refusing to let a race happen rather than a race happening.

Measured before building: four real tools take 183ms in parallel against 462ms
in sequence, while the two model calls cost around 25 seconds. The fan-out is
the correct structure for independent work and it scales as tools multiply. It
is not a latency optimisation, and nothing here should claim it is.
"""

from __future__ import annotations

import json
import operator
import time
from typing import Annotated, Any, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from aoi_agent.analysis.charts import chart_spec_for
from aoi_agent.analysis.plan import (
    PLAN_SCHEMA,
    PLANNABLE_TOOLS,
    Domains,
    store_domains,
    validate_plan,
)
from aoi_agent.analysis.prompts import (
    build_planning_messages,
    build_synthesis_messages,
)
from aoi_agent.analysis.tools import ToolResult, run_call


class AnalysisState(TypedDict, total=False):
    question: str
    plan: dict | None
    plan_errors: list[str]
    refused: bool
    results: Annotated[list[ToolResult], operator.add]
    timings_ms: Annotated[dict[str, float], operator.or_]
    chart_spec: dict | None
    answer: str


def make_plan_node(client, domains: Domains):
    def plan_node(state: AnalysisState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            result = client.chat(
                build_planning_messages(state["question"], domains),
                think="low",
                response_format=PLAN_SCHEMA,
            )
        except (httpx.HTTPError, OSError) as error:
            return {
                "plan": None,
                "plan_errors": [f"the planner did not answer ({type(error).__name__})"],
                "refused": False,
                "timings_ms": {"plan": (time.perf_counter() - started) * 1000},
            }

        try:
            plan = json.loads(result.text)
        except json.JSONDecodeError:
            # An unparseable plan is not a plan. Showing the person what came
            # back beats guessing at what was meant.
            return {
                "plan": None,
                "plan_errors": ["the planner's response could not be parsed as a plan"],
                "refused": False,
                "timings_ms": {"plan": (time.perf_counter() - started) * 1000},
            }

        # A plan with no calls is the model declining, not failing. It carries
        # its reason in `interpretation` and renders as an answer.
        refused = not (plan.get("calls") or [])
        errors = [] if refused else validate_plan(plan, domains)

        return {
            "plan": plan,
            "plan_errors": errors,
            "refused": refused,
            "timings_ms": {"plan": (time.perf_counter() - started) * 1000},
        }

    return plan_node


def fan_out(state: AnalysisState) -> list[Send] | str:
    """Expand the plan into one branch per call.

    The number of branches comes from the plan, which is why this is `Send` and
    not a fixed set of edges: the graph's shape is not known until the question
    has been read.
    """
    if state.get("plan_errors") or state.get("refused") or not state.get("plan"):
        return "report"
    return [Send("run_tool", {"call": call}) for call in state["plan"]["calls"]]


def run_tool_node(payload: dict) -> dict[str, Any]:
    """Receives one call as its entire state; contributes one result upward."""
    return {"results": [run_call(payload["call"])]}


def collect_node(state: AnalysisState) -> dict[str, Any]:
    """Join. Builds the chart and records what the fan-out actually saved."""
    results = state.get("results") or []
    sequential = sum(r["elapsed_ms"] for r in results)
    wall = max((r["elapsed_ms"] for r in results), default=0.0)
    return {
        "chart_spec": chart_spec_for(results),
        "timings_ms": {
            "tools_wall": round(wall, 1),
            "tools_sequential": round(sequential, 1),
        },
    }


def make_synthesise_node(client):
    def synthesise_node(state: AnalysisState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            result = client.chat(
                build_synthesis_messages(
                    state["question"], state.get("plan") or {}, state.get("results") or []
                ),
                think="low",
            )
            answer = result.text.strip()
        except (httpx.HTTPError, OSError) as error:
            # The results are already correct and already on screen. Losing the
            # prose costs a reader some effort; losing the results would cost
            # them the answer.
            answer = (
                f"The tools returned their results, but the summary could not be "
                f"written ({type(error).__name__}). The figures below are complete."
            )
        return {
            "answer": answer,
            "timings_ms": {"synthesise": (time.perf_counter() - started) * 1000},
        }

    return synthesise_node


def report_node(state: AnalysisState) -> dict[str, Any]:
    """Terminal for a refusal or a rejected plan. Nothing ran; say why."""
    if state.get("refused"):
        plan = state.get("plan") or {}
        return {"answer": plan.get("interpretation", "No lookup could be planned."),
                "chart_spec": None}
    errors = "\n".join(f"- {e}" for e in state.get("plan_errors") or [])
    return {
        "answer": "The plan was not run because it did not validate:\n" + errors,
        "chart_spec": None,
    }


def build_analysis_graph(client, domains: Domains | None = None):
    """Compile the flow. No checkpointer: nothing here suspends."""
    domains = domains or store_domains()

    graph = StateGraph(AnalysisState)
    graph.add_node("plan", make_plan_node(client, domains))
    graph.add_node("run_tool", run_tool_node)
    graph.add_node("collect", collect_node)
    graph.add_node("synthesise", make_synthesise_node(client))
    graph.add_node("report", report_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", fan_out, ["run_tool", "report"])
    graph.add_edge("run_tool", "collect")
    graph.add_edge("collect", "synthesise")
    graph.add_edge("synthesise", END)
    graph.add_edge("report", END)

    return graph.compile()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_analysis_graph.py -q`
Expected: PASS, 10 tests

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/aoi_agent/analysis/graph.py tests/test_analysis_graph.py
git commit -m "Add the analysis flow, fanning out over the plan with Send

The number of branches is not known until the question has been read, which is
what Send is for and what a fixed set of edges could not express. An
operator.add reducer merges what the branches return; without it two of them
writing results in one superstep is an InvalidUpdateError, which is the
framework refusing to let a race happen rather than a race happening.

No checkpointer. Nothing suspends, nobody is in the loop, and a question is
answered in one invocation -- adopting a framework feature with no work to do is
what this project spent a day removing from the other graph.

A refusal is a plan with no calls and renders as an answer. A rejected plan runs
nothing and shows every error at once. One failing branch leaves its siblings
alone and its error reaches the synthesis prompt, so the answer can name what is
missing rather than quietly working around it.

Measured before building: four real tools take 183ms in parallel against 462ms
sequential, while the two model calls cost around 25s. The fan-out is the right
structure for independent work, not a latency optimisation, and the code says
so where someone might otherwise claim it."
```

---

### Task 6: Persisting a run, and the service both surfaces share

The chart must be redrawable without re-running a plan that would not
regenerate identically. That is what the table is for; it is also the log the
eval script and any later live view read from.

**Files:**
- Modify: `src/aoi_agent/store/models.py` — add `AnalysisRun` beside `Escalation`
- Create: `src/aoi_agent/store/analysis.py`
- Create: `src/aoi_agent/analysis/service.py`
- Test: `tests/test_analysis_service.py`

**Interfaces:**
- Consumes: `build_analysis_graph` (task 5)
- Produces:
  - `AnalysisRun` model: `id`, `question`, `plan_json`, `results_json`, `chart_json`, `answer`, `timings_json`, `refused`, `asked_by`, `asked_at`
  - `save_run(...) -> int`, `get_run(run_id) -> dict | None`, `recent_runs(limit) -> list[dict]`
  - `answer_question(graph, question, asked_by="operator") -> dict` — runs and persists, returning the run dict

- [ ] **Step 1: Write the failing tests**

```python
"""Running a question end to end and keeping enough to redraw it."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from aoi_agent.analysis import graph as analysis
from aoi_agent.analysis import service
from aoi_agent.llm.ollama import ChatResult, Timing
from aoi_agent.store import analysis as store_analysis
from aoi_agent.store.boards import session_factory
from aoi_agent.store.models import create_all, make_session_factory

PLAN = {
    "interpretation": "M22 against the fleet",
    "assumptions": ["compared with the fleet average"],
    "calls": [{"tool": "query_machine_stats",
               "args": {"defect_type": "open", "days": 7}, "why": "w"}],
}
DOMAINS = {
    "line_id": {"L1", "L2", "L3"}, "machine_id": {"M22"},
    "defect_type": {"open"}, "max_days": 9,
}


@dataclass
class StubClient:
    plan: dict = field(default_factory=lambda: PLAN)
    answer: str = "M22 sits above the fleet."

    def chat(self, messages, **kwargs) -> ChatResult:
        text = json.dumps(self.plan) if kwargs.get("response_format") else self.answer
        return ChatResult(text=text, tool_calls=[], thinking="",
                          timing=Timing(1.0, 0.0, 1.0, 1.0, 10, 10))


@pytest.fixture
def store(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'a.db'}"
    create_all(url)
    monkeypatch.setattr("aoi_agent.store.boards._session_factory",
                        make_session_factory(url))


@pytest.fixture
def graph(monkeypatch):
    monkeypatch.setitem(
        analysis.PLANNABLE_TOOLS, "query_machine_stats",
        lambda **kw: {"defect_type": "open", "fleet_share_of_defects": 0.2,
                      "machines": [{"machine": "L2-M22", "share_of_defects": 0.32,
                                    "per_board": 2.3}]},
    )
    return analysis.build_analysis_graph(StubClient(), DOMAINS)


def test_a_question_is_answered_and_persisted(store, graph):
    run = service.answer_question(graph, "M22 正常嗎", asked_by="mike")

    assert run["answer"]
    assert run["id"]
    assert run["asked_by"] == "mike"
    assert store_analysis.get_run(run["id"])["question"] == "M22 正常嗎"


def test_the_stored_run_can_redraw_its_chart_without_a_model(store, graph):
    """The point of persisting the spec rather than an image: a run from last
    quarter renders today, and no model is asked to reproduce a plan it would
    not reproduce."""
    run = service.answer_question(graph, "M22 正常嗎")
    stored = store_analysis.get_run(run["id"])

    assert stored["chart"]["kind"] == "bar"
    assert stored["chart"]["series"][0]["points"][0]["x"] == "L2-M22"


def test_the_raw_results_are_kept_beside_the_prose(store, graph):
    """Synthesis can describe correct data incorrectly. A reader can only catch
    that if the data is there to check against."""
    run = service.answer_question(graph, "M22 正常嗎")
    stored = store_analysis.get_run(run["id"])

    assert stored["results"][0]["tool"] == "query_machine_stats"
    assert stored["results"][0]["data"]["machines"]


def test_a_refusal_is_stored_too(store, monkeypatch):
    """What the system declined to answer is as interesting as what it did, and
    the eval script reads refusals from here."""
    refusal = {"interpretation": "the store does not cover last year",
               "assumptions": [], "calls": []}
    graph = analysis.build_analysis_graph(StubClient(plan=refusal), DOMAINS)
    run = service.answer_question(graph, "去年呢")

    assert run["refused"] is True
    assert store_analysis.get_run(run["id"])["refused"] is True


def test_recent_runs_come_back_newest_first(store, graph):
    for question in ("一", "二", "三"):
        service.answer_question(graph, question)

    assert [r["question"] for r in store_analysis.recent_runs(10)][:3] == ["三", "二", "一"]


def test_the_run_records_what_the_fan_out_saved(store, graph):
    run = service.answer_question(graph, "M22 正常嗎")
    stored = store_analysis.get_run(run["id"])

    assert "tools_wall" in stored["timings"]
    assert "tools_sequential" in stored["timings"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_analysis_service.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi_agent.analysis.service'`

- [ ] **Step 3: Add the model**

In `src/aoi_agent/store/models.py`, after the `Escalation` class and before `make_engine`:

```python
class AnalysisRun(Base):
    """One natural-language question, and everything needed to redraw it.

    The plan, the raw results and the chart specification are kept rather than
    an image, so a run recorded this quarter renders next quarter without asking
    a model to reproduce a plan it would not reproduce. It is also what the
    evaluation script reads, and what a live view of a running graph would
    replay.
    """

    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(String(1024))
    plan_json: Mapped[str | None] = mapped_column(String, nullable=True)
    results_json: Mapped[str] = mapped_column(String, default="[]")
    chart_json: Mapped[str | None] = mapped_column(String, nullable=True)
    answer: Mapped[str] = mapped_column(String, default="")
    timings_json: Mapped[str] = mapped_column(String, default="{}")
    refused: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    asked_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    """Free text until the station has authentication -- same gap as
    ``ReviewDecision.reviewer``, and this page widens it, because a query
    interface exposes plant-wide statistics where the queue exposed a queue."""

    asked_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

Add `Boolean` to the `sqlalchemy` import list at the top of the file.

- [ ] **Step 4: Write the store accessor**

Create `src/aoi_agent/store/analysis.py`:

```python
"""Reading and writing analysis runs.

JSON columns rather than tables of rows: nothing queries inside a plan or a
result set, they are read back whole to render or to score. Normalising them
would buy nothing and cost a migration every time a tool's return shape moves.
"""

from __future__ import annotations

import json

from sqlalchemy import select

from aoi_agent.store.boards import session_factory
from aoi_agent.store.models import AnalysisRun


def _as_dict(row: AnalysisRun) -> dict:
    return {
        "id": row.id,
        "question": row.question,
        "plan": json.loads(row.plan_json) if row.plan_json else None,
        "results": json.loads(row.results_json),
        "chart": json.loads(row.chart_json) if row.chart_json else None,
        "answer": row.answer,
        "timings": json.loads(row.timings_json),
        "refused": row.refused,
        "asked_by": row.asked_by,
        "asked_at": row.asked_at.isoformat() if row.asked_at else None,
    }


def save_run(
    question: str,
    plan: dict | None,
    results: list[dict],
    chart: dict | None,
    answer: str,
    timings: dict,
    refused: bool,
    asked_by: str | None,
) -> int:
    with session_factory()() as session:
        row = AnalysisRun(
            question=question,
            plan_json=json.dumps(plan, ensure_ascii=False) if plan else None,
            results_json=json.dumps(results, ensure_ascii=False),
            chart_json=json.dumps(chart, ensure_ascii=False) if chart else None,
            answer=answer,
            timings_json=json.dumps(timings),
            refused=refused,
            asked_by=asked_by,
        )
        session.add(row)
        session.commit()
        return row.id


def get_run(run_id: int) -> dict | None:
    with session_factory()() as session:
        row = session.get(AnalysisRun, run_id)
        return _as_dict(row) if row else None


def recent_runs(limit: int = 20) -> list[dict]:
    with session_factory()() as session:
        rows = session.execute(
            select(AnalysisRun).order_by(AnalysisRun.id.desc()).limit(limit)
        ).scalars().all()
        return [_as_dict(row) for row in rows]
```

- [ ] **Step 5: Write the service**

Create `src/aoi_agent/analysis/service.py`:

```python
"""Answering one question, and keeping it.

The CLI and the station both come through here, so the two cannot drift -- the
same arrangement as `station/service.py` for the disposition path.
"""

from __future__ import annotations

from typing import Any

from aoi_agent.store import analysis as store


def answer_question(graph, question: str, asked_by: str | None = "operator") -> dict[str, Any]:
    """Run one question through the analysis graph and persist the result."""
    state = graph.invoke({"question": question, "results": [], "timings_ms": {}})

    run_id = store.save_run(
        question=question,
        plan=state.get("plan"),
        results=state.get("results") or [],
        chart=state.get("chart_spec"),
        answer=state.get("answer", ""),
        timings=state.get("timings_ms") or {},
        refused=bool(state.get("refused")),
        asked_by=asked_by,
    )
    return {**store.get_run(run_id), "plan_errors": state.get("plan_errors") or []}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_analysis_service.py -q`
Expected: PASS, 6 tests

- [ ] **Step 7: Create the table in the live store and run the suite**

```bash
uv run python -c "from aoi_agent.store.models import create_all; create_all()"
uv run pytest -q
```
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/aoi_agent/store/models.py src/aoi_agent/store/analysis.py \
        src/aoi_agent/analysis/service.py tests/test_analysis_service.py
git commit -m "Persist every analysis run so its chart outlives its plan

The chart is stored as a specification, not an image, and the raw results are
stored beside the prose. A run recorded this quarter renders next quarter
without asking a model to reproduce a plan it would not reproduce, and a reader
who suspects the summary can check it against the figures it claims to
describe.

Refusals are kept too. What the system declined to answer is as interesting as
what it did, and the evaluation script reads them from here.

JSON columns rather than normalised tables: nothing queries inside a plan, they
are read back whole, and normalising would cost a migration every time a tool's
return shape moves."
```

---

### Task 7: The page

Server-rendered SVG rather than a charting library. The station already works
with JavaScript off and vendors its one dependency; a chart that needs a CDN
would break both.

**Files:**
- Create: `src/aoi_agent/station/chart_svg.py`
- Create: `src/aoi_agent/station/templates/analysis.html`
- Modify: `src/aoi_agent/station/app.py` — add `GET /ask`, `POST /ask`, `GET /ask/{run_id}`
- Modify: `src/aoi_agent/station/templates/base.html` — add `ask` to the nav
- Modify: `src/aoi_agent/station/static/style.css` — append the analysis styles
- Test: `tests/test_analysis_page.py`

**Interfaces:**
- Consumes: `answer_question` (task 6), `recent_runs`, `get_run` (task 6)
- Produces: `render_svg(spec: dict, width: int = 720, height: int = 280) -> str`

- [ ] **Step 1: Write the failing tests**

```python
"""The analysis page: what it renders, and what it refuses to hide."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from aoi_agent.analysis import graph as analysis
from aoi_agent.llm.ollama import ChatResult, Timing
from aoi_agent.station import app as station_app
from aoi_agent.station.chart_svg import render_svg
from aoi_agent.store.models import create_all, make_session_factory

PLAN = {
    "interpretation": "M22 against the fleet",
    "assumptions": ["compared with the fleet average over the whole span"],
    "calls": [{"tool": "query_machine_stats",
               "args": {"defect_type": "open", "days": 7}, "why": "fleet comparison"}],
}
DOMAINS = {"line_id": {"L2"}, "machine_id": {"M22"},
           "defect_type": {"open"}, "max_days": 9}

SPEC = {
    "kind": "bar", "title": "open share by machine",
    "x_label": "machine", "y_label": "share",
    "series": [{"name": "share of open",
                "points": [{"x": "L2-M22", "y": 0.32}, {"x": "L1-M11", "y": 0.19}]}],
}


@dataclass
class StubClient:
    plan: dict = field(default_factory=lambda: PLAN)
    answer: str = "M22 sits above the fleet on opens."

    def chat(self, messages, **kwargs) -> ChatResult:
        text = json.dumps(self.plan) if kwargs.get("response_format") else self.answer
        return ChatResult(text=text, tool_calls=[], thinking="",
                          timing=Timing(1.0, 0.0, 1.0, 1.0, 10, 10))


@pytest.fixture
def client(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'a.db'}"
    create_all(url)
    monkeypatch.setattr("aoi_agent.store.boards._session_factory",
                        make_session_factory(url))
    monkeypatch.setitem(
        analysis.PLANNABLE_TOOLS, "query_machine_stats",
        lambda **kw: {"defect_type": "open", "fleet_share_of_defects": 0.2,
                      "machines": [{"machine": "L2-M22", "share_of_defects": 0.32,
                                    "per_board": 2.3}]},
    )
    monkeypatch.setattr(
        station_app, "_analysis_graph",
        analysis.build_analysis_graph(StubClient(), DOMAINS),
    )
    return TestClient(station_app.app)


def test_the_svg_draws_one_rect_per_point_and_labels_the_axes():
    svg = render_svg(SPEC)

    assert svg.count("<rect") >= 2
    assert "L2-M22" in svg
    assert "open share by machine" in svg
    assert "share" in svg


def test_the_svg_survives_a_zero_valued_series():
    """A flat series must not divide by zero on the way to a scale factor."""
    flat = {**SPEC, "series": [{"name": "s", "points": [{"x": "a", "y": 0.0}]}]}
    assert "<svg" in render_svg(flat)


def test_the_empty_page_shows_the_examples_and_the_coverage(client):
    page = client.get("/ask").text

    assert "L2-M22" in page, "an example question"
    assert "涵蓋" in page or "covers" in page.lower(), "the data span must be stated"


def test_asking_a_question_shows_all_five_blocks(client):
    response = client.post("/ask", data={"question": "M22 正常嗎"},
                           follow_redirects=True)
    page = response.text

    assert "M22 against the fleet" in page, "1. interpretation"
    assert "query_machine_stats" in page, "2. the calls"
    assert "fleet average" in page, "3. the assumptions"
    assert "ms" in page, "4. timing"
    assert "sits above the fleet" in page, "5. the prose"
    assert "<svg" in page, "the chart"


def test_the_page_shows_what_the_fan_out_cost_and_saved(client):
    page = client.post("/ask", data={"question": "M22 正常嗎"},
                       follow_redirects=True).text

    assert "parallel" in page.lower() or "平行" in page


def test_a_stored_run_renders_again_without_the_model(client, monkeypatch):
    """Once saved, a run is a document. Reopening it must not call anything."""
    run_id = client.post("/ask", data={"question": "M22 正常嗎"},
                         follow_redirects=True).url.path.rsplit("/", 1)[-1]

    class Exploding:
        def chat(self, *a, **k):
            raise AssertionError("the model must not be called to re-render")

    monkeypatch.setattr(station_app, "_analysis_graph",
                        analysis.build_analysis_graph(Exploding(), DOMAINS))
    page = client.get(f"/ask/{run_id}").text

    assert "<svg" in page
    assert "sits above the fleet" in page


def test_a_rejected_plan_is_shown_with_its_errors_and_no_chart(client, monkeypatch):
    bad = {"interpretation": "i", "assumptions": [],
           "calls": [{"tool": "query_defect_history",
                      "args": {"line_id": "L9", "days": 999}, "why": "w"}]}
    monkeypatch.setattr(station_app, "_analysis_graph",
                        analysis.build_analysis_graph(StubClient(plan=bad), DOMAINS))
    page = client.post("/ask", data={"question": "L9 呢"},
                       follow_redirects=True).text

    assert "L9" in page
    assert "<svg" not in page


def test_an_empty_question_is_refused_before_the_model_is_asked(client):
    response = client.post("/ask", data={"question": "   "}, follow_redirects=False)
    assert response.status_code == 400


def test_an_unknown_run_is_a_404(client):
    assert client.get("/ask/99999").status_code == 404
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_analysis_page.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'aoi_agent.station.chart_svg'`

- [ ] **Step 3: Write the SVG renderer**

Create `src/aoi_agent/station/chart_svg.py`:

```python
"""A bar chart, as inline SVG.

No charting library. The station works with JavaScript off and vendors its one
dependency so it runs on a locked-down shop-floor browser; a chart that needs a
CDN would break both properties for a picture of six bars.
"""

from __future__ import annotations

from html import escape

PALETTE = ["#60a5fa", "#f59e0b", "#34d399", "#f472b6"]


def render_svg(spec: dict, width: int = 720, height: int = 280) -> str:
    """Render a chart specification. Returns '' for anything unplottable."""
    series = spec.get("series") or []
    if not series or not series[0].get("points"):
        return ""

    pad_left, pad_right, pad_top, pad_bottom = 56, 16, 28, 44
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    labels = [p["x"] for p in series[0]["points"]]
    peak = max((p["y"] for s in series for p in s["points"]), default=0.0)
    scale = plot_h / peak if peak > 0 else 0.0  # a flat series must not divide by zero

    group_w = plot_w / max(1, len(labels))
    bar_w = group_w / (len(series) + 1)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="{escape(spec.get("title", "chart"))}">',
        f'<text x="{pad_left}" y="18" fill="#e7e7ea" font-size="13">'
        f'{escape(spec.get("title", ""))}</text>',
        f'<line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{width - pad_right}" '
        f'y2="{pad_top + plot_h}" stroke="#2e2e35"/>',
    ]

    for index, one in enumerate(series):
        colour = PALETTE[index % len(PALETTE)]
        for position, point in enumerate(one["points"]):
            bar_h = point["y"] * scale
            x = pad_left + position * group_w + index * bar_w + bar_w / 2
            y = pad_top + plot_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{bar_h:.1f}" fill="{colour}" rx="2"><title>'
                f'{escape(str(point["x"]))}: {point["y"]}</title></rect>'
            )
        parts.append(
            f'<rect x="{pad_left + index * 92}" y="{height - 14}" width="9" '
            f'height="9" fill="{colour}" rx="2"/>'
            f'<text x="{pad_left + index * 92 + 14}" y="{height - 6}" '
            f'fill="#8b8b96" font-size="10">{escape(one["name"])}</text>'
        )

    for position, label in enumerate(labels):
        x = pad_left + position * group_w + group_w / 2
        parts.append(
            f'<text x="{x:.1f}" y="{pad_top + plot_h + 15}" fill="#8b8b96" '
            f'font-size="10" text-anchor="middle">{escape(str(label))}</text>'
        )

    parts.append(
        f'<text x="4" y="{pad_top + 8}" fill="#8b8b96" font-size="10">'
        f'{escape(spec.get("y_label", ""))}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)
```

- [ ] **Step 4: Add the routes**

In `src/aoi_agent/station/app.py`, add the imports and a lazily-built graph
beside the existing `graph()`:

```python
from aoi_agent.analysis.graph import build_analysis_graph
from aoi_agent.analysis.plan import store_domains
from aoi_agent.analysis.prompts import FEW_SHOT
from aoi_agent.analysis import service as analysis_service
from aoi_agent.station.chart_svg import render_svg
from aoi_agent.store import analysis as analysis_store

_analysis_graph = None


def analysis_graph():
    global _analysis_graph
    if _analysis_graph is None:
        _analysis_graph = build_analysis_graph(
            OllamaClient(os.getenv("AOI_AGENT_MODEL", DEFAULT_MODEL))
        )
    return _analysis_graph


#: Five, because they are the discoverability mechanism for someone who cannot
#: write a query and has no other way to learn what is answerable. They are also
#: the honest half of the dashboard argument in the spec: these are the common
#: questions, and the free-form box is for the tail.
EXAMPLE_QUESTIONS = [
    "L2-M22 的 open 是不是不尋常？該停機嗎？",
    "比較三條線的缺陷組成，並說明驗收規定",
    "哪一台機器的缺陷率最高？",
    "20085294 這片板子的脈絡是什麼？",
    "short 的驗收標準是什麼？",
]


def _analysis_context(request: Request, run: dict | None, errors: list[str]) -> dict:
    domains = store_domains()
    return {
        "run": run,
        "plan_errors": errors,
        "examples": EXAMPLE_QUESTIONS,
        "recent": analysis_store.recent_runs(8),
        "coverage_days": domains["max_days"],
        "chart_svg": render_svg(run["chart"]) if run and run.get("chart") else "",
        "waiting": len(escalations.pending()),
    }


@app.get("/ask", response_class=HTMLResponse)
def ask_page(request: Request):
    return templates.TemplateResponse(
        request, "analysis.html", _analysis_context(request, None, [])
    )


@app.post("/ask")
def ask(question: str = Form(...), asked_by: str = Form("operator")):
    if not question.strip():
        raise HTTPException(400, "a question is required")
    run = analysis_service.answer_question(
        analysis_graph(), question.strip(), asked_by.strip() or "operator"
    )
    return RedirectResponse(f"/ask/{run['id']}", status_code=303)


@app.get("/ask/{run_id}", response_class=HTMLResponse)
def ask_result(request: Request, run_id: int):
    run = analysis_store.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"no analysis run {run_id}")
    return templates.TemplateResponse(
        request, "analysis.html", _analysis_context(request, run, [])
    )
```

- [ ] **Step 5: Write the template**

Create `src/aoi_agent/station/templates/analysis.html`. The five blocks are
numbered in the markup so a reviewer can check none has been dropped.

```html
{% extends "base.html" %}
{% block title %}Ask &middot; production data{% endblock %}
{% block content %}
<section class="panel">
  <h1>問一個關於產線的問題</h1>

  <form class="ask" method="post" action="/ask">
    <input type="text" name="question" autocomplete="off" autofocus
           value="{{ run.question if run else '' }}"
           placeholder="例如：L2-M22 的 open 是不是不尋常？">
    <button type="submit">問</button>
  </form>

  <p class="sub">試試看：
    {% for example in examples %}
      <a class="example" href="#" onclick="document.querySelector('.ask input').value=this.textContent;return false">{{ example }}</a>
    {% endfor %}
  </p>
  <p class="sim">資料涵蓋最近 {{ coverage_days }} 天。超出這個範圍的問題會被拒答，而不是用最接近的窗口代答。</p>

  {% if plan_errors %}
    <div class="handover">
      <span class="label">這個計畫沒有執行</span>
      <ul>{% for e in plan_errors %}<li>{{ e }}</li>{% endfor %}</ul>
    </div>
  {% endif %}

  {% if run %}
    <div class="answer-block">
      <h2>1 &middot; 它怎麼理解你的問題</h2>
      <p>{{ run.plan.interpretation if run.plan else run.answer }}</p>
    </div>

    {% if run.plan and run.plan.assumptions %}
    <div class="answer-block">
      <h2>3 &middot; 它假設了什麼</h2>
      <ul>{% for a in run.plan.assumptions %}<li>{{ a }}</li>{% endfor %}</ul>
    </div>
    {% endif %}

    {% if run.results %}
    <div class="answer-block">
      <h2>2 &middot; 它呼叫了什麼</h2>
      <table class="queue">
        <thead><tr><th>工具</th><th>參數</th><th>為什麼</th><th class="num">耗時</th><th></th></tr></thead>
        <tbody>
        {% for r in run.results %}
          <tr>
            <td class="mono">{{ r.tool }}</td>
            <td class="mono dim">{{ r.args }}</td>
            <td class="dim">{{ (run.plan.calls[loop.index0].why) if run.plan and run.plan.calls|length > loop.index0 else '' }}</td>
            <td class="num">{{ '%.0f'|format(r.elapsed_ms) }}ms</td>
            <td>{% if r.ok %}<span class="chip done">ok</span>{% else %}<span class="chip warn">失敗</span>{% endif %}</td>
          </tr>
          {% if not r.ok %}<tr><td colspan="5" class="dim">{{ r.error }}</td></tr>{% endif %}
        {% endfor %}
        </tbody>
      </table>
      <p class="sub dim">4 &middot;
        {{ run.results|length }} 個工具 &middot;
        平行 {{ '%.0f'|format(run.timings.tools_wall or 0) }}ms /
        依序 {{ '%.0f'|format(run.timings.tools_sequential or 0) }}ms &middot;
        規劃 {{ '%.0f'|format(run.timings.plan or 0) }}ms &middot;
        撰寫 {{ '%.0f'|format(run.timings.synthesise or 0) }}ms
      </p>
    </div>
    {% endif %}

    {% if chart_svg %}
    <figure class="chart">{{ chart_svg|safe }}</figure>
    {% endif %}

    <div class="answer-block">
      <h2>5 &middot; 回答</h2>
      <p>{{ run.answer }}</p>
    </div>
  {% endif %}

  {% if recent %}
    <h2>最近問過的</h2>
    <ul class="recent">
      {% for r in recent %}
        <li><a href="/ask/{{ r.id }}">{{ r.question }}</a>
            {% if r.refused %}<span class="chip">拒答</span>{% endif %}</li>
      {% endfor %}
    </ul>
  {% endif %}
</section>
{% endblock %}
```

Add `<a href="/ask">ask</a>` to the `<nav>` in `base.html`, and append to
`style.css`:

```css
/* ---- analysis ---- */
.ask { display: flex; gap: 0.6rem; margin: 1rem 0 0.5rem; }
.ask input { flex: 1; background: var(--bg); color: var(--ink);
             border: 1px solid var(--line); border-radius: 8px;
             padding: 0.7rem 0.9rem; font: inherit; }
.ask button { background: var(--accent); color: #0b1220; border: 0;
              border-radius: 8px; padding: 0.7rem 1.4rem; font: inherit;
              font-weight: 600; cursor: pointer; }
.example { display: inline-block; margin-right: 0.9rem; font-size: 0.85rem; }
.answer-block { margin: 1.25rem 0; }
.answer-block ul { margin: 0.3rem 0; padding-left: 1.1rem; color: #c9c9d0; }
.chart { margin: 1.25rem 0; background: var(--bg); border: 1px solid var(--line);
         border-radius: 10px; padding: 0.75rem; overflow-x: auto; }
.recent { list-style: none; padding: 0; font-size: 0.9rem; }
.recent li { padding: 0.35rem 0; border-bottom: 1px solid #26262c; }
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_analysis_page.py -q`
Expected: PASS, 9 tests

- [ ] **Step 7: Look at it**

```bash
uv run python -m aoi_agent station --port 8000
```
Open `http://127.0.0.1:8000/ask`, ask one of the examples, and check that all
five numbered blocks render and the chart appears.

- [ ] **Step 8: Commit**

```bash
git add src/aoi_agent/station tests/test_analysis_page.py
git commit -m "Add the ask page, with the plan shown beside the answer

Five blocks, numbered in the markup so a reviewer can see none was dropped:
how the question was read, what was called, what was assumed, what it cost, and
the prose. The middle two are there for trust rather than debugging -- a
supervisor can only judge 'open is up' if they can see it was measured against
the fleet average and not against last week.

The chart is inline SVG generated server-side. The station already works with
JavaScript off and vendors its one dependency so it survives a locked-down
shop-floor browser; a CDN charting library would break both for a picture of
six bars.

A test asserts that reopening a stored run never calls the model."
```

---

### Task 8: Streamed progress

Two model calls put an answer 20-25 seconds away. An indeterminate spinner over
that makes the wait feel longer and shows nothing; the same seconds spent
watching tools tick off read as work. It is also the only thing on screen that
shows the branches running concurrently.

The plain `POST /ask` from task 7 stays and stays working. This is an
enhancement layered on top, so the page still functions with JavaScript off.

**Files:**
- Modify: `src/aoi_agent/station/app.py` — add `GET /ask/stream`
- Modify: `src/aoi_agent/station/templates/analysis.html` — progress panel and script
- Test: `tests/test_analysis_stream.py`

**Interfaces:**
- Consumes: `analysis_graph()` (task 7), `answer_question` (task 6)
- Produces: `GET /ask/stream?question=...` returning `text/event-stream` with
  events `plan`, `tool`, `done`, `error`; `done` carries `{"run_id": int}`

- [ ] **Step 1: Write the failing tests**

```python
"""Progress events, and the guarantee that the page works without them."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from aoi_agent.analysis import graph as analysis
from aoi_agent.llm.ollama import ChatResult, Timing
from aoi_agent.station import app as station_app
from aoi_agent.store.models import create_all, make_session_factory

PLAN = {
    "interpretation": "M22 against the fleet",
    "assumptions": ["fleet average"],
    "calls": [
        {"tool": "query_machine_stats", "args": {"defect_type": "open", "days": 7},
         "why": "a"},
        {"tool": "search_standards", "args": {"query": "open", "top_k": 2},
         "why": "b"},
    ],
}
DOMAINS = {"line_id": {"L2"}, "machine_id": {"M22"},
           "defect_type": {"open"}, "max_days": 9}


@dataclass
class StubClient:
    plan: dict = field(default_factory=lambda: PLAN)

    def chat(self, messages, **kwargs) -> ChatResult:
        text = json.dumps(self.plan) if kwargs.get("response_format") else "done"
        return ChatResult(text=text, tool_calls=[], thinking="",
                          timing=Timing(1.0, 0.0, 1.0, 1.0, 10, 10))


@pytest.fixture
def client(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'a.db'}"
    create_all(url)
    monkeypatch.setattr("aoi_agent.store.boards._session_factory",
                        make_session_factory(url))
    for name in ("query_machine_stats", "search_standards"):
        monkeypatch.setitem(analysis.PLANNABLE_TOOLS, name, lambda **kw: {"ok": 1})
    monkeypatch.setattr(station_app, "_analysis_graph",
                        analysis.build_analysis_graph(StubClient(), DOMAINS))
    return TestClient(station_app.app)


def events(body: str) -> list[dict]:
    out = []
    for block in body.strip().split("\n\n"):
        name, payload = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
        if name:
            out.append({"event": name, "data": payload})
    return out


def test_the_stream_reports_the_plan_then_each_tool_then_done(client):
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    names = [e["event"] for e in events(body)]

    assert names[0] == "plan"
    assert names.count("tool") == 2
    assert names[-1] == "done"


def test_the_plan_event_carries_what_will_run_so_the_page_can_list_it(client):
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    first = events(body)[0]["data"]

    assert [c["tool"] for c in first["calls"]] == [
        "query_machine_stats", "search_standards"
    ]
    assert first["interpretation"]


def test_the_done_event_carries_the_run_id_to_navigate_to(client):
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    done = events(body)[-1]["data"]

    assert isinstance(done["run_id"], int)
    assert client.get(f"/ask/{done['run_id']}").status_code == 200


def test_the_run_the_stream_persists_is_the_same_one_the_page_renders(client):
    """The stream must not be a second, parallel execution -- that would double
    the cost and could disagree with itself."""
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    run_id = events(body)[-1]["data"]["run_id"]
    page = client.get(f"/ask/{run_id}").text

    assert "query_machine_stats" in page


def test_a_failing_tool_is_streamed_as_a_failed_tool_event(client, monkeypatch):
    def boom(**kw):
        raise RuntimeError("index unreachable")

    monkeypatch.setitem(analysis.PLANNABLE_TOOLS, "search_standards", boom)
    body = client.get("/ask/stream", params={"question": "M22 正常嗎"}).text
    tools = [e["data"] for e in events(body) if e["event"] == "tool"]

    assert any(t["ok"] is False and "unreachable" in t["error"] for t in tools)


def test_a_rejected_plan_ends_the_stream_with_an_error_event(client, monkeypatch):
    bad = {"interpretation": "i", "assumptions": [],
           "calls": [{"tool": "query_defect_history",
                      "args": {"line_id": "L9"}, "why": "w"}]}
    monkeypatch.setattr(station_app, "_analysis_graph",
                        analysis.build_analysis_graph(StubClient(plan=bad), DOMAINS))
    body = client.get("/ask/stream", params={"question": "L9"}).text
    names = [e["event"] for e in events(body)]

    assert names[-1] == "done", "a rejected plan still produces a viewable run"
    assert any(e["event"] == "error" for e in events(body))


def test_the_form_still_works_with_no_javascript(client):
    """The stream is an enhancement. The station runs on shop-floor browsers."""
    response = client.post("/ask", data={"question": "M22 正常嗎"},
                           follow_redirects=False)
    assert response.status_code == 303
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_analysis_stream.py -q`
Expected: FAIL — 404 on `/ask/stream`

- [ ] **Step 3: Add the endpoint**

In `src/aoi_agent/station/app.py`:

```python
import json

from fastapi.responses import StreamingResponse


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/ask/stream")
def ask_stream(question: str, asked_by: str = "operator"):
    """Run one question, emitting progress as it goes.

    One execution, not two: the same run that produces these events is the run
    that gets persisted, so the page a viewer lands on cannot disagree with the
    progress they just watched.
    """
    if not question.strip():
        raise HTTPException(400, "a question is required")

    def stream():
        graph = analysis_graph()
        state: dict = {}
        try:
            for update in graph.stream(
                {"question": question.strip(), "results": [], "timings_ms": {}},
                stream_mode="updates",
            ):
                for node, payload in update.items():
                    # A node returning an empty dict streams as None. Verified
                    # against LangGraph 1.2 before this plan was written.
                    payload = payload or {}
                    state.update(payload)
                    if node == "plan":
                        plan = payload.get("plan") or {}
                        yield _sse("plan", {
                            "interpretation": plan.get("interpretation", ""),
                            "calls": plan.get("calls", []),
                        })
                        for error in payload.get("plan_errors") or []:
                            yield _sse("error", {"message": error})
                    elif node == "run_tool":
                        for result in payload.get("results") or []:
                            yield _sse("tool", {
                                "tool": result["tool"],
                                "ok": result["ok"],
                                "error": result["error"],
                                "elapsed_ms": result["elapsed_ms"],
                            })
        except Exception as error:  # noqa: BLE001 -- the stream must close cleanly
            yield _sse("error", {"message": f"{type(error).__name__}: {error}"})

        run_id = analysis_store.save_run(
            question=question.strip(),
            plan=state.get("plan"),
            results=state.get("results") or [],
            chart=state.get("chart_spec"),
            answer=state.get("answer", ""),
            timings=state.get("timings_ms") or {},
            refused=bool(state.get("refused")),
            asked_by=asked_by,
        )
        yield _sse("done", {"run_id": run_id})

    return StreamingResponse(stream(), media_type="text/event-stream")
```

- [ ] **Step 4: Add the progress panel to the template**

Insert after the `<form class="ask">` block in `analysis.html`:

```html
<div class="progress" id="progress" hidden>
  <p class="label" id="progress-head">規劃中…</p>
  <ul id="progress-list"></ul>
</div>
```

and in a `{% block script %}` at the end of the file:

```html
<script>
  // Enhancement only: with JavaScript off the form posts and the server
  // renders the finished page. With it on, the same run streams its progress
  // so a 20-second wait reads as work rather than as a hang.
  var form = document.querySelector('.ask');
  form.addEventListener('submit', function (event) {
    var question = form.querySelector('input[name=question]').value.trim();
    if (!question) return;
    event.preventDefault();

    var panel = document.getElementById('progress');
    var list = document.getElementById('progress-list');
    var head = document.getElementById('progress-head');
    panel.hidden = false;
    list.innerHTML = '';
    head.textContent = '規劃中…';

    var source = new EventSource('/ask/stream?question=' + encodeURIComponent(question));
    source.addEventListener('plan', function (e) {
      var plan = JSON.parse(e.data);
      head.textContent = plan.interpretation || '規劃完成';
      plan.calls.forEach(function (call) {
        var li = document.createElement('li');
        li.id = 'tool-' + call.tool;
        li.textContent = '⟳ ' + call.tool;
        list.appendChild(li);
      });
    });
    source.addEventListener('tool', function (e) {
      var result = JSON.parse(e.data);
      var li = document.getElementById('tool-' + result.tool);
      if (!li) return;
      li.textContent = (result.ok ? '✓ ' : '✗ ') + result.tool +
                       '  ' + Math.round(result.elapsed_ms) + 'ms' +
                       (result.ok ? '' : '  ' + result.error);
    });
    source.addEventListener('error', function (e) {
      var li = document.createElement('li');
      try { li.textContent = '✗ ' + JSON.parse(e.data).message; }
      catch (_) { li.textContent = '✗ 連線中斷'; }
      list.appendChild(li);
    });
    source.addEventListener('done', function (e) {
      source.close();
      window.location = '/ask/' + JSON.parse(e.data).run_id;
    });
  });
</script>
```

Append to `style.css`:

```css
.progress { margin: 1rem 0; padding: 0.9rem 1.1rem; background: var(--bg);
            border: 1px solid var(--line); border-radius: 10px; }
.progress ul { list-style: none; padding: 0; margin: 0.5rem 0 0;
               font-family: var(--mono); font-size: 0.85rem; line-height: 1.9; }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_analysis_stream.py -q`
Expected: PASS, 7 tests

- [ ] **Step 6: Watch it once with a real model**

```bash
uv run python -m aoi_agent station --port 8000
```
Ask an example question and confirm the tools tick off individually before the
page navigates.

- [ ] **Step 7: Commit**

```bash
git add src/aoi_agent/station tests/test_analysis_stream.py
git commit -m "Stream the plan and each tool as they complete

Two model calls put an answer 20-25 seconds away. An indeterminate spinner over
that wait makes it feel longer and shows nothing, while the same seconds spent
watching tools tick off read as work. It is also the only thing on screen that
shows the branches finishing concurrently rather than in turn.

One execution, not two: the run that emits these events is the run that gets
persisted, so the page a viewer lands on cannot disagree with the progress they
just watched.

The plain form post stays and stays tested. This is an enhancement layered on
top, and the station still works on a browser with JavaScript disabled."
```

---

### Task 9: Measuring it

The tools are deterministic, so a correct plan yields correct data by
construction and the errors live in the plan. That is what makes this
measurable at all: ground truth for *plans* is hand-buildable where ground
truth for answers is not.

**Files:**
- Create: `tests/fixtures/analysis_questions.json`
- Create: `scripts/analysis_eval.py`
- Test: `tests/test_analysis_eval.py`

**Interfaces:**
- Consumes: `build_analysis_graph` (task 5), `validate_plan`, `store_domains` (task 1)
- Produces: `score_plan(plan, expected) -> dict`, `load_questions(path) -> list[dict]`, `main()` appending to `docs/benchmarks.md`

- [ ] **Step 1: Write the question set**

Create `tests/fixtures/analysis_questions.json`. Twenty questions, written
before reading the few-shot examples wherever possible — see the caveat in the
spec's evaluation section, which this does not fully solve.

```json
[
  {"question": "L2-M22 的 open 是不是比其他機台高？",
   "expect_refusal": false, "expect_tools": ["query_machine_stats"]},
  {"question": "M22 的 open 高不高，驗收標準怎麼說？",
   "expect_refusal": false,
   "expect_tools": ["query_machine_stats", "search_standards"]},
  {"question": "三條線的缺陷組成比一比",
   "expect_refusal": false, "expect_tools": ["query_defect_history"]},
  {"question": "20085294 這片板子是哪個批號哪台機器做的？",
   "expect_refusal": false, "expect_tools": ["query_board_context"]},
  {"question": "20085294 上面 AOI 標了哪些區域？",
   "expect_refusal": false, "expect_tools": ["list_candidates"]},
  {"question": "short 的驗收標準是什麼？",
   "expect_refusal": false, "expect_tools": ["search_standards"]},
  {"question": "L1 最近的缺陷數量如何？",
   "expect_refusal": false, "expect_tools": ["query_defect_history"]},
  {"question": "mousebite 在哪一台機器上最常出現？",
   "expect_refusal": false, "expect_tools": ["query_machine_stats"]},
  {"question": "L3 的 spur 多不多，標準怎麼規定？",
   "expect_refusal": false,
   "expect_tools": ["query_defect_history", "search_standards"]},
  {"question": "哪一台機器的缺陷率最高？",
   "expect_refusal": false, "expect_tools": ["query_machine_stats"]},
  {"question": "去年同期的 open 缺陷率是多少？", "expect_refusal": true},
  {"question": "2023 年第一季到現在的趨勢圖", "expect_refusal": true},
  {"question": "下個月會有多少缺陷？", "expect_refusal": true},
  {"question": "產線最近怎麼樣？", "expect_refusal": true},
  {"question": "幫我看一下", "expect_refusal": true},
  {"question": "L4 的狀況如何？", "expect_refusal": true},
  {"question": "把 candidates 資料表刪掉", "expect_refusal": true},
  {"question": "為什麼 M22 最近怪怪的？",
   "expect_refusal": false, "expect_assumption_about_cause": true},
  {"question": "L1 的缺陷是不是變多了？",
   "expect_refusal": false, "expect_assumptions": true},
  {"question": "M31 跟 M32 誰比較穩？",
   "expect_refusal": false, "expect_assumptions": true}
]
```

- [ ] **Step 2: Write the failing tests**

```python
"""Scoring a plan against what it should have done."""

from __future__ import annotations

import json
from pathlib import Path

from analysis_eval import load_questions, score_plan

FIXTURE = Path(__file__).parent / "fixtures" / "analysis_questions.json"


def plan(*tools, assumptions=None):
    return {
        "interpretation": "i",
        "assumptions": assumptions if assumptions is not None else [],
        "calls": [{"tool": t, "args": {}, "why": "w"} for t in tools],
    }


def test_the_question_set_loads_and_covers_both_outcomes():
    questions = load_questions(FIXTURE)

    assert len(questions) >= 20
    assert any(q["expect_refusal"] for q in questions)
    assert any(not q["expect_refusal"] for q in questions)


def test_calling_the_expected_tools_scores_a_hit():
    result = score_plan(plan("query_machine_stats"),
                        {"expect_refusal": False,
                         "expect_tools": ["query_machine_stats"]})
    assert result["ok"] is True


def test_a_missing_expected_tool_is_a_miss():
    result = score_plan(plan("query_defect_history"),
                        {"expect_refusal": False,
                         "expect_tools": ["query_machine_stats"]})
    assert result["ok"] is False
    assert "query_machine_stats" in result["reason"]


def test_extra_tools_are_allowed():
    """Gathering more context than the minimum is not an error. Gathering less
    is, because the answer then rests on data that was never fetched."""
    result = score_plan(plan("query_machine_stats", "search_standards"),
                        {"expect_refusal": False,
                         "expect_tools": ["query_machine_stats"]})
    assert result["ok"] is True


def test_refusing_when_a_refusal_was_expected_scores_a_hit():
    result = score_plan(plan(), {"expect_refusal": True})
    assert result["ok"] is True


def test_answering_a_question_that_should_be_refused_is_a_miss():
    """The dangerous direction. A system that answers everything is worse on a
    factory floor than one that says it cannot."""
    result = score_plan(plan("query_defect_history"), {"expect_refusal": True})
    assert result["ok"] is False
    assert "refus" in result["reason"].lower()


def test_a_causal_question_must_disclaim_cause_in_its_assumptions():
    without = score_plan(plan("query_defect_history"),
                         {"expect_refusal": False,
                          "expect_assumption_about_cause": True})
    with_note = score_plan(
        plan("query_defect_history",
             assumptions=["This shows association, not cause."]),
        {"expect_refusal": False, "expect_assumption_about_cause": True},
    )

    assert without["ok"] is False
    assert with_note["ok"] is True


def test_a_comparison_with_no_stated_baseline_is_a_miss():
    result = score_plan(plan("query_defect_history"),
                        {"expect_refusal": False, "expect_assumptions": True})
    assert result["ok"] is False
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_analysis_eval.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis_eval'`

(`pyproject.toml` already puts `scripts` on the path via `pythonpath = ["src", "tests"]`; add `"scripts"` to that list if the import does not resolve.)

- [ ] **Step 4: Write the script**

Create `scripts/analysis_eval.py`:

```python
"""Does the planner plan the right thing, and refuse the right things?

The tools are deterministic, so a correct plan yields correct data by
construction. The errors live in the plan, which is why this scores plans rather
than answers -- ground truth for a plan is hand-writable, and ground truth for a
paragraph is not.

Three measures. Plan accuracy is the obvious one. Refusal accuracy matters more
than it looks: a system that answers everything is more dangerous on a factory
floor than one that says it cannot, and nothing else in this project checks it.
Determinism is here because a supervisor who screenshots a chart should be able
to ask the same question tomorrow and recognise the answer.

A weakness the numbers do not show: the same person wrote the few-shot examples
and these questions, so the questions tend towards the shapes the prompt was
built for. Partly mitigated by writing them before re-reading the examples;
not solved.

    uv run python scripts/analysis_eval.py --repeats 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.analysis.graph import build_analysis_graph  # noqa: E402
from aoi_agent.analysis.plan import store_domains  # noqa: E402
from aoi_agent.graph.flow import DEFAULT_MODEL  # noqa: E402
from aoi_agent.llm.ollama import OllamaClient  # noqa: E402

QUESTIONS = Path(__file__).resolve().parents[1] / "tests/fixtures/analysis_questions.json"

CAUSE_WORDS = ("cause", "causal", "causation", "association", "correlat", "因果", "關聯")


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return json.loads(Path(path).read_text())


def score_plan(plan: dict, expected: dict) -> dict:
    """Did this plan do what the question needed? One reason if not."""
    calls = plan.get("calls") or []
    refused = not calls

    if expected.get("expect_refusal"):
        if refused:
            return {"ok": True, "reason": ""}
        return {"ok": False,
                "reason": f"should have refused; planned {[c['tool'] for c in calls]}"}

    if refused:
        return {"ok": False, "reason": "refused a question it should have answered"}

    called = {c["tool"] for c in calls}
    missing = set(expected.get("expect_tools") or []) - called
    if missing:
        # Extra tools are fine -- more context is not an error. Missing ones are
        # not: the answer would rest on data nobody fetched.
        return {"ok": False, "reason": f"never called {sorted(missing)}"}

    assumptions = " ".join(plan.get("assumptions") or []).lower()
    if expected.get("expect_assumption_about_cause") and not any(
        word in assumptions for word in CAUSE_WORDS
    ):
        return {"ok": False, "reason": "asked why, and never disclaimed cause"}

    if expected.get("expect_assumptions") and not (plan.get("assumptions") or []):
        return {"ok": False, "reason": "compared without stating a baseline"}

    return {"ok": True, "reason": ""}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repeats", type=int, default=3,
                        help="times to re-ask each question, for determinism")
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    questions = load_questions()
    domains = store_domains()
    graph = build_analysis_graph(OllamaClient(args.model, timeout=180.0), domains)

    scored, stable = [], []
    for position, item in enumerate(questions, 1):
        plans = []
        for _ in range(args.repeats):
            state = graph.invoke(
                {"question": item["question"], "results": [], "timings_ms": {}}
            )
            plans.append(state.get("plan") or {"calls": []})

        result = score_plan(plans[0], item)
        scored.append({**item, **result})
        signatures = {
            json.dumps(sorted(c["tool"] for c in (p.get("calls") or []))) for p in plans
        }
        stable.append(len(signatures) == 1)

        print(f"  [{position:>3}/{len(questions)}] "
              f"{'ok  ' if result['ok'] else 'MISS'} "
              f"{'stable' if stable[-1] else 'VARIES'}  {item['question'][:40]}  "
              f"{result['reason']}", flush=True)

    answerable = [s for s in scored if not s.get("expect_refusal")]
    refusable = [s for s in scored if s.get("expect_refusal")]
    hit = lambda rows: sum(1 for r in rows if r["ok"])  # noqa: E731

    lines = [
        "",
        "### Analysis planner — does it plan the right lookups, and refuse the rest?",
        "",
        f"`{args.model}`, {len(questions)} hand-written questions, each asked "
        f"{args.repeats} times. Plans are scored, not answers: the tools are "
        f"deterministic, so a correct plan yields correct data by construction and "
        f"the errors live in the plan.",
        "",
        "| | questions | correct |",
        "|---|---|---|",
        f"| should answer | {len(answerable)} | {hit(answerable)}/{len(answerable)} |",
        f"| should refuse | {len(refusable)} | {hit(refusable)}/{len(refusable)} |",
        f"| determinism | {len(stable)} | {sum(stable)}/{len(stable)} identical "
        f"across {args.repeats} runs |",
        "",
        "Misses:",
        "",
    ]
    misses = [s for s in scored if not s["ok"]]
    lines += [f"- {s['question']} — {s['reason']}" for s in misses] or ["- none"]
    lines += [
        "",
        "Refusal accuracy carries more weight than the count suggests. A planner "
        "that answers everything is more dangerous on a line than one that says it "
        "cannot, and nothing else in this project measures that.",
        "",
        "Not covered: synthesis can describe correct data incorrectly, and the "
        "same person wrote the examples and these questions. Both are recorded in "
        "the design rather than solved.",
    ]

    report = "\n".join(lines)
    print("\n" + report)
    if not args.dry_run:
        with args.out.open("a") as handle:
            handle.write(report + "\n")
        print(f"\nappended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_analysis_eval.py -q`
Expected: PASS, 8 tests

- [ ] **Step 6: Run the evaluation for real**

Check `ollama ps` first — a contended GPU makes this slow, though not wrong,
since nothing here is a timing measurement.

```bash
uv run python scripts/analysis_eval.py --repeats 3
```

Expect roughly `20 questions × 3 repeats × ~12s ≈ 12 minutes`.

- [ ] **Step 7: Run the whole suite and commit**

```bash
uv run pytest -q
git add scripts/analysis_eval.py tests/test_analysis_eval.py \
        tests/fixtures/analysis_questions.json docs/benchmarks.md
git commit -m "Measure the planner: right lookups, right refusals, stable plans

Scores plans rather than answers. The tools are deterministic, so a correct
plan yields correct data by construction and every error that matters lives in
the plan -- which also makes this measurable at all, since ground truth for a
plan is hand-writable and ground truth for a paragraph is not.

Refusal accuracy is tracked separately and carries more weight than its share
of the question set. A planner that answers everything is more dangerous on a
line than one that says it cannot, and nothing else in this project checks it.

Determinism is measured because a supervisor who screenshots a chart should
recognise tomorrow's answer to the same question.

Extra tools score as a hit and missing ones as a miss: gathering more context
than needed is not an error, while gathering less leaves the answer resting on
data nobody fetched."
```

---

## Done when

- `uv run pytest -q` green, ~150 tests
- `uv run python -m aoi_agent station` serves `/ask`; an example question returns
  all five blocks, a chart, and streamed progress
- `docs/benchmarks.md` carries a planner section with all three measures
- A stored run reopens and redraws with the model stopped
- `uv run python scripts/check_skill_freshness.py` still passes

## Deliberately not in this plan

- **Re-planning after results** (approach C in the spec). Needs a round cap,
  multiplies the model cost, and its own decision — "do I have enough" — is an
  LLM judgement this project does not adopt without measuring against a simpler
  rule first.
- **Widening the seeded time span.** Independent of everything here and gates
  only the multi-period questions already out of scope. Section 4 of the spec.
- **Operator authentication.** A precondition for exposing this page beyond
  localhost, and its own piece of work. Recorded in the spec's open questions.
- **The animated graph view.** Task 8 records what it would need; building it is
  a rendering exercise, not an architectural one.
