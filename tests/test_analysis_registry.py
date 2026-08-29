"""What may be registered as a plannable tool, and why a signature cannot say.

The no-SQL invariant reads: typed parameters over a fixed query set, because a
valid but semantically wrong query returns a plausible number and gets acted
on. Everything enforcing it looked at the *arguments a plan passes* --
`tests/test_analysis_plan.py` refuses an unknown tool, an unknown argument and
an out-of-domain value. None of it looked at the *surface the registry
offers*, so `run_query(sql: str)` added to `PLANNABLE_TOOLS` passed all five:
`sql` is a known argument of a known tool, and the value has no domain to be
outside of.

The hard part is that `search_standards(query: str, top_k=3)` is a real
registered tool that takes free text, and it is the same signature. Nothing in
`(x: str)` distinguishes prose handed to an embedding index from syntax handed
to a query engine, so a rule about types or parameter names would ban a working
feature and stop nothing -- rename `sql` to `query` and it is through.

So the discrimination is declared per parameter and checked at registration,
and the declarations are backed where they can be: a corpus has to be one the
system actually has, the tool's module and the corpus module are both checked
for a route to the store, and the tool's own body is checked for SQL built out
of a string. What is left is a declaration a determined author could falsify,
which is a line of source in the registry rather than a silent pass.
"""

from __future__ import annotations

import pytest

from aoi_agent.analysis.plan import (
    PLANNABLE_TOOLS,
    REGISTRATIONS,
    Registration,
    UnregistrableTool,
    registration_errors,
    registry,
)
from aoi_agent.mcp_servers.production import query_board_context
from aoi_agent.mcp_servers.standards import search_standards


def run_query(sql: str) -> dict:
    """The tool this whole file exists to refuse.

    Written the way somebody would actually write it -- a session, a string, a
    result -- so that what is rejected is a working text-to-SQL tool and not a
    stub with the right name.
    """
    from sqlalchemy import text

    from aoi_agent.store.boards import session_factory

    with session_factory()() as session:
        return {"rows": [tuple(row) for row in session.execute(text(sql))]}


def test_the_registry_as_it_stands_accounts_for_every_parameter_it_exposes():
    """The gate, pointed at the real registry. If this fails, a tool has grown
    a parameter nobody classified -- which is the state the audit found."""
    assert registration_errors(REGISTRATIONS) == []


def test_a_tool_taking_free_text_query_language_cannot_be_registered():
    """The mutation the invariant audit recorded as passing 691 tests."""
    errors = registration_errors((Registration(run_query),))

    assert any("sql" in error for error in errors), errors
    with pytest.raises(UnregistrableTool, match="sql"):
        registry((Registration(run_query),))


def test_calling_the_query_language_an_identifier_does_not_get_it_registered():
    """`identifiers` is the account for a string used as a value -- a lot, a
    board -- and it is the obvious place to hide a query language, since it is
    a declaration and declarations can lie. It costs something to lie here: the
    tool's own body still has to contain no SQL built from a string, and a tool
    that executes its own parameter is exactly that."""
    errors = registration_errors(
        (Registration(run_query, identifiers=frozenset({"sql"})),)
    )

    assert any("text()" in error or "text-to-SQL" in error for error in errors), errors


def test_free_text_has_to_name_a_corpus_this_system_actually_has():
    errors = registration_errors(
        (Registration(run_query, retrieval={"sql": "the production database"}),)
    )

    assert any("not a document corpus" in error for error in errors), errors


def test_free_text_declared_on_a_module_that_can_reach_the_store_is_refused():
    """The declaration is not taken on trust. `query_board_context` lives in
    the production server, which imports SQLAlchemy and the board models;
    calling its `board` parameter a retrieval query is refused on the strength
    of what its module can reach, not on the strength of the claim."""
    errors = registration_errors(
        (Registration(query_board_context, retrieval={"board": "standards"}),)
    )

    assert any("imports aoi_agent.store.boards" in error for error in errors), errors


def test_search_standards_still_registers_with_its_free_text_intact():
    """The feature this check must not break. `query` is arbitrary prose, it
    is the whole point of the tool, and it is admissible because the text goes
    to a Chroma index over markdown and comes back as passages with their
    document and heading on them -- a wrong passage is visibly the wrong
    passage, where a wrong number is not."""
    registration = Registration(search_standards, retrieval={"query": "standards"})

    assert registration_errors((registration,)) == []
    assert registry((registration,)) == {"search_standards": search_standards}
    assert PLANNABLE_TOOLS["search_standards"] is search_standards


def test_a_catch_all_signature_cannot_be_registered():
    """`**kwargs` is not a parameter surface, it is the absence of one: every
    argument name is valid, so nothing downstream can refuse anything."""

    def anything(**kwargs) -> dict:
        return {}

    errors = registration_errors((Registration(anything),))

    assert any("surface is open" in error for error in errors), errors


def test_an_unreadable_annotation_is_treated_as_text_rather_than_waved_through():
    """A parameter nobody annotated is the case to be careful about."""

    def whatever(anything) -> dict:
        return {}

    errors = registration_errors((Registration(whatever),))

    assert any("arbitrary text" in error for error in errors), errors


def test_a_declaration_naming_a_parameter_the_tool_does_not_have_is_refused():
    """A registration that has drifted from its tool. Left alone it reads as
    an account of a parameter surface that no longer exists."""
    errors = registration_errors(
        (Registration(search_standards, retrieval={"prompt": "standards"}),)
    )

    assert any("no parameter 'prompt'" in error for error in errors), errors


@pytest.mark.parametrize("tool", sorted(PLANNABLE_TOOLS))
def test_no_registered_tool_builds_sql_out_of_a_string(tool):
    """Said directly, and per tool, so the failure names the one that changed.
    The check the registration gate runs over each body, run again here as its
    own assertion rather than as a clause inside the gate's."""
    registration = next(r for r in REGISTRATIONS if r.name == tool)
    from aoi_agent.analysis.plan import _sql_from_a_string

    assert _sql_from_a_string(registration) == []



# --- the sql account, added 2026-08-29 ---------------------------------------

def test_the_read_only_sql_tool_registers_under_the_sql_account():
    from aoi_agent.mcp_servers.sql_readonly import run_sql

    registration = Registration(run_sql, sql={"sql": "production_readonly"})
    assert registration_errors((registration,)) == []


def test_a_query_language_declared_guarded_but_run_through_text_is_refused():
    """The declaration is checked against the body: `run_query` hands its
    parameter to `text()`, not to the guard, and saying otherwise does not
    make it so."""
    errors = registration_errors((Registration(run_query, sql={"sql": "production_readonly"}),))
    assert any("never passes 'sql' to guarded_select" in e for e in errors), errors
    assert any("text()" in e for e in errors), errors


def test_a_guarded_parameter_may_reach_nothing_but_the_guard():
    """Passing it to the guard *and* somewhere else is refused too -- a log
    line, a helper, a second query. One door."""
    from aoi_agent.analysis.sql_guard import guarded_select

    def leaky(sql: str) -> dict:
        print(sql)
        return guarded_select(sql)

    errors = registration_errors((Registration(leaky, sql={"sql": "production_readonly"}),))
    assert any("somewhere other than as the argument of guarded_select()" in e for e in errors), errors


def test_a_guard_that_does_not_exist_cannot_be_named():
    from aoi_agent.analysis.sql_guard import guarded_select

    def fine(sql: str) -> dict:
        return guarded_select(sql)

    errors = registration_errors((Registration(fine, sql={"sql": "the real database"}),))
    assert any("not a guard this system has" in e for e in errors), errors


def test_the_sql_tool_can_be_left_out_for_the_control_arm(monkeypatch):
    from aoi_agent.analysis import plan as plan_module

    monkeypatch.setenv(plan_module.SQL_TOOL_ENV, "0")
    assert plan_module.sql_tool_enabled() is False
    monkeypatch.setenv(plan_module.SQL_TOOL_ENV, "1")
    assert plan_module.sql_tool_enabled() is True
