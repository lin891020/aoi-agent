"""The two string tables, and the one thing that keeps them from drifting.

A translation table rots in a particular way: someone changes the Chinese, the
English still renders, nothing fails, and the two pages quietly say different
things. Nobody notices until a reader who only has the second one is confused
by it. So the parity of the key sets is a test, not a convention.
"""

from __future__ import annotations

import pytest

from aoi_agent.i18n import (
    DEFAULT_LOCALE,
    LOCALES,
    STRINGS,
    normalise,
    translate,
)


def test_every_locale_carries_exactly_the_same_keys():
    """The only mechanism standing between "changed the Chinese" and "forgot
    the English"."""
    reference = set(STRINGS[DEFAULT_LOCALE])

    for locale in LOCALES:
        missing = reference - set(STRINGS[locale])
        extra = set(STRINGS[locale]) - reference
        assert not missing, f"{locale} is missing {sorted(missing)}"
        assert not extra, f"{locale} has {sorted(extra)} that {DEFAULT_LOCALE} lacks"


def test_a_placeholder_in_one_language_is_a_placeholder_in_the_other():
    """`Line {line_id}` and a Chinese string that forgot `{line_id}` both
    render, and only one of them says which line."""
    import re

    fields = lambda text: set(re.findall(r"\{(\w+)\}", text))  # noqa: E731

    for key, template in STRINGS[DEFAULT_LOCALE].items():
        for locale in LOCALES:
            assert fields(template) == fields(STRINGS[locale][key]), (
                f"{key}: {DEFAULT_LOCALE} takes {fields(template)}, "
                f"{locale} takes {fields(STRINGS[locale][key])}"
            )


def test_the_shop_floor_language_is_the_default():
    """The line this is built for reads Traditional Chinese. English is the
    second language here, and the default is where that gets said."""
    assert DEFAULT_LOCALE == "zh-TW"
    assert normalise(None) == "zh-TW"
    assert normalise("de") == "zh-TW"
    assert normalise("en") == "en"


@pytest.mark.parametrize("locale", LOCALES)
def test_a_missing_key_renders_as_itself_rather_than_raising(locale):
    """A page whose figures are all correct should not 500 over a heading, and
    the key on screen names the thing to fix. The parity test above is what
    makes this branch unreachable in practice."""
    assert translate("chart.title.no_such_thing", locale) == "chart.title.no_such_thing"


@pytest.mark.parametrize("locale", LOCALES)
def test_a_missing_argument_renders_the_template_rather_than_raising(locale):
    """Same argument one level down: `{line_id}` on screen is a visible fault,
    a `KeyError` is a lost page."""
    rendered = translate("chart.series.line_id", locale)

    assert "{line_id}" in rendered


# ---------------------------------------------------------------------------
# The switch, and what it must not touch
# ---------------------------------------------------------------------------

import pathlib  # noqa: E402

import pytest  # noqa: E402,F811
from fastapi.testclient import TestClient  # noqa: E402

from aoi_agent.llm.ollama import ChatResult, Timing  # noqa: E402
from aoi_agent.station import app as station_app  # noqa: E402
from aoi_agent.store.models import create_all, make_session_factory  # noqa: E402
from conftest import read_in, sign_in  # noqa: E402

TEMPLATES = pathlib.Path(station_app.__file__).parent / "templates"


@pytest.fixture
def client(tmp_path, monkeypatch, operators):
    """A station on a store of its own.

    Without the redirect below this fixture reached the developer's real
    `data/aoi_agent.db` -- which passed until the schema moved under it, and
    would have been writing test rows into a store holding operator
    corrections the whole time.
    """
    url = f"sqlite:///{tmp_path / 'a.db'}"
    create_all(url)
    monkeypatch.setattr("aoi_agent.store.boards._session_factory",
                        make_session_factory(url))
    return sign_in(TestClient(station_app.app))


def test_the_station_opens_in_the_language_the_line_reads(client):
    page = client.get("/").text

    assert 'lang="zh-TW"' in page
    assert "待人工複判的區域" in page


def test_the_switch_changes_the_language_and_comes_back_to_the_page(client):
    client.get("/locale/en?next=/corrections", follow_redirects=False)
    page = client.get("/corrections").text

    assert 'lang="en"' in page
    assert "Where operators overruled the model" in page


def test_the_choice_outlives_a_sign_out(client):
    """A cookie, not the session. How a person reads the screen is not a claim
    about who they are, and on a shared terminal the preference should not be
    revoked by the next sign-out."""
    client.get("/locale/en", follow_redirects=False)
    client.post("/logout", follow_redirects=False)

    assert 'lang="en"' in client.get("/login").text


def test_the_sign_in_page_can_be_switched_without_a_session():
    """Somebody who cannot read the login form cannot sign in to fix that, so
    `/locale` is public and the control is on the page."""
    fresh = TestClient(station_app.app)
    assert fresh.get("/locale/en", follow_redirects=False).status_code == 303

    page = fresh.get("/login").text
    assert 'lang="en"' in page and "Sign in" in page


def test_an_unknown_language_leaves_a_working_station(client):
    """Reachable by typing a URL. A wrong one should not 404 a shop-floor
    terminal out of its queue."""
    client.get("/locale/klingon", follow_redirects=False)

    assert 'lang="zh-TW"' in client.get("/").text


def test_the_switch_cannot_be_used_to_send_someone_off_the_station(client):
    """`next` goes through `_safe_next`, like the sign-in form's does."""
    response = client.get(
        "/locale/en?next=https://example.com/x", follow_redirects=False)

    assert response.headers["location"] == "/"


@pytest.mark.parametrize("path", ["/", "/corrections", "/ask", "/login"])
def test_every_page_offers_the_other_language(client, path):
    page = client.get(path).text

    assert 'class="locale"' in page, f"{path} has no language switch"
    assert "/locale/en" in page


def test_no_template_carries_a_user_facing_string_of_its_own():
    """The scan that stops the tables being bypassed.

    A sentence typed straight into a template renders in one language whatever
    the switch says, and the key-parity test cannot see it -- there is no key
    to be missing. Latin-1 words are everywhere in this codebase legitimately
    (attributes, class names, `mono` identifiers), so what this looks for is
    CJK text outside a `t()` call: a Chinese sentence in a template is
    necessarily untranslated, because the English table is where its twin would
    have to live.
    """
    import re

    offenders = []
    for template in sorted(TEMPLATES.glob("*.html")):
        source = template.read_text()
        # Jinja comments hold prose deliberately -- they render nothing.
        source = re.sub(r"\{#.*?#\}", "", source, flags=re.S)
        # And so does the argument of a `t()` call... which is a key, ASCII by
        # construction, so anything CJK left after stripping comments is text.
        for line_no, line in enumerate(source.splitlines(), 1):
            if re.search(r"[一-鿿]", line):
                offenders.append(f"{template.name}:{line_no}: {line.strip()[:70]}")

    assert not offenders, "untranslated text in a template:\n  " + "\n  ".join(offenders)


def test_every_plannable_tool_has_a_readable_name():
    """Registering a tool is what puts it on the `/ask` page, so the tables
    have to keep up with the registry rather than with somebody's memory. A
    missing key renders as `tool.query_solder_paste` in a column a supervisor
    reads."""
    from aoi_agent.analysis.plan import PLANNABLE_TOOLS

    for name in PLANNABLE_TOOLS:
        for locale in LOCALES:
            key = f"tool.{name}"
            assert STRINGS[locale][key] != key, f"{key} missing from {locale}"


def test_the_registry_name_stays_on_the_page_beside_the_readable_one(
    client, monkeypatch
):
    """The readable name is an addition, not a replacement. `query_defect_history`
    is what the plan called and what the validator checked a signature against;
    a page showing only a friendly label has swapped the auditable half for
    decoration."""
    from aoi_agent.store import analysis as analysis_store

    run_id = analysis_store.save_run(
        question="q",
        plan={"interpretation": "i", "assumptions": [], "calls": []},
        results=[{"tool": "query_defect_history", "args": {}, "ok": True,
                  "data": {"by_class": {"open": 1}}, "error": None,
                  "elapsed_ms": 1.0, "why": "w", "position": 0}],
        chart=None, answer="a", timings={}, refused=False, asked_by="tester",
    )
    page = read_in(client, "zh-TW").get(f"/ask/{run_id}").text

    assert "缺陷歷史" in page
    assert "query_defect_history" in page


# ---------------------------------------------------------------------------
# What the switch must not rewrite
# ---------------------------------------------------------------------------

def _asked_in(
    lang: str, question: str = "比較三條線", results: list | None = None,
    refused: bool = False,
) -> int:
    from aoi_agent.store import analysis as analysis_store

    return analysis_store.save_run(
        question=question,
        plan={"interpretation": "PLAN-PROSE", "assumptions": ["ASSUMED-PROSE"],
              "calls": []},
        results=results or [], chart=None, answer="ANSWER-PROSE", timings={},
        refused=refused, asked_by="mike", asked_lang=lang,
    )


# One tool result, as the store holds it. The figure is what the rewrite must
# be written from; the hidden key is what it must never be shown.
A_RESULT = [{
    "tool": "list_candidates", "args": {"board": "20085294"}, "why": "w",
    "position": 0, "ok": True, "elapsed_ms": 1.0, "error": None,
    "data": {"regions": [{"index": 1, "score": 0.4711, "ground_truth": "ANSWER-KEY"}]},
}]


class RewritingClient:
    """Answers the synthesis call and keeps what it was shown."""

    def __init__(self, answer: str = "REWRITTEN-PROSE"):
        self.answer = answer
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs) -> ChatResult:
        self.calls.append(messages)
        return ChatResult(text=self.answer, tool_calls=[], thinking="",
                          timing=Timing(1.0, 0.0, 1.0, 1.0, 10, 10))


class Exploding:
    def chat(self, *a, **k):
        raise AssertionError("a stored run must render without the model")


def test_the_question_is_never_rewritten_by_the_switch(client):
    """The record says what was asked. Translating it puts a question nobody
    asked next to the name of the person who asked -- and `asked_by` is what
    makes that row worth keeping."""
    run_id = _asked_in("zh-TW", question="L2-M22 的 open 是不是不尋常？")

    for locale in ("zh-TW", "en"):
        page = read_in(client, locale).get(f"/ask/{run_id}").text
        assert "L2-M22 的 open 是不是不尋常？" in page


def test_the_plan_sections_keep_the_language_they_were_written_in(client):
    """`interpretation` and the assumptions came out of the planning call, and
    the planning call is not made again."""
    run_id = _asked_in("zh-TW")
    page = read_in(client, "en").get(f"/ask/{run_id}").text

    assert "PLAN-PROSE" in page
    assert "ASSUMED-PROSE" in page


def test_a_section_the_switch_does_not_touch_says_so(client):
    run_id = _asked_in("zh-TW")

    english = read_in(client, "en").get(f"/ask/{run_id}").text
    assert "recorded in the language it was asked in" in english

    chinese = read_in(client, "zh-TW").get(f"/ask/{run_id}").text
    assert "以提問時的語言記錄" not in chinese, (
        "a run asked in this language has nothing to explain"
    )


def test_a_run_from_before_the_column_is_labelled_rather_than_claimed(client):
    """`unrecorded` is not the language being read either, so the badge is
    shown -- and "recorded in the language it was asked in" is exactly true of
    a row whose language nobody recorded."""
    from aoi_agent.provenance import UNRECORDED

    run_id = _asked_in(UNRECORDED)

    for locale in ("zh-TW", "en"):
        page = read_in(client, locale).get(f"/ask/{run_id}").text
        assert STRINGS[locale]["analysis.as_asked"] in page


# ---------------------------------------------------------------------------
# The one thing the switch may re-derive, and how
# ---------------------------------------------------------------------------

def test_the_switch_offers_to_write_the_answer_again_rather_than_doing_it(client, monkeypatch):
    """Opening a stored run in the other language shows the original answer,
    badged, with the button that writes this language -- and calls no model.
    The rewrite costs a synthesis call, so it is a POST and not a side effect
    of a GET. Until 2026-09-02 the page had the badge and neither the button
    nor the rewrite: the docs described a path nothing called."""
    monkeypatch.setattr(station_app, "_analysis_client", Exploding())
    run_id = _asked_in("zh-TW", results=A_RESULT)

    english = read_in(client, "en").get(f"/ask/{run_id}").text
    assert "ANSWER-PROSE" in english
    assert STRINGS["en"]["analysis.answer.rewrite"] in english
    assert f'action="/ask/{run_id}/answer"' in english

    chinese = read_in(client, "zh-TW").get(f"/ask/{run_id}").text
    assert STRINGS["zh-TW"]["analysis.answer.rewrite"] not in chinese, (
        "a run asked in this language has its answer already"
    )


def test_writing_again_goes_down_the_synthesis_path_from_the_stored_results(client, monkeypatch):
    """One model call, shown the stored results and not the stored prose;
    the hidden key filtered the way the first pass filters it; the result kept
    beside the original rather than over it, and the page then shows each
    language its own answer under a badge saying which it is."""
    model = RewritingClient("REWRITTEN-PROSE")
    monkeypatch.setattr(station_app, "_analysis_client", model)
    run_id = _asked_in("zh-TW", results=A_RESULT)

    response = read_in(client, "en").post(f"/ask/{run_id}/answer", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == f"/ask/{run_id}"

    assert len(model.calls) == 1
    shown = "\n".join(m["content"] for m in model.calls[0])
    assert "0.4711" in shown, "written from the results"
    assert "ANSWER-PROSE" not in shown, "not from the prose -- that would be a translation"
    assert "ANSWER-KEY" not in shown and "ground_truth" not in shown

    english = read_in(client, "en").get(f"/ask/{run_id}").text
    assert "REWRITTEN-PROSE" in english
    assert "ANSWER-PROSE" not in english
    assert STRINGS["en"]["analysis.answer.rewritten"] in english
    assert STRINGS["en"]["analysis.answer.rewrite"] not in english, "nothing left to write"

    chinese = read_in(client, "zh-TW").get(f"/ask/{run_id}").text
    assert "ANSWER-PROSE" in chinese, "the original is kept"
    assert "REWRITTEN-PROSE" not in chinese
    assert STRINGS["zh-TW"]["analysis.answer.rewritten"] not in chinese


def test_a_language_already_held_is_not_written_twice(client, monkeypatch):
    """The stored answer is the one whose figures were checked. A second press
    -- or a refresh of the POST -- redirects and calls nothing."""
    model = RewritingClient()
    monkeypatch.setattr(station_app, "_analysis_client", model)
    run_id = _asked_in("zh-TW", results=A_RESULT)
    english = read_in(client, "en")

    english.post(f"/ask/{run_id}/answer", follow_redirects=False)
    monkeypatch.setattr(station_app, "_analysis_client", Exploding())
    second = english.post(f"/ask/{run_id}/answer", follow_redirects=False)
    assert second.status_code == 303
    assert len(model.calls) == 1

    # The language it was asked in is held from the start; asking for it
    # again is the same no-op.
    third = read_in(client, "zh-TW").post(f"/ask/{run_id}/answer", follow_redirects=False)
    assert third.status_code == 303


def test_a_refusal_has_nothing_to_write_again(client, monkeypatch):
    """A refusal's answer is the reason it was refused, derived from no result.
    Writing it again would be a translation, which this path never produces:
    no button, and the route says why rather than calling the model."""
    monkeypatch.setattr(station_app, "_analysis_client", Exploding())
    run_id = _asked_in("zh-TW", refused=True)

    english = read_in(client, "en").get(f"/ask/{run_id}").text
    assert STRINGS["en"]["analysis.as_asked"] in english
    assert STRINGS["en"]["analysis.answer.rewrite"] not in english

    response = read_in(client, "en").post(f"/ask/{run_id}/answer", follow_redirects=False)
    assert response.status_code == 400
    assert read_in(client, "en").post("/ask/999999/answer", follow_redirects=False).status_code == 404


def test_the_figure_check_reads_the_answer_the_page_shows(client, monkeypatch):
    """A rewrite is prose from the same results and gets the same check: a
    figure no result renders as is flagged on the rewrite in its language and
    not on the original in its own."""
    model = RewritingClient("The top score was 0.99 on that board.")
    monkeypatch.setattr(station_app, "_analysis_client", model)
    run_id = _asked_in("zh-TW", results=A_RESULT)
    read_in(client, "en").post(f"/ask/{run_id}/answer", follow_redirects=False)

    english = read_in(client, "en").get(f"/ask/{run_id}").text
    assert "0.99" in english and 'class="claims' in english

    chinese = read_in(client, "zh-TW").get(f"/ask/{run_id}").text
    assert 'class="claims' not in chinese


def test_every_plannable_tool_says_what_it_does_in_both_languages():
    """The rules block on `/ask` lists what can be asked, one line per
    registered tool, and the line has to exist in the reader's language --
    the docstring the refusal quotes is English."""
    from aoi_agent.analysis.plan import PLANNABLE_TOOLS

    for name in PLANNABLE_TOOLS:
        for locale in LOCALES:
            key = f"tool.{name}.does"
            assert STRINGS[locale][key] != key, f"{key} missing from {locale}"


def test_the_lines_language_defaults_to_the_stations_and_tolerates_a_bad_value(monkeypatch):
    from aoi_agent.i18n import DEFAULT_LOCALE, LINE_LANGUAGE_ENV, line_language

    monkeypatch.delenv(LINE_LANGUAGE_ENV, raising=False)
    assert line_language() == DEFAULT_LOCALE
    monkeypatch.setenv(LINE_LANGUAGE_ENV, "en")
    assert line_language() == "en"
    # A typo must not take the explanations off the line; it falls back.
    monkeypatch.setenv(LINE_LANGUAGE_ENV, "klingon")
    assert line_language() == DEFAULT_LOCALE


def test_the_language_note_lists_every_class_it_asks_to_be_kept():
    # "Leave defect classes as they appear" was measured on 2026-09-01 and the
    # Chinese rationale translated `open` on 24 of 41 anyway. A closed set is
    # enumerated, not described, and in both languages.
    from aoi_agent.data.deeppcb import CLASS_NAMES, FALSE_CALL
    from aoi_agent.i18n import LANGUAGE_NOTE

    for note in LANGUAGE_NOTE.values():
        for name in (*CLASS_NAMES.values(), FALSE_CALL):
            assert f"`{name}`" in note
    # and the two renderings it actually produced are forbidden by name
    assert "開路" in LANGUAGE_NOTE["zh-TW"] and "短路" in LANGUAGE_NOTE["zh-TW"]
