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

from aoi_agent.station import app as station_app  # noqa: E402
from conftest import read_in, sign_in  # noqa: E402

TEMPLATES = pathlib.Path(station_app.__file__).parent / "templates"


@pytest.fixture
def client(operators):
    return sign_in(TestClient(station_app.app))


def test_the_station_opens_in_the_language_the_line_reads(client):
    page = client.get("/").text

    assert 'lang="zh-TW"' in page
    assert "等待人工判定的區域" in page


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
