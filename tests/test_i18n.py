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
