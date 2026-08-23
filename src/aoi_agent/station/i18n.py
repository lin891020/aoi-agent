"""The station's text, in the two languages it is read in.

The line this is built for reads Traditional Chinese; the acceptance criteria
and the benchmark reports are written in English. Both are true at once and the
station had been half of each -- `/ask` in Chinese, the queue and the board
record in English -- which is worse than either.

**Language is a rendering, not a record.** What this module translates is
chrome: headings, column names, buttons, axis labels. It never touches the
question a supervisor typed, and it never touches what the planning call wrote
about that question -- those are what happened, and a record rewritten into
another language is a record of something nobody did. The one piece of prose
that does follow the language is the synthesised answer, and it follows by
being *written again* from the stored results, down the same measured path,
never by being translated. See `analysis/service.py`.

A key missing from one table is a bug the suite catches --
`tests/test_i18n.py` compares the two key sets and fails on any difference,
which is the only thing standing between "changed the Chinese" and "forgot the
English". At runtime a missing key renders as the key itself: visible and ugly,
rather than a 500 on a page whose figures are all still correct.
"""

from __future__ import annotations

#: What the shop floor reads. English is the second language here, not the
#: first, and the default says so.
DEFAULT_LOCALE = "zh-TW"

STRINGS: dict[str, dict[str, str]] = {
    "zh-TW": {
        # -- charts ---------------------------------------------------------
        # Identifiers are not translated. `mousebite` is what the store calls
        # that class and what the work instruction calls it; a chart axis
        # showing 鼠咬 and a payload showing `mousebite` are two vocabularies
        # for a reader to reconcile in their head.
        "chart.title.defects_by_class": "各類缺陷數量",
        "chart.title.share_by_machine": "各機台缺陷佔比",
        "chart.axis.defect_class": "缺陷類別",
        "chart.axis.count": "數量",
        "chart.axis.machine": "機台",
        "chart.axis.share_of_own_defects": "佔該機台缺陷的比例",
        "chart.series.line_id": "產線 {line_id}",
        "chart.series.machine_id": "機台 {machine_id}",
        "chart.series.lot_id": "批號 {lot_id}",
        "chart.series.everything": "全部",
        "chart.series.share_of": "{defect_type} 佔比",
        "chart.series.fleet_average": "全廠平均",
    },
    "en": {
        # -- charts ---------------------------------------------------------
        "chart.title.defects_by_class": "Defects by class",
        "chart.title.share_by_machine": "Defect share by machine",
        "chart.axis.defect_class": "class",
        "chart.axis.count": "count",
        "chart.axis.machine": "machine",
        "chart.axis.share_of_own_defects": "share of that machine's defects",
        "chart.series.line_id": "Line {line_id}",
        "chart.series.machine_id": "Machine {machine_id}",
        "chart.series.lot_id": "Lot {lot_id}",
        "chart.series.everything": "All",
        "chart.series.share_of": "share of {defect_type}",
        "chart.series.fleet_average": "fleet average",
    },
}

#: The locales a caller may ask for. Derived from the tables rather than
#: declared beside them, so adding a language cannot leave a list behind.
LOCALES: tuple[str, ...] = tuple(STRINGS)


def normalise(locale: str | None) -> str:
    """The locale to use for a value that may have come from a cookie."""
    return locale if locale in STRINGS else DEFAULT_LOCALE


def translate(key: str, locale: str | None = None, /, **args: object) -> str:
    """One string, in one language.

    Falls back to the key rather than raising. A page whose figures are all
    correct should not 500 over a heading, and the key on screen names the
    thing to fix. Bad arguments fall back the same way: a template that renders
    `{line_id}` literally is a visible fault, an exception is a lost page.
    """
    table = STRINGS[normalise(locale)]
    template = table.get(key)
    if template is None:
        return key
    try:
        return template.format(**args)
    except (KeyError, IndexError):
        return template


def label_from(spec: dict, field: str, locale: str | None = None) -> str:
    """A chart specification's label, whichever way it was stored.

    ``field`` is the bare name -- ``title``, ``y_label``, ``name``. A spec
    written after 2026-08-23 carries ``<field>_key`` and optional
    ``<field>_args`` and is translated. One written before carries the rendered
    English sentence and is shown as it is: the language it was drawn in is not
    recoverable from it, and inventing one would be worse than the gap. That
    absence is named by `tests/test_chart_svg.py` rather than left for a reader
    to find in a redrawn chart from last quarter.
    """
    key = spec.get(f"{field}_key")
    if key:
        return translate(key, locale, **(spec.get(f"{field}_args") or {}))
    return str(spec.get(field, "") or "")
