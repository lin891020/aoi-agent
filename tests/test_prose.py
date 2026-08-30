"""The boundary between a model's Markdown and the page.

`result_view.py` is the other one of these, and it took five review rounds
because each fix closed one door into the same room. The lesson it left is that
a boundary needs a single entrance, so this module has one function and it
returns structure. The test that matters most here is not any single escaping
case -- it is `test_no_block_or_span_can_carry_markup`, which says there is no
shape this function can produce that a template could render as a tag.
"""

from __future__ import annotations

import pytest

from aoi_agent.station.prose import blocks

# The shape a real answer arrives in, taken from a run on 2026-08-23.
ANSWER = """**Assumptions (restated)**
- The comparison uses the most recent 7-day window.
- Criteria are shown for every defect class.

**Defect composition (7-day window)**

| Line | Boards | Total defects |
|------|--------|---------------|
| L1 | 137 | 966 |
| L2 | 142 | 1,049 |

Any confirmed `open` is a **critical** defect.
"""


def kinds(answer: str) -> list[str]:
    return [block["kind"] for block in blocks(answer)]


def texts(block: dict) -> str:
    """Every character the block would put on screen, span kinds ignored."""
    if block["kind"] == "table":
        cells = [*block["head"], *[cell for row in block["rows"] for cell in row]]
        return "".join(span["text"] for cell in cells for span in cell)
    if block["kind"] == "list":
        return "".join(span["text"] for item in block["items"] for span in item)
    return "".join(span["text"] for span in block["spans"])


def test_a_real_answer_becomes_the_blocks_it_was_written_as():
    assert kinds(ANSWER) == [
        "paragraph", "list", "paragraph", "table", "paragraph",
    ]


def test_a_pipe_table_becomes_a_table_rather_than_a_row_of_characters():
    table = next(b for b in blocks(ANSWER) if b["kind"] == "table")

    assert [texts({"kind": "paragraph", "spans": c}) for c in table["head"]] == [
        "Line", "Boards", "Total defects",
    ]
    assert len(table["rows"]) == 2
    assert texts({"kind": "paragraph", "spans": table["rows"][1][2]}) == "1,049"


def test_bold_and_code_become_spans_rather_than_asterisks_and_backticks():
    last = blocks(ANSWER)[-1]
    by_kind = {span["kind"]: span["text"] for span in last["spans"]}

    assert by_kind["code"] == "open"
    assert by_kind["strong"] == "critical"
    assert "**" not in texts(last) and "`" not in texts(last)


# ---------------------------------------------------------------------------
# What must not happen
# ---------------------------------------------------------------------------

MARKUP = [
    "<script>alert(1)</script>",
    '<img src=x onerror="alert(1)">',
    "<b>not bold</b>",
    "<!-- comment -->",
    "<style>body{display:none}</style>",
    "</p><script>x</script><p>",
]


@pytest.mark.parametrize("hostile", MARKUP)
def test_html_in_an_answer_stays_the_characters_it_was_typed_as(hostile):
    """Not escaped here -- *kept*. Escaping happens in Jinja on the way out;
    this function's job is to never produce anything but text."""
    produced = blocks(f"A line with {hostile} in it.")

    assert len(produced) == 1
    assert hostile in texts(produced[0])
    for span in produced[0]["spans"]:
        assert span["kind"] in {"text", "strong", "code"}


def test_no_block_or_span_can_carry_markup():
    """The property the whole module rests on, asserted over every shape.

    A span has a kind and a string. There is no field a tag could travel in, so
    there is no answer -- hostile, malformed, or merely unusual -- that could
    make this function emit one. This is why the template needs no `|safe`, and
    why `|safe` must never be added: it would move the boundary from here,
    where it is one function, to there, where it is every use site.
    """
    hostile = "\n\n".join([
        *MARKUP,
        "# <h1>heading</h1>",
        "- <li>item</li>",
        "| <td>a</td> | b |\n|---|---|\n| <script>x</script> | d |",
        "**<b>bold</b>**",
        "`<code>x</code>`",
    ])

    def walk(node) -> None:
        if isinstance(node, dict):
            if set(node) == {"kind", "text"}:
                assert isinstance(node["text"], str)
                assert node["kind"] in {"text", "strong", "code"}
                return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        else:
            assert isinstance(node, (str, int, bool)), f"unexpected {type(node)}"

    for block in blocks(hostile):
        assert block["kind"] in {"heading", "paragraph", "list", "table"}
        walk(block)


def test_a_ragged_table_is_left_as_text_rather_than_rendered_short():
    """A row with fewer cells than its header renders figures under the wrong
    headings, which is this page's characteristic failure arriving by a
    different door. A visible pipe row sends the reader to the figures panel; a
    silently misaligned table does not."""
    ragged = "| a | b | c |\n|---|---|---|\n| 1 | 2 |\n"
    produced = blocks(ragged)

    assert all(block["kind"] != "table" for block in produced)
    assert "| 1 | 2 |" in "".join(texts(block) for block in produced)


def test_a_table_with_no_rule_row_is_not_a_table():
    """Three rows, not two. With two, dropping the rule check still yields no
    table -- the data row gets eaten in the rule's place and nothing is left to
    be a row -- so a two-row case passes whether the check is there or not.
    """
    assert all(
        block["kind"] != "table"
        for block in blocks("| a | b |\n| 1 | 2 |\n| 3 | 4 |\n")
    )


def test_a_model_heading_cannot_outrank_the_page_headings():
    """`#` in an answer is the answer's own outline, nested under the section it
    sits in. A screen reader reading the page's structure should not find the
    prose competing with `5 · 回答` for the top level."""
    levels = [b["level"] for b in blocks("# one\n\n## two\n\n###### six\n")]

    assert min(levels) >= 3 and max(levels) <= 6


def test_an_unclosed_bold_run_stays_as_typed():
    """Guessing where the model meant to close it is a rewrite of the answer."""
    produced = blocks("A **partly bold line that never closes")

    assert texts(produced[0]) == "A **partly bold line that never closes"


def test_backticks_win_over_asterisks_so_an_expression_survives():
    produced = blocks("Use `a ** b` for the power.")
    code = [s["text"] for s in produced[0]["spans"] if s["kind"] == "code"]

    assert code == ["a ** b"]


def test_an_empty_answer_is_no_blocks_rather_than_an_empty_paragraph():
    assert blocks("") == []
    assert blocks("   \n\n  \n") == []


def test_a_wrapped_list_item_stays_one_item():
    """The standards documents are hard-wrapped; a bullet's second line is
    the same bullet, not a paragraph after it."""
    from aoi_agent.station.prose import blocks
    text = ("- Confirmed open: scrap or route to rework for jumper repair, per the product's\n"
            "repair class. Class 3 product may not be jumper-repaired.\n"
            "- Suspected open that measures continuous on electrical test: record as a\n"
            "cosmetic thinning.")
    out = blocks(text)
    assert [b["kind"] for b in out] == ["list"]
    items = ["".join(s["text"] for s in item) for item in out[0]["items"]]
    assert len(items) == 2
    assert items[0].endswith("may not be jumper-repaired.")
    assert items[1].endswith("cosmetic thinning.")


def test_a_blank_line_still_ends_a_list():
    from aoi_agent.station.prose import blocks
    out = blocks("- one\n\nA paragraph.")
    assert [b["kind"] for b in out] == ["list", "paragraph"]
