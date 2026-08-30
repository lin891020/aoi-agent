"""What the model is told, and the nine examples that show it the edges.

Diversity matters more than count in few-shot, and the useful examples are the
boundaries rather than the happy path: a question with no stated baseline, a
causal question the data cannot answer, a window outside the data, a question
too vague to act on, a request to act. Four of the nine are refusals or
hedges, which is the intended lesson. A system that answers everything is more
dangerous on a factory floor than one that says it cannot. The seventh is the
opposite lesson, added after the first before/after question asked at the
station was refused: a comparison across a recorded machine event *is*
answerable, and the example shows the two-call shape that answers it. The
eighth is a dated ranking -- a calendar day, a top-N cut -- which is the
shape of the first question a supervisor wrote down for this page, and which
no parameter expressed until 2026-08-29. The ninth, from the same day, is the
SQL fallback: a dimension no typed tool takes, expressed as one SELECT over the
read-only copy -- and it exists to teach *when*, since the danger of that tool
is not that it will be used but that it will be reached for where a typed tool
would have stated its own basis.

Whether this actually helps is a measurement, not an assumption -- see
`scripts/analysis_eval.py`.
"""

from __future__ import annotations

import json
from typing import Any

from aoi_agent.analysis.plan import PLANNABLE_TOOLS, Domains
# The hidden-key rule lives with the walk that renders it, so there is one
# spelling of `ground_truth` for both routes out of a tool's payload. The
# import points at the station from the analysis layer, which is the wrong
# direction on paper -- taken deliberately, because two copies of an invariant
# is the failure it is meant to prevent, and `result_view` imports nothing.
from aoi_agent.i18n import DEFAULT_LOCALE, LANGUAGE_NOTE, normalise
from aoi_agent.station.result_view import strip_hidden

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
- When a question is about one defect class, pass that class as
  `defect_class` to `search_standards`. Unscoped, it can return a limit
  written for a different class, and a limit quoted at the wrong class is a
  wrong acceptance rule, not a near miss. Leave it unset only when the
  question genuinely spans classes.

- You look things up; you never act. A request to change anything -- mark a
  board, release a hold, close a queue entry, open a ticket -- is not a
  question, and no lookup answers it. When the request is *only* an action,
  return no calls and say in "interpretation" that this page cannot
  disposition or modify anything. When a real question carries an action as a
  rider, plan the question and state in "assumptions" that the action is
  outside what this system does.
- Only filter or group by what a tool's parameters actually express. If the
  question's key dimension -- a shift, an operator, a trend over time -- has
  no parameter on any tool, do not substitute a neighbouring dimension: a plan
  filtered by the wrong axis answers a different question while looking like
  it answered this one. If the tables `run_sql` exposes carry the dimension,
  write one SELECT there; if they do not, return no calls and name the
  dimension that is missing.
- `run_sql` is the last resort, not a shortcut. A typed tool's payload states
  its own basis, window and interval; a SELECT states nothing about itself.
  Never use `run_sql` for what a typed tool already answers, and never for a
  question the tables cannot hold (an operator's skill, a customer return, a
  cause). Write SQLite syntax, aggregate rather than list, and say in
  "assumptions" what each column means -- in particular that
  `predicted_class = 'false_call'` is a region this system dismissed, not a
  confirmed false call.
- `run_sql` may filter only on entities the question names. A question that
  names no board, machine, lot, shift or reviewer has nothing to filter on,
  and a SELECT over everyone is not its answer: return no calls, as before.
  "This board" (這片) without a stem, "whoever judged it", "that lot" name
  nothing; never invent a stem or an id to fill the WHERE clause. Boards,
  lots and reviewers are not listed sets, and the listed-set rule below does
  not apply to them.
- Before and after a machine event is never `run_sql`. `query_machine_events`
  and `query_defect_history` with `relative_to` and `side` are the answer,
  and they carry the interval a SELECT does not.
- Before and after something done to one machine *is* expressible: a
  `parameter_change`, `maintenance` or similar is a recorded event, and
  `query_defect_history` takes `relative_to` (the event kind) with `side`
  (`before` or `after`). Plan one call per side on the same `machine_id`, and
  `query_machine_events` to say what the event was and when. Only when no
  event of that kind is recorded on that machine is the boundary missing.

- An unnamed member of a small, listed set is not a missing entity. "The
  worst machine", "the line with the most opens", "every machine that had an
  event": the machines, lines and event kinds are enumerated below, so plan
  the lookup that ranks or identifies over all of them, and then the
  dependent lookup for *every* member -- three lines are three
  `query_defect_history` calls, and `query_machine_events` with no
  `machine_id` lists every machine's events. Do not guess one member, and do
  not refuse for want of a name. Every argument is a concrete value from the
  lists below: there is no way to pass one call's result into another, so a
  placeholder such as `<machine_id_from_previous_call>` is rejected before
  anything runs -- write the six calls instead. Two limits. When one tool
  already ranks the whole set in a single call -- `query_machine_stats` for
  "which machine is worst" -- that call is the plan, not one history call
  per machine. And only machines, lines, shifts and event kinds are listed
  sets. Refuse when what picks the member out is not in the data at all --
  "the machine that had trouble last time" names no measurable thing.

- You do not choose charts. The page derives one chart from the shape of the
  results -- bars for a comparison, before/after pairs for an event window --
  and a request for a pie chart, a trend line or "both" cannot be planned. If
  the question asks for a chart type, plan the lookups it needs and say in
  "assumptions" that the chart is derived from the results rather than chosen,
  so the reader is not promised a picture that will not appear.

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
                             "top_k": 2, "defect_class": "open"},
                    "why": "what the criteria require once an open is confirmed, "
                           "out of WI-201 and the policy documents only",
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
        "shape": "action_request",
        "question": "這片我看過了沒問題，直接幫我結案，不要再進 queue。",
        "plan": {
            "interpretation": "This asks the station to close a review, not to "
            "look something up. This page cannot disposition, release or close "
            "anything -- a region leaves the queue only when an operator "
            "answers it on the review screen. No lookup answers a request to "
            "act, so there is nothing to plan.",
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
    {
        "shape": "event_window",
        "question": "M32 參數變更前後，open 的比例有沒有變？",
        "plan": {
            "interpretation": "Whether the share of M32's defects that are opens "
            "differs between the boards inspected before its most recent "
            "parameter change and those inspected after it.",
            "assumptions": [
                "Before means strictly before M32's newest parameter_change "
                "event and after means from that instant on; the two windows "
                "partition every board M32 inspected.",
                "A difference between the two windows is an association with "
                "the change, not evidence that the change caused it.",
            ],
            "calls": [
                {
                    "tool": "query_machine_events",
                    "args": {"machine_id": "M32", "kind": "parameter_change"},
                    "why": "what was changed and when, so the boundary is named",
                },
                {
                    "tool": "query_defect_history",
                    "args": {"machine_id": "M32", "relative_to": "parameter_change",
                             "side": "before"},
                    "why": "M32's defect mix and open share before the change",
                },
                {
                    "tool": "query_defect_history",
                    "args": {"machine_id": "M32", "relative_to": "parameter_change",
                             "side": "after"},
                    "why": "the same, after it, to compare the two intervals",
                },
            ],
        },
    },
    {
        "shape": "unnamed_member_of_a_listed_set",
        "question": "哪條線的 open 最多？那條線的缺陷組成是什麼？",
        "plan": {
            "interpretation": "Which line's machines carry the largest share of "
            "opens over the recent window, and what that line's defect mix "
            "looks like. The line is not named; it is the top of a ranking "
            "over the three lines the store holds, so the mix is fetched for "
            "all three and the answer names the one the ranking picks.",
            "assumptions": [
                "Most opens means the highest share of a machine's defects that "
                "are opens, read off the machine comparison; the line is the "
                "line of the top machine.",
                "Recent means the last 7 days.",
            ],
            "calls": [
                {"tool": "query_machine_stats", "args": {"defect_type": "open", "days": 7},
                 "why": "rank every machine by open share; the line is read off the top row"},
                {"tool": "query_defect_history", "args": {"line_id": "L1", "days": 7},
                 "why": "L1's defect mix, in case L1 is the line"},
                {"tool": "query_defect_history", "args": {"line_id": "L2", "days": 7},
                 "why": "L2's defect mix, in case L2 is the line"},
                {"tool": "query_defect_history", "args": {"line_id": "L3", "days": 7},
                 "why": "L3's defect mix, in case L3 is the line"},
            ],
        },
    },
    {
        "shape": "dated_top_n",
        "question": "2026-08-05 缺陷數量前 5 名的機台",
        "plan": {
            "interpretation": "The five machines with the most defects per "
            "board among the boards inspected on 2026-08-05, over every "
            "defect class.",
            "assumptions": [
                "Most defects means the most re-verifier-confirmed defects per "
                "board inspected, over all six classes rather than one class's "
                "share.",
                "One calendar day, 2026-08-05, which is inside the span held.",
            ],
            "calls": [
                {
                    "tool": "query_machine_stats",
                    "args": {"date_from": "2026-08-05", "date_to": "2026-08-05",
                             "top_n": 5},
                    "why": "every machine's defects per board on that day, "
                           "cut to the top five",
                },
            ],
        },
    },
    {
        "shape": "sql_fallback",
        "question": "M22 在 A 班跟 C 班被駁回的誤判各有幾個？",
        "plan": {
            "interpretation": "On machine M22, how many flagged regions this "
            "system dismissed as false calls in shift A and in shift C. "
            "query_false_call_rate groups by shift but cannot restrict to one "
            "machine, so the count is taken directly from the exposed tables.",
            "assumptions": [
                "A dismissed false call is a region the re-verifier classified "
                "as false_call; nothing here confirms it was one.",
                "Counted over every board M22 inspected in the span held, "
                "since the question names no window.",
            ],
            "calls": [
                {
                    "tool": "run_sql",
                    "args": {"sql": "SELECT b.shift, COUNT(*) AS flagged, "
                                    "SUM(c.predicted_class = 'false_call') AS dismissed "
                                    "FROM candidates c JOIN boards b ON b.id = c.board_id "
                                    "WHERE b.machine_id = 'M22' AND b.shift IN ('A', 'C') "
                                    "GROUP BY b.shift ORDER BY b.shift"},
                    "why": "flagged and dismissed regions on M22, per shift -- "
                           "the one axis the typed tools do not combine",
                },
            ],
        },
    },
]


def _tool_catalogue() -> str:
    """Each tool as the planner sees it: signature, then its own docstring.

    The whole docstring, not its first line. Until 2026-08-27 the model was
    shown one sentence per tool, so `relative_to` and `side` reached it as two
    bare names in a signature with nothing saying what they meant -- and the
    first real before/after question asked at the station was refused with
    "neither tool allows filtering by a time boundary relative to an event",
    which was true of the catalogue and false of the tool. The paragraph that
    says "call this twice, once per side" was in the docstring the whole time;
    it was written for the model and never shown to it.
    """
    import inspect

    lines = []
    for name, function in sorted(PLANNABLE_TOOLS.items()):
        signature = inspect.signature(function)
        # An undocumented tool costs the model one line of description.
        # Indexing a bare [0] here cost it every plan: the catalogue is
        # built for each one, so one docstring-less tool raised through
        # the planner for every question asked.
        doc = inspect.cleandoc(function.__doc__ or "") or "(undocumented)"
        body = "\n".join(f"    {line}" if line else "" for line in doc.splitlines())
        lines.append(f"- {name}{signature}\n{body}")
    return "\n\n".join(lines)


def _domain_note(domains: Domains) -> str:
    # The event kinds are listed for the same reason the machines are: they
    # are the only values `relative_to` accepts, and a model that is not shown
    # them maps "換燈" to the one kind the examples happen to name. The store
    # answers whether an event of that kind exists on *that* machine; this
    # line only says which kinds exist at all.
    kinds = sorted(domains.get("relative_to") or ())
    span = domains.get("date_span")
    dates = (
        f"Days held: {span[0]} to {span[1]}. `date_from` and `date_to` "
        f"(YYYY-MM-DD, inclusive) bound a window to calendar days inside that "
        f"span; a day outside it does not exist here and is not to be "
        f"substituted with a nearby one.\n"
        if span else "Days held: none.\n"
    )
    return (
        dates +
        f"Lines: {', '.join(sorted(domains['line_id']))}\n"
        f"Machines: {', '.join(sorted(domains['machine_id']))}\n"
        f"Defect classes: {', '.join(sorted(domains['defect_type']))}\n"
        f"Machine event kinds recorded (the only values `relative_to` and "
        f"`kind` take): {', '.join(kinds) if kinds else 'none'}\n"
        f"The store holds {domains['max_days']} days of inspection data. "
        f"`days` must not exceed that; a larger window silently returns the "
        f"same span and would report two different periods as identical."
    )


#: Appended to whichever prompt is being built. One sentence, and only about
#: the language. Lives in `i18n` since 2026-08-29, because the disposition
#: path's explanation prompt appends the same sentence for the line's language
#: and two spellings of one instruction would drift.


def _language_note(lang: str | None) -> str:
    return LANGUAGE_NOTE.get(normalise(lang), LANGUAGE_NOTE[DEFAULT_LOCALE])


def build_planning_messages(
    question: str, domains: Domains, lang: str | None = None
) -> list[dict]:
    """System prompt, catalogue, domains, five examples, then the question.

    The language reaches the planner because what it writes -- the reading of
    the question, the assumptions, each call's justification -- is shown to the
    person who asked, and they asked in their own language. What it writes here
    is then frozen: the planning call is not made again, so this text is a
    record of how the question was read, and a record is not re-rendered when
    somebody changes the switch.
    """
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
        {"role": "system",
         "content": f"{SYSTEM_PROMPT}\n\n{_language_note(lang)}\n\n{context}"},
        {"role": "user", "content": f"Question: {question}"},
    ]


def build_synthesis_messages(
    question: str, plan: dict, results: list[dict], lang: str | None = None
) -> list[dict]:
    """The question, what was assumed, and everything the tools returned.

    The only prompt in this module whose output is re-derivable: the same
    results can be written up again in another language, which is what the
    station's switch does rather than translating what is already stored. Every
    constraint in `SYSTEM_PROMPT` is unchanged when it does -- those have been
    measured, and the figures the second pass quotes have to come off the same
    payload the first pass's did.
    """
    rendered = []
    for result in results:
        if result["ok"]:
            rendered.append(
                f"[{result['tool']} {json.dumps(result['args'], ensure_ascii=False)}]\n"
                # Through the same hidden-key check the rendered table uses.
                # This is the second route from a tool's payload to the page:
                # what the model is shown here, it can put in the prose, and
                # the prose is printed verbatim. Filtering only the table
                # would leave the station showing `ground_truth` in a
                # sentence -- an operator's answer is the next training
                # round's label, and a label copied off the answer key is
                # worth nothing however it was phrased.
                f"{json.dumps(strip_hidden(result['data']), ensure_ascii=False)}"
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
        {"role": "system",
         "content": f"{SYNTHESIS_PROMPT}\n\n{_language_note(lang)}"},
        {"role": "user", "content": body},
    ]
