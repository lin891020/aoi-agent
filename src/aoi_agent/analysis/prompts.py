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
# The hidden-key rule lives with the walk that renders it, so there is one
# spelling of `ground_truth` for both routes out of a tool's payload. The
# import points at the station from the analysis layer, which is the wrong
# direction on paper -- taken deliberately, because two copies of an invariant
# is the failure it is meant to prevent, and `result_view` imports nothing.
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
        # An undocumented tool costs the model one line of description.
        # Indexing a bare [0] here cost it every plan: the catalogue is
        # built for each one, so one docstring-less tool raised through
        # the planner for every question asked.
        summary = next(iter((function.__doc__ or "").strip().splitlines()), "")
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
        {"role": "system", "content": SYNTHESIS_PROMPT},
        {"role": "user", "content": body},
    ]
