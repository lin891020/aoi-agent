"""The re-verification station.

What an operator gets when the agent hands a region over: the queue of regions
still waiting, and for each one the same evidence the agent had -- the images,
the model's reading, the production context, the acceptance criteria it
retrieved, and why it declined to decide.

Two things this deliberately does not do:

* It never shows ``ground_truth``. The operator's answer is a label for the
  next training round, and a label copied off the answer key is worth nothing.
* It never re-runs the flow to render a page. The suspended state is already in
  the checkpointer; reading it costs a disk seek, whereas re-running it costs a
  20B model's inference and could hand back a different rationale than the one
  the operator is looking at.

Forms post and redirect rather than requiring JavaScript. htmx keeps the queue
count live and the keyboard shortcuts make the common path fast, but with both
switched off the station still works -- a review station that breaks on a
locked-down shop-floor browser is worse than a CLI.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from aoi_agent.analysis import service as analysis_service
from aoi_agent.analysis.graph import build_analysis_graph
from aoi_agent.analysis.plan import store_domains
from aoi_agent.graph.flow import DEFAULT_MODEL, build_graph
from aoi_agent.llm.ollama import OllamaClient
from aoi_agent.station import images, service
from aoi_agent.station.chart_svg import render_svg
from aoi_agent.station.result_view import error_text, readable_rows
from aoi_agent.store import analysis as analysis_store
from aoi_agent.store import escalations
from aoi_agent.store.boards import correction_summary, corrections, resolve_candidate

HERE = Path(__file__).parent

app = FastAPI(title="AOI re-verification station")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")

_graph = None
_analysis_graph = None


def graph():
    """Built on first use, not at import.

    The station has to come up whether or not Ollama is running: an operator
    still needs to see and answer the queue that was raised before the model
    fell over.
    """
    global _graph
    if _graph is None:
        _graph = build_graph(OllamaClient(os.getenv("AOI_AGENT_MODEL", DEFAULT_MODEL)))
    return _graph


def analysis_graph():
    """The analysis flow, built on first use for the same reason as ``graph()``.

    A separate graph rather than a mode of the disposition one: it has no
    checkpointer, because nothing here suspends and nobody is in the loop.
    """
    global _analysis_graph
    if _analysis_graph is None:
        _analysis_graph = build_analysis_graph(
            OllamaClient(os.getenv("AOI_AGENT_MODEL", DEFAULT_MODEL))
        )
    return _analysis_graph


def _reference(stem: str, index: int) -> str:
    return f"{stem}#{index}"


def _candidate_or_404(stem: str, index: int) -> dict:
    record = resolve_candidate(_reference(stem, index))
    if record is None:
        raise HTTPException(404, f"no candidate {stem}#{index}")
    return record


def _next_pending(after: str | None = None) -> str | None:
    """The reference the operator should see next, skipping the one just done."""
    queue = [row["reference"] for row in escalations.pending()]
    for reference in queue:
        if reference != after:
            return reference
    return None


@app.get("/", response_class=HTMLResponse)
def queue_page(request: Request):
    queue = escalations.pending()
    return templates.TemplateResponse(
        request, "queue.html", {"queue": queue, "waiting": len(queue)}
    )


def _go_to_next(after: str | None) -> RedirectResponse:
    reference = _next_pending(after)
    if reference is None:
        return RedirectResponse("/", status_code=303)
    stem, _, index = reference.partition("#")
    return RedirectResponse(f"/c/{stem}/{index}", status_code=303)


@app.get("/next")
def next_in_queue():
    """Send the operator to the region that has waited longest."""
    return _go_to_next(None)


@app.get("/next/{stem}/{index}")
def next_after(stem: str, index: int):
    """The same, skipping one just answered.

    The region travels as two path segments rather than as a ``<stem>#<index>``
    reference, in a query string or anywhere else in a URL: the ``#`` starts a
    fragment and the server never sees the index at all.
    """
    return _go_to_next(_reference(stem, index))


@app.get("/c/{stem}/{index}", response_class=HTMLResponse)
def station_page(request: Request, stem: str, index: int):
    reference = _reference(stem, index)
    record = _candidate_or_404(stem, index)
    escalation = escalations.get(service.thread_for(reference))
    state = service.flow_state(graph(), reference)

    return templates.TemplateResponse(
        request,
        "station.html",
        {
            "reference": reference,
            "stem": stem,
            "index": index,
            "candidate": record,
            "escalation": escalation,
            "state": state,
            "options": service.VERDICT_OPTIONS,
            "waiting": len(escalations.pending()),
            "next_reference": _next_pending(after=reference),
        },
    )


@app.post("/c/{stem}/{index}/verdict")
def submit_verdict(
    stem: str,
    index: int,
    verdict: str = Form(...),
    reviewer: str = Form("operator"),
):
    """Hand the operator's answer back to the suspended run."""
    reference = _reference(stem, index)
    _candidate_or_404(stem, index)

    if verdict not in service.VERDICT_OPTIONS:
        raise HTTPException(400, f"{verdict!r} is not a valid verdict")

    escalation = escalations.get(service.thread_for(reference))
    if escalation is None or escalation["status"] != "pending":
        # Two operators opened the same region, or a back button was pressed.
        # The first answer stands; recording a second would put two labels on
        # one region and silently corrupt the training set.
        return RedirectResponse("/next", status_code=303)

    service.resume_review(graph(), reference, verdict, reviewer.strip() or "operator")
    return RedirectResponse(f"/next/{stem}/{index}", status_code=303)


@app.get("/c/{stem}/{index}/triptych.png")
def triptych_png(stem: str, index: int):
    record = _candidate_or_404(stem, index)
    try:
        png = images.triptych(stem, images.candidate_from_record(record))
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    return Response(png, media_type="image/png")


@app.get("/c/{stem}/{index}/patch.png")
def patch_png(stem: str, index: int):
    record = _candidate_or_404(stem, index)
    try:
        png = images.model_patch(stem, images.candidate_from_record(record))
    except KeyError as error:
        raise HTTPException(404, str(error)) from error
    return Response(png, media_type="image/png")


@app.get("/queue-count", response_class=HTMLResponse)
def queue_count():
    """Polled by the station header so an operator sees work arriving."""
    return HTMLResponse(str(len(escalations.pending())))

@app.get("/corrections", response_class=HTMLResponse)
def corrections_page(request: Request):
    """Where operators overruled the model.

    The project's claim is that these rows become the next training set, and a
    claim you cannot see is a claim nobody believes. The aggregate is the part
    worth having over the CLI listing: a class the operators overturn again and
    again is a training-set or threshold problem, and that is invisible in a
    chronological list past the first screenful.

    Still no ``ground_truth`` here, even though this page is engineering-facing
    rather than operator-facing. It is one link from the queue, and an operator
    who reads the answers here carries them back to the region they are judging.
    Whether the operators were themselves right is a question for the evaluation
    scripts, which are not served over HTTP.
    """
    return templates.TemplateResponse(
        request,
        "corrections.html",
        {
            "rows": corrections(200),
            "summary": correction_summary(),
            "waiting": len(escalations.pending()),
        },
    )



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


def _analysis_context(
    request: Request, run: dict | None, errors: list[str] | None = None
) -> dict:
    """Everything ``analysis.html`` renders, for all three of its entrances.

    ``errors`` is the live plan-validation list, which only a caller holding the
    graph's state has: a stored run keeps the errors inside its answer, because
    that is what the reader needs and normalising them would be a column nobody
    queries. Passing them explicitly is how the streaming route hands them over
    before the run is saved.

    A chart is a whole specification or it is absent -- there is no empty spec
    to render, so the guard is on the key being missing, not on a kind. Only
    two of the five tools have chart builders, which is why `readable_data` is
    here too: the raw figures sit beside the prose for every tool, so a reader
    can catch a summary that describes correct data incorrectly.
    """
    return {
        "run": run,
        "plan_errors": list(errors or (run or {}).get("plan_errors") or []),
        "examples": EXAMPLE_QUESTIONS,
        "recent": analysis_store.recent_runs(8),
        "coverage_days": store_domains()["max_days"],
        "chart_svg": render_svg(run["chart"]) if run and run.get("chart") else "",
        # Passed as a callable rather than precomputed: the results are read
        # straight out of the store and nothing here should copy them to hang a
        # display field on. It is what keeps `ground_truth` filtered in Python,
        # at the dict boundary, rather than in the template.
        "readable_data": readable_rows,
        # Same reason: an error is printed as written, and Jinja would `str()`
        # a dict-valued one straight past the guard the rows go through.
        "tool_error": error_text,
        "waiting": len(escalations.pending()),
    }


@app.get("/ask", response_class=HTMLResponse)
def ask_page(request: Request):
    return templates.TemplateResponse(
        request, "analysis.html", _analysis_context(request, None)
    )


@app.post("/ask")
def ask(question: str = Form(...), asked_by: str = Form("operator")):
    """Run the question, then redirect to the run it produced.

    Post-redirect-get, so a refresh re-reads a stored document rather than
    asking a 20B model the same question again and possibly answering it
    differently.
    """
    if not question.strip():
        raise HTTPException(400, "a question is required")
    run = analysis_service.answer_question(
        analysis_graph(), question.strip(), asked_by.strip() or "operator"
    )
    return RedirectResponse(f"/ask/{run['id']}", status_code=303)


@app.get("/ask/{run_id}", response_class=HTMLResponse)
def ask_result(request: Request, run_id: int):
    """A saved run, re-rendered from the store. Nothing here calls a model."""
    run = analysis_store.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"no analysis run {run_id}")
    return templates.TemplateResponse(
        request, "analysis.html", _analysis_context(request, run)
    )
