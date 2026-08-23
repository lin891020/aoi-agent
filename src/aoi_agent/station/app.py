"""The re-verification station.

Two pages, for two people.

The queue is what an operator gets when the agent hands a region over: the
regions still waiting, and for each one the same evidence the agent had -- the
images, the model's reading, the production context, the acceptance criteria it
retrieved, and why it declined to decide. That page dispositions boards.

``/ask`` is for the shift supervisor, who walks up with a question rather than
a region: "is M22 drifting, and does that matter?". It reads production data
and answers in prose over a chart, and it dispositions nothing -- no board is
held or shipped by anything on it, which is why the analysis flow has no
checkpointer and no escalation queue behind it. Its failures terminate in a
message on the page. See `analysis/graph.py`, and the scoping note in
CLAUDE.md's invariants.

Both pages are behind a sign-in, since 2026-08-23. The exposure was not one
thing: an unauthenticated visitor to the queue saw the regions on one line,
while the same visitor at ``/ask`` could pull production statistics for the
whole plant, and the second is a change of kind rather than of degree. But the
reason the sign-in is here is the queue's, not ``/ask``'s -- an operator's
answer becomes the next training round's label, and a label whose author is a
text box is a label nobody can weigh. See ``station.auth``, which states what
the scheme does not protect against, and ``store.boards.record_decision``,
which is what actually refuses an unattributable answer. The middleware below
is the door; the store is the lock.

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

import json
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from aoi_agent.analysis import service as analysis_service
from aoi_agent.analysis.graph import build_analysis_graph
from aoi_agent.analysis.plan import store_domains
from aoi_agent.graph.flow import DEFAULT_MODEL, build_graph, explanation_notice
from aoi_agent.provenance import UNAVAILABLE, UNRECORDED, ReviewerIdentity
from aoi_agent.llm.ollama import OllamaClient
from aoi_agent.station import auth, images, service
from aoi_agent.station.chart_svg import render_svg
from aoi_agent.station.prose import blocks as prose_blocks
from aoi_agent.station.result_view import clip, error_text, readable_rows, shown_count
from aoi_agent.store import analysis as analysis_store
from aoi_agent.store import dispositions, escalations
from aoi_agent.store.boards import correction_summary, corrections, resolve_candidate

HERE = Path(__file__).parent

app = FastAPI(title="AOI re-verification station")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")

_graph = None
_analysis_graph = None
_analysis_domains = None


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


def analysis_domains():
    """The value domains the validator enforces, read once and then held.

    One snapshot, shared with the page. The graph freezes its domains into the
    plan node's closure at first use, so a page that re-read `store_domains()`
    on every render could tell a supervisor the store holds four days while the
    validator was still rejecting anything over nine -- a stated coverage and an
    enforced limit that disagree, with nothing on screen to say which is real.

    Frozen rather than live on both sides, because the alternative is rebuilding
    the graph whenever the store moves under it. Reseeding is a development
    action and the station is restarted after one; a supervisor's session is not
    where a changed dataset should appear halfway through.
    """
    global _analysis_domains
    if _analysis_domains is None:
        _analysis_domains = store_domains()
    return _analysis_domains


def analysis_graph():
    """The analysis flow, built on first use for the same reason as ``graph()``.

    A separate graph rather than a mode of the disposition one: it has no
    checkpointer, because nothing here suspends and nobody is in the loop.
    """
    global _analysis_graph
    if _analysis_graph is None:
        _analysis_graph = build_analysis_graph(
            OllamaClient(os.getenv("AOI_AGENT_MODEL", DEFAULT_MODEL)),
            analysis_domains(),
        )
    return _analysis_graph


#: Reachable without a session, and nothing else is.
#:
#: ``/static`` because a login page with no stylesheet is a login page nobody
#: trusts, and ``/login`` because otherwise there is nowhere to go. Everything
#: else -- the queue, a region, its images, the board record, the corrections
#: page, ``/ask`` and its stream -- is behind the check. Listed as an allowlist
#: rather than a set of decorated routes so that a route added later is
#: protected by default; the failure mode of the other arrangement is a new
#: endpoint that nobody remembered to mark.
PUBLIC_PREFIXES = ("/login", "/static", "/favicon.ico")


def _is_public(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in PUBLIC_PREFIXES)


def _safe_next(target: str | None) -> str:
    """Where to send an operator after signing in.

    Only a path on this station. An open redirect off a login form is the
    cheapest phishing primitive there is, and the parameter here exists purely
    so a bookmarked region survives the sign-in.
    """
    if not target or not target.startswith("/") or target.startswith("//"):
        return "/"
    return target


@app.middleware("http")
async def require_operator(request: Request, call_next):
    """Read the operator off the signed cookie, or refuse the request.

    Safe methods redirect to the login page, so a bookmarked region behaves
    the way a person expects. Everything else is a flat 401: a POST bounced
    into a redirect would lose the form body, and an operator who thought they
    had answered a region would have answered nothing.
    """
    operator = auth.operator_from_session(request.cookies.get(auth.COOKIE_NAME))
    request.state.operator = operator

    if operator is None and not _is_public(request.url.path):
        if request.method in ("GET", "HEAD"):
            target = request.url.path
            if request.url.query:
                target = f"{target}?{request.url.query}"
            return RedirectResponse(
                f"/login?next={quote(target, safe='')}", status_code=303
            )
        return PlainTextResponse(
            "not signed in -- the station takes an answer only from a named "
            "operator, because the answer becomes a training label",
            status_code=401,
        )
    return await call_next(request)


def operator_of(request: Request) -> ReviewerIdentity:
    """The identity a decision written on this request carries.

    From the session, never from the form. The station used to take the
    reviewer's name off a text input with ``operator`` prefilled, which is how
    9,140 decisions came to name nobody at all.
    """
    name = getattr(request.state, "operator", None)
    if not name:
        raise HTTPException(401, "not signed in")
    return ReviewerIdentity.signed_in(name)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/", error: str = ""):
    if getattr(request.state, "operator", None):
        return RedirectResponse(_safe_next(next), status_code=303)
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "next": _safe_next(next),
            "error": error,
            # Said on the page rather than in a log nobody reads: with no
            # operator file there is nobody who can answer the queue, and the
            # station will look broken rather than locked.
            "no_operators": not auth.load_operators(),
            "ephemeral_sessions": not os.getenv(auth.SECRET_ENV),
            "operators_path": str(auth.operators_path()),
        },
    )


@app.post("/login")
def login(request: Request, name: str = Form(...), secret: str = Form(...),
          next: str = Form("/")):
    """Exchange a passphrase for a signed session.

    A plain form post and a redirect, like every other write on this station:
    with JavaScript off, this works.
    """
    identity = auth.authenticate(name, secret)
    if identity is None:
        # One message for both failures. Which of the name and the passphrase
        # was wrong is not information this page owes an anonymous visitor.
        return RedirectResponse(
            f"/login?next={quote(_safe_next(next), safe='')}"
            "&error=that+name+and+passphrase+do+not+match",
            status_code=303,
        )
    response = RedirectResponse(_safe_next(next), status_code=303)
    response.set_cookie(
        auth.COOKIE_NAME,
        auth.issue_session(identity.name),
        max_age=auth.SESSION_MAX_AGE_S,
        httponly=True,
        samesite="lax",
        secure=auth.secure_cookie_for(request.url.scheme),
        path="/",
    )
    return response


@app.post("/logout")
def logout():
    """End the session. A POST, so a prefetched link cannot sign anyone out."""
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(auth.COOKIE_NAME, path="/")
    return response


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
    # Counted on the page rather than left to a report nobody runs. The LLM's
    # only remaining job is writing these, so a shift in which it wrote none is
    # a thing the person watching the queue should be able to see happening.
    unexplained = sum(
        1 for row in queue if (row["explanation_status"] or "ok") != "ok"
    )
    return templates.TemplateResponse(
        request,
        "queue.html",
        {"queue": queue, "waiting": len(queue), "unexplained": unexplained},
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
            # Rendered from the status, never stored as though the model had
            # written it. WI-300: an absent rationale is absent, and the gap is
            # not to be filled by any other means.
            "explanation_notice": explanation_notice(
                (escalation or {}).get("explanation_status") or ""
            ),
            "options": service.VERDICT_OPTIONS,
            "waiting": len(escalations.pending()),
            "next_reference": _next_pending(after=reference),
        },
    )


@app.post("/c/{stem}/{index}/verdict")
def submit_verdict(
    request: Request,
    stem: str,
    index: int,
    verdict: str = Form(...),
):
    """Hand the operator's answer back to the suspended run.

    There is no ``reviewer`` field on this form any more. The name comes off
    the session, so a posted one is ignored rather than trusted -- which is the
    difference between a record and a text box.
    """
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

    service.resume_review(graph(), reference, verdict, operator_of(request))
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


@app.get("/board/{stem}", response_class=HTMLResponse)
def board_page(request: Request, stem: str):
    """What was decided about one board, and under what.

    The question a customer return starts with -- "who decided this board was
    fine, when, and on what basis" -- had no page and no query behind it until
    2026-08-23, because the store held judgements about regions and nothing
    about boards. This renders the board-level record and the regions beneath
    it side by side, so the aggregate can be checked against what it was
    computed from rather than believed.

    Read-only. Nothing here dispositions anything: it is the record of
    dispositions already made.
    """
    rows = dispositions.decision_provenance(stem)
    if not rows:
        raise HTTPException(404, f"no board {stem}")
    return templates.TemplateResponse(
        request,
        "board.html",
        {
            "stem": stem,
            "rows": rows,
            "history": dispositions.history(stem),
            "assessment": dispositions.assess(stem),
            "absences": (UNAVAILABLE, UNRECORDED),
            "waiting": len(escalations.pending()),
        },
    )


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


def _analysis_context(run: dict | None) -> dict:
    """Everything ``analysis.html`` renders, for both of its entrances.

    A rejected or unparseable plan's reasons are not a key here. They reach a
    reader two other ways instead: live, as the stream's own `error` events;
    persisted, folded into `run.answer` by `report_node`. Both routes that call
    this (`ask_page`, `ask_result`) only ever render a stored run or none at
    all, and a stored run carries no separate errors column -- there used to be
    a `plan_errors` parameter here for a template block that no route could
    ever fill, since `/ask/stream` renders no template and a plain GET on
    `/ask/{run_id}` never carries the live state the stream held. Both the
    block and the parameter were removed together; see the Task 8 fix-round
    report for the trace of why.

    A chart is a whole specification or it is absent -- there is no empty spec
    to render, so the guard is on the key being missing, not on a kind. Only
    two of the five tools have chart builders, which is why `readable_data` is
    here too: the raw figures sit beside the prose for every tool, so a reader
    can catch a summary that describes correct data incorrectly.
    """
    return {
        "run": run,
        "examples": EXAMPLE_QUESTIONS,
        "recent": analysis_store.recent_runs(8),
        # The same snapshot the validator was built from, not a fresh read:
        # see `analysis_domains`. A page that states a coverage the validator
        # does not enforce is worse than a page whose coverage is a restart
        # behind.
        "coverage_days": analysis_domains()["max_days"],
        "chart_svg": render_svg(run["chart"]) if run and run.get("chart") else "",
        # Passed as a callable rather than precomputed: the results are read
        # straight out of the store and nothing here should copy them to hang a
        # display field on. It is what keeps `ground_truth` filtered in Python,
        # at the dict boundary, rather than in the template.
        "readable_data": readable_rows,
        # Same reason: an error is printed as written, and Jinja would `str()`
        # a dict-valued one straight past the guard the rows go through.
        "tool_error": error_text,
        # How many of the rows are fields the tool returned. The overflow
        # note is a row but not an item, and the summary counted it.
        "shown_count": shown_count,
        # The answer's Markdown, as blocks of spans. A callable for the same
        # reason as `readable_data`: the parsing is the boundary and it lives in
        # Python. What comes back is structure with no field a tag could travel
        # in, which is why the template renders it with no `|safe` -- see
        # `station/prose.py`.
        "prose_blocks": prose_blocks,
        "waiting": len(escalations.pending()),
    }


@app.get("/ask", response_class=HTMLResponse)
def ask_page(request: Request):
    return templates.TemplateResponse(
        request, "analysis.html", _analysis_context(None)
    )


@app.post("/ask")
def ask(request: Request, question: str = Form(...)):
    """Run the question, then redirect to the run it produced.

    Post-redirect-get, so a refresh re-reads a stored document rather than
    asking a 20B model the same question again and possibly answering it
    differently.
    """
    if not question.strip():
        raise HTTPException(400, "a question is required")
    run = analysis_service.answer_question(
        analysis_graph(), question.strip(), operator_of(request).name
    )
    return RedirectResponse(f"/ask/{run['id']}", status_code=303)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _merge_state(state: dict, payload: dict) -> None:
    """Fold one node's update into the running copy, honouring its reducers.

    ``stream_mode="updates"`` hands back one update per ``Send`` branch, not
    one per superstep, so two ``run_tool`` branches arrive as two separate
    dicts, each holding its own single-item ``results`` list. A plain
    ``state.update(payload)`` treats every key as last-write-wins, which is
    correct for a scalar like ``answer`` but silently drops every branch
    except whichever ``run_tool`` update is folded in last -- the same for
    ``timings_ms``, which the plan, collect and synthesise nodes all write to
    in turn. `AnalysisState` declares `results` with `operator.add` and
    `timings_ms` with `operator.or_` for exactly this reason; this mirrors
    both reducers so the copy held here for `save_run` cannot disagree with
    the state the graph itself accumulated.
    """
    for key, value in payload.items():
        if key == "results":
            # `or []`, not `and value` before choosing this branch: an empty
            # update must no-op, the same as `existing + [] == existing` under
            # `operator.add`. Falling through to `state[key] = value` on a
            # falsy update -- the original bug here -- would wipe every branch
            # already accumulated the moment one more arrived empty.
            state["results"] = [*state.get("results", []), *(value or [])]
        elif key == "timings_ms":
            state["timings_ms"] = {**state.get("timings_ms", {}), **(value or {})}
        else:
            state[key] = value


def _tool_base_key(tool: str, args: dict | None) -> str:
    """The content half of a call's key: everything but which occurrence it is.

    Two calls to the same tool with different arguments must not collide on
    `tool` alone: `analysis.html` keys each progress row by this, and a
    collision means the second call's `tool` event finds nothing to update --
    a row stuck reading "running" for the life of the page. `json.dumps(...,
    sort_keys=True)` rather than a client-side `JSON.stringify` hash, because
    the two do not canonicalise key order the same way and a client-side
    recomputation could silently disagree with the server's for identical
    arguments.

    Content alone is not enough: two byte-identical calls -- `validate_plan`
    has no rule against them -- produce the same base key. See the position
    queue in `ask_stream`, which appends *which* occurrence this is.
    """
    return f"{tool}:{json.dumps(args or {}, sort_keys=True, ensure_ascii=False)}"


@app.get("/ask/stream")
def ask_stream(request: Request, question: str):
    """Run one question, emitting progress as it goes.

    One execution, not two: the same run that produces these events is the run
    that gets persisted, so the page a viewer lands on cannot disagree with the
    progress they just watched.
    """
    if not question.strip():
        raise HTTPException(400, "a question is required")
    # Off the session, exactly as `POST /ask` takes it. It used to be a query
    # parameter with a default, which meant the name beside a stored question
    # was whatever the caller typed into a URL.
    asked_by = operator_of(request).name

    def stream():
        graph = analysis_graph()
        state: dict = {}
        # Declared positions not yet claimed by a `tool` event, grouped by
        # `_tool_base_key`. `validate_plan` allows two byte-identical calls
        # (deliberately not changed here -- see the Task 8 fix-round report),
        # so content alone cannot tell their rows apart. The plan node runs,
        # and is streamed, before any `run_tool` branch: by the time a `tool`
        # event needs a key, every position for its base key is already
        # queued, so popping the smallest one gives each occurrence of an
        # identical call a distinct, stable key -- which of the two identical
        # branches gets which position does not matter, since nothing about
        # them differs.
        pending_positions: dict[str, list[int]] = defaultdict(list)
        # A comment line, sent before anything else. SSE comments (a line
        # starting with `:`) are ignored by `EventSource` but still count as
        # response bytes, which flushes the headers through a buffering proxy
        # immediately rather than leaving the connection looking idle until
        # the first real event -- on a slow plan call, that can be several
        # seconds away.
        yield ": stream-open\n\n"
        try:
            for update in graph.stream(
                {"question": question.strip(), "results": [], "timings_ms": {}},
                stream_mode="updates",
            ):
                for node, payload in update.items():
                    # A node returning an empty dict streams as None. Verified
                    # against LangGraph 1.2 before this plan was written.
                    payload = payload or {}
                    _merge_state(state, payload)
                    if node == "plan":
                        plan = payload.get("plan")
                        if plan is not None:
                            # Only when the planner actually answered. An
                            # unreachable planner has no plan at all -- the
                            # error below still fires -- and announcing
                            # "planning complete" a moment before it would
                            # show the operator something that did not
                            # happen.
                            calls = []
                            for index, call in enumerate(plan.get("calls", [])):
                                base = _tool_base_key(
                                    call.get("tool", ""), call.get("args")
                                )
                                pending_positions[base].append(index)
                                calls.append({**call, "key": f"{index}:{base}"})
                            yield _sse("plan", {
                                "interpretation": plan.get("interpretation", ""),
                                "calls": calls,
                            })
                        for error in payload.get("plan_errors") or []:
                            yield _sse("error", {"message": error})
                    elif node == "run_tool":
                        for result in payload.get("results") or []:
                            base = _tool_base_key(result["tool"], result["args"])
                            queue = pending_positions.get(base)
                            index = queue.pop(0) if queue else 0
                            yield _sse("tool", {
                                "tool": result["tool"],
                                "key": f"{index}:{base}",
                                "ok": result["ok"],
                                # Clipped the same as every other tool-reported
                                # string on the rendered page
                                # (`result_view.clip`, 160 chars): a failing
                                # tool's raw exception text is otherwise
                                # unbounded, and a traceback-length message
                                # becomes one unbounded row in the progress
                                # panel.
                                "error": clip(result["error"]) if result["error"] else None,
                                "elapsed_ms": result["elapsed_ms"],
                            })
                    elif node == "collect":
                        # The join, and the only announcement of the phase that
                        # follows it. `collect` is the node the `Send` branches
                        # converge on: under `stream_mode="updates"` every
                        # `run_tool` branch has already been streamed by the
                        # time this update exists, and `synthesise` -- the
                        # second model call, around eight seconds of it -- has
                        # not been entered yet, because it is the next
                        # superstep and this generator is what advances it.
                        # Verified against the graph rather than assumed: the
                        # update order is plan, run_tool x N, collect,
                        # synthesise.
                        #
                        # Without this event the page had nothing to say for
                        # that whole phase. Every tool row ticked to a ✓, the
                        # panel looked finished, and the run then sat silent
                        # until the redirect -- a visible state saying done
                        # over a system still working, which is the failure
                        # this event exists to remove. It carries the branch
                        # count only so the phase line can name what was
                        # joined; nothing here is a claim about time.
                        yield _sse("synthesising", {
                            "tools": len(state.get("results") or []),
                        })
            # Persisted inside the try: if `save_run` itself raises, the
            # `except` below still turns that into one clean `error` event
            # instead of an uncaught exception unwinding the ASGI response
            # mid-stream. An abrupt end with no `done` event is indistinguish-
            # able, to `EventSource`, from a dropped connection -- it
            # reconnects and reruns the whole question, looping for as long as
            # the tab stays open. See the `error` handler in analysis.html,
            # which closes the connection on exactly that native error.
            # Through `analysis_service`, the same function `POST /ask` uses.
            # Not an inlined `save_run` again: this path observes the graph in
            # completion order, so the normalising that puts the results back
            # in plan order lives with the write rather than in whichever of
            # the two callers remembered it.
            run_id = analysis_service.persist_run(state, question.strip(), asked_by)
        except Exception as error:  # noqa: BLE001 -- the stream must close cleanly
            yield _sse("error", {"message": f"{type(error).__name__}: {error}"})
            return
        # A client that navigates away mid-run raises `GeneratorExit` at
        # whichever `yield` was in flight above. It is not an `Exception`, so
        # the `except` above never sees it: it propagates straight out of this
        # generator and `save_run` is never reached. That is deliberate, not a
        # gap to close with a `finally` -- the run already cost two model
        # calls with nobody left to read the answer, and there is no run to
        # attach a `done` event to. Persisting it anyway would be a background
        # write nobody asked for.
        yield _sse("done", {"run_id": run_id})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            # Without these, a buffering reverse proxy (nginx by default, and
            # anything shop-floor between the browser and this process) can
            # hold the whole response until it closes, which turns the stream
            # back into the blank-screen wait this feature exists to remove.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/ask/{run_id}", response_class=HTMLResponse)
def ask_result(request: Request, run_id: int):
    """A saved run, re-rendered from the store. Nothing here calls a model."""
    run = analysis_store.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"no analysis run {run_id}")
    return templates.TemplateResponse(
        request, "analysis.html", _analysis_context(run)
    )
