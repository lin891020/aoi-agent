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

import jinja2
from markupsafe import Markup

from aoi_agent.analysis import service as analysis_service
from aoi_agent.analysis.graph import build_analysis_graph
from aoi_agent.analysis.plan import PLANNABLE_TOOLS, store_domains
from aoi_agent.graph.flow import DEFAULT_MODEL, build_graph, explanation_notice
from aoi_agent.provenance import UNAVAILABLE, UNRECORDED, ReviewerIdentity
from aoi_agent.llm.ollama import OllamaClient
from aoi_agent.station import auth, images, service
from aoi_agent.station.chart_svg import render_svg
from aoi_agent.i18n import (
    LOCALE_COOKIE,
    LOCALES,
    STRINGS,
    normalise,
    translate,
)
from aoi_agent.station.prose import blocks as prose_blocks, lead_and_rest
from aoi_agent.station import timing_view
from aoi_agent.station.result_view import clip, error_text, readable_rows, shown_count
from aoi_agent.store import analysis as analysis_store
from aoi_agent.store import dispositions, escalations
from aoi_agent.store.boards import (
    correction_count,
    correction_summary,
    corrections,
    resolve_candidate,
)

HERE = Path(__file__).parent

app = FastAPI(title="AOI re-verification station")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


@jinja2.pass_context
def _t(context, key: str, **args: object) -> str:
    """`t("queue.title")` in any template, in this request's language.

    A context function rather than a plain global, so the locale comes off the
    request that is being rendered instead of being threaded through every
    route's context dict by hand. Starlette puts `request` in every template
    context, and one route forgetting to pass a locale is exactly the drift the
    key-parity test cannot see.
    """
    request = context.get("request")
    locale = getattr(request.state, "locale", None) if request is not None else None
    return translate(key, locale, **args)


@jinja2.pass_context
def _here(context) -> str:
    """The path this request is on, for a link that comes back to it.

    Path and query, never the host: it is fed to `_safe_next`, which refuses
    anything that is not a path on this station, and building the value here
    the same way keeps the two ends agreeing.
    """
    request = context.get("request")
    if request is None:
        return "/"
    return f"{request.url.path}?{request.url.query}" if request.url.query \
        else request.url.path


@jinja2.pass_context
def _strings_for(context, prefix: str) -> str:
    """Every string under `prefix`, as JSON for a `<script type="application/
    json">` block the page parses with `textContent`.

    The browser reads the same table the server renders from. A second list of
    translations kept in step by hand is the drift the key-parity test exists
    to catch, and it cannot see one that lives in JavaScript.
    """
    request = context.get("request")
    locale = getattr(request.state, "locale", None) if request is not None else None
    table = STRINGS[normalise(locale)]
    return _script_json(
        {key: value for key, value in table.items() if key.startswith(prefix)}
    )


def _script_json(value: object) -> Markup:
    """JSON for a `<script type="application/json">` block.

    Kept in one function because the safety is the encoding rather than the
    marking, and two call sites encoding it two ways is how one of them stops
    being safe.
    """
    encoded = json.dumps(value, ensure_ascii=False)
    # A `<script>` element's content is raw text: HTML entities inside it are
    # not decoded, so Jinja's autoescaping would put `&quot;` where the quotes
    # belong and `JSON.parse` would fail on every page. It has to go out
    # unescaped -- and the reason that is safe is this line, not the marking:
    # `<`, `>` and `&` become `\uXXXX`, which JSON reads back as the same
    # characters and which cannot spell `</script`. So no string in the table,
    # whatever it is changed to later, can close the element.
    for character in ("<", ">", "&"):
        encoded = encoded.replace(character, f"\\u{ord(character):04x}")
    return Markup(encoded)


templates.env.globals["t"] = _t
templates.env.globals["here"] = _here
templates.env.globals["strings_for"] = _strings_for
templates.env.globals["lead_and_rest"] = lead_and_rest
#: What the switch offers. Each language names itself in itself -- somebody
#: looking for English does not read 英文, and somebody looking for Chinese does
#: not read "Chinese".
templates.env.globals["locale_choices"] = tuple(
    (code, {"zh-TW": "中文", "en": "English"}[code]) for code in LOCALES
)

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
#: `/locale` is here so the sign-in page can be read in either language --
#: it writes a display preference and nothing else, and a person who cannot
#: read the login form cannot sign in to fix that.
PUBLIC_PREFIXES = ("/login", "/locale", "/static", "/favicon.ico")


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
    # A cookie rather than the session: a preference about how to read the
    # screen is not a fact about who is signed in, and it should survive a
    # sign-out on a shared shop-floor terminal.
    request.state.locale = normalise(request.cookies.get(LOCALE_COOKIE))

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


def locale_of(request: Request) -> str:
    """The language this request is being read in.

    `getattr` with a fallback, like `operator_of` below and `_t` above: the
    middleware sets it on every real request, and a route driven directly --
    which the stream tests do, because `TestClient.stream` buffers the whole
    body -- should still produce a run rather than an `AttributeError`.
    """
    return normalise(getattr(request.state, "locale", None))


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


@app.get("/locale/{code}")
def set_locale(code: str, next: str = "/"):
    """Switch the interface language and come back to the same page.

    A `GET` because it is a link in the header and has to work with scripting
    off. It changes nothing but a display preference -- no record is written,
    no board moves -- so it does not want a form and a token.

    An unknown code lands on the default rather than 404ing: this is reachable
    by typing a URL, and a wrong one should leave a working station in the
    language it started in.
    """
    response = RedirectResponse(_safe_next(next), status_code=303)
    response.set_cookie(
        LOCALE_COOKIE, normalise(code),
        max_age=60 * 60 * 24 * 365, httponly=False, samesite="lax",
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
    """The queue, and an honest statement of how much of it is on screen.

    ``waiting`` is a ``COUNT(*)`` and ``queue`` is at most one page of rows.
    They were one number until 2026-08-25 -- ``len(escalations.pending())`` --
    and ``pending`` caps at 200, so a queue of 250 rendered "200 waiting" with
    nothing on screen saying so. Reproduced before it was fixed: 250 in, 200
    shown, 50 people waiting invisibly. At this project's own throughput
    figures an unattended queue crosses 200 in about a quarter of an hour, so
    this was not a scale problem for later.

    The same applies to ``unexplained``: computed over a screenful it cannot
    see the shift it exists to make visible.
    """
    queue = escalations.pending()
    waiting = escalations.pending_count()
    return templates.TemplateResponse(
        request,
        "queue.html",
        {
            "queue": queue,
            "waiting": waiting,
            "unexplained": escalations.pending_unexplained_count(),
            "deferred_count": escalations.deferred_count(),
            # Not `waiting > len(queue)` at the template: the template should be
            # given the fact, not the arithmetic that derives it.
            "not_shown": max(waiting - len(queue), 0),
        },
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
            # Whether this region can still take an answer -- the fact, not the
            # rule. The template asked `status == 'pending'` in two places, so
            # a deferred region rendered as "already answered" with no form on
            # it: deferring it made it unanswerable in the markup as well as in
            # the route. One source for the set, and both readers ask it.
            "answerable": bool(escalation)
            and escalation["status"] in ANSWERABLE,
            # Whether *this* operator may answer it. The route refuses a
            # non-senior verdict on a handed-back region with a 403; until
            # 2026-08-28 the page still drew the seven buttons for them, so
            # the refusal arrived after the click instead of before it.
            "may_answer": not escalation
            or escalation["status"] != escalations.DEFERRED
            or auth.role_of(getattr(request.state, "operator", None)) == auth.SENIOR,
            "state": state,
            # Rendered from the status, never stored as though the model had
            # written it. WI-300: an absent rationale is absent, and the gap is
            # not to be filled by any other means.
            "explanation_notice": explanation_notice(
                (escalation or {}).get("explanation_status") or ""
            ),
            "options": service.VERDICT_OPTIONS,
            # The triptych's gap, as a fraction of its width. Derived from the
            # constants that render it rather than written down twice: the
            # ruler refuses a segment that crosses the gap, and a stale copy of
            # this number would make it refuse the wrong ones.
            "gap_fraction": images.PANEL_GAP / (
                images.CONTEXT_SIZE * images.SCALE * 3 + images.PANEL_GAP * 2
            ),
            "waiting": escalations.pending_count(),
            # Who has already declined this region, shown to whoever opens it
            # next. Somebody who declined it yesterday should not spend five
            # minutes rediscovering that, and a note saying what they could not
            # tell is the closest thing this station has to handing over a case.
            "declines": escalations.declines_for(service.thread_for(reference)),
            "next_reference": _next_pending(after=reference),
        },
    )


#: Queue states an answer may still be recorded against.
#:
#: ``deferred`` belongs here and its absence was a real defect for the length of
#: one commit: this route tested ``status != "pending"`` and a region somebody
#: had declined became permanently unanswerable through the station, silently,
#: by redirect. A deferral is supposed to move a region to someone else, not
#: strand it, and the store-level tests could not see this because the store was
#: never the thing refusing.
ANSWERABLE = (escalations.PENDING, escalations.DEFERRED)


@app.post("/c/{stem}/{index}/verdict")
def submit_verdict(
    request: Request,
    stem: str,
    index: int,
    verdict: str = Form(...),
    measurement: str = Form(""),
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
    if (
        escalation is not None
        and escalation["status"] == escalations.DEFERRED
        and auth.role_of(getattr(request.state, "operator", None)) != auth.SENIOR
    ):
        # The one permission this station models. A region reaches this state
        # because a trained person looked at it and said they could not read
        # it, so handing it to the next ordinary operator is handing it back to
        # the same judgement -- and the failure that would produce is a guess
        # recorded as a label, which is what the whole deferral path exists to
        # prevent. Refused rather than redirected: a redirect here is
        # indistinguishable from "somebody else got there first", and an
        # operator who is not allowed to answer needs to be told that.
        raise HTTPException(403, f"answering a handed-back region needs {auth.SENIOR}")
    if escalation is None or escalation["status"] not in ANSWERABLE:
        # Two operators opened the same region, or a back button was pressed.
        # The first answer stands; recording a second would put two labels on
        # one region and silently corrupt the training set.
        return RedirectResponse("/next", status_code=303)

    # The reading, if the operator took one. Unlike `verdict` this is not
    # validated against a vocabulary -- it is free text describing what was
    # measured -- so it is length-capped at the column and never parsed back
    # into a decision. Nothing routes on it; it is evidence beside the answer.
    service.resume_review(
        graph(), reference, verdict, operator_of(request),
        measurement=(measurement or "").strip()[:256] or None,
    )
    return RedirectResponse(f"/next/{stem}/{index}", status_code=303)


@app.post("/c/{stem}/{index}/defer")
def submit_deferral(request: Request, stem: str, index: int, note: str = Form("")):
    """Record that this operator could not judge the region, and move them on.

    A separate route from ``submit_verdict`` rather than an eighth entry in
    ``VERDICT_OPTIONS``, which would have been fewer lines. The verdict route
    ends in ``record_decision``; a string that can reach it is a string that can
    become a training label, and "unsure" is the one label that must never be
    one. Two routes because they are two different acts.

    The name comes off the session, like the verdict's. The note does not -- it
    is what the operator typed -- so it is capped and stored as text nothing
    parses.
    """
    reference = _reference(stem, index)
    _candidate_or_404(stem, index)

    service.defer_review(
        reference, operator_of(request), (note or "").strip()[:2000] or None
    )
    return RedirectResponse(f"/next/{stem}/{index}", status_code=303)


@app.get("/deferred", response_class=HTMLResponse)
def deferred_page(request: Request):
    """Regions nobody could judge, hardest-looking first.

    Its own page rather than a section of the queue. These are not waiting for
    the next operator in the ordinary sense -- an operator who works down this
    list and declines everything on it has done the right thing twice, and
    burying them in the main queue would put them back in front of the person
    who already said no.

    **What this page does not do is route them to anyone**, because this station
    has no notion of who is more senior than whom -- every operator can answer
    every region. So this is a list, honestly, and not an assignment. The
    decline count is the only ranking available and it is a real one: a region
    three people declined is a different object from one somebody skipped.
    """
    rows = escalations.deferred()
    return templates.TemplateResponse(
        request,
        "deferred.html",
        {
            "rows": rows,
            "waiting": escalations.pending_count(),
            "total": escalations.deferred_count(),
            "not_shown": max(escalations.deferred_count() - len(rows), 0),
            # Visible to everyone -- seeing is not doing, and a list only
            # seniors can see is a list nobody knows is growing. Whether *this*
            # reader may act on it is a separate fact and the page states it.
            "may_answer": auth.role_of(
                getattr(request.state, "operator", None)
            ) == auth.SENIOR,
            # No senior configured at all is the state where this queue grows
            # with nobody able to empty it and nothing anywhere raising. Said on
            # the page, because the person who needs to read it is looking at
            # the page and not at a log.
            "seniors": auth.seniors(),
        },
    )


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


@app.get("/boards", response_class=HTMLResponse)
def boards_page(request: Request, status: str | None = None, limit: int = 50):
    """Every board this system has dispositioned, and which way.

    The station showed the queue -- the regions the agent could *not* settle --
    and one board page reachable only by a link from a queued region. So the
    82% it did settle had no route into the UI at all, and a reviewer opening
    the station read the failures and took them for the system. This is the
    index that page was always missing.

    Read-only, like the page it links to. Nothing here dispositions anything.

    ``status`` is refused rather than ignored when it is not a disposition this
    store writes: a filter that silently matches everything returns a plausible
    page for a typed URL, and the count above it would then be a true number
    answering a question nobody asked.
    """
    known = (dispositions.HELD, dispositions.RELEASED, dispositions.WAITING)
    if status is not None and status not in known:
        raise HTTPException(
            400,
            f"unknown disposition {status!r}; expected one of {', '.join(known)}",
        )
    counts = dispositions.board_counts()
    if status == dispositions.WAITING:
        # Not a disposition: these boards have no row, and the list is the
        # boards with a region on the queue. Beside the two counts rather
        # than inside them, so the reader who ran fifty and sees twenty-nine
        # is told where the rest are instead of left to wonder.
        rows = dispositions.waiting(limit=limit)
    else:
        rows = dispositions.recent(limit=limit, status=status)
    total = counts[status] if status else counts["total"]
    return templates.TemplateResponse(
        request,
        "boards.html",
        {
            "rows": rows,
            "counts": counts,
            "status": status,
            "total": total,
            "not_shown": max(total - len(rows), 0),
            "waiting": escalations.pending_count(),
        },
    )


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
            "waiting": escalations.pending_count(),
        },
    )


@app.get("/queue-count", response_class=HTMLResponse)
def queue_count():
    """Polled by the station header so an operator sees work arriving.

    A ``COUNT(*)``. This badge is on every page, so when it counted the length
    of a capped list it was the same wrong number in five places at once.
    """
    return HTMLResponse(str(escalations.pending_count()))

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
    shown = corrections(200)
    return templates.TemplateResponse(
        request,
        "corrections.html",
        {
            "rows": shown,
            # Over every human decision, not over `shown`. The list is a page;
            # the aggregate is the claim.
            "summary": correction_summary(),
            "waiting": escalations.pending_count(),
            "not_shown": max(correction_count() - len(shown), 0),
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


def _flow_events(run: dict) -> list[dict]:
    """A finished run, as the event sequence the flow view reads.

    Rebuilt rather than stored: the events were never a record, the plan and
    the results are, and a second copy of them in another shape would be a
    second thing to keep true.
    """
    plan = run.get("plan") or {}
    calls = plan.get("calls") or []
    events: list[dict] = [{
        "event": "plan",
        "data": {"interpretation": plan.get("interpretation", ""),
                 "calls": [{"tool": c.get("tool", ""), "args": c.get("args") or {}}
                           for c in calls]},
    }]
    for position, result in enumerate(run.get("results") or []):
        events.append({"event": "tool", "data": {
            "tool": result.get("tool", ""),
            "args": result.get("args") or {},
            "ok": bool(result.get("ok")),
            "elapsed_ms": result.get("elapsed_ms", 0),
            "position": result.get("position", position),
        }})
    if calls and not run.get("refused"):
        events.append({"event": "synthesising", "data": {}})
    events.append({"event": "done", "data": {"run_id": run.get("id")}})
    return events


def _analysis_context(run: dict | None, locale: str | None = None) -> dict:
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
        # The rules block: what can be asked, read off the registry at
        # request time so a tool added there appears here without anyone
        # remembering to list it. The sentences are `tool.<name>.does` in
        # the reader's language; the key-parity test holds both tables.
        "capabilities": list(PLANNABLE_TOOLS),
        "recent": analysis_store.recent_runs(8),
        # The same snapshot the validator was built from, not a fresh read:
        # see `analysis_domains`. A page that states a coverage the validator
        # does not enforce is worse than a page whose coverage is a restart
        # behind.
        "coverage_days": analysis_domains()["max_days"],
        # In the reader's language: the spec stores keys, not sentences, so
        # the same stored chart reads in whichever language the page is in.
        # Until 2026-08-28 the locale was not passed and the chart came out
        # in the default language under an English heading.
        "chart_svg": render_svg(run["chart"], locale=locale) if run and run.get("chart") else "",
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
        # The same events the live panel was fed, rebuilt from what was
        # stored. The diagram is a pure function of them, so a run reopened
        # next quarter draws the shape its plan actually took rather than
        # nothing at all -- and the fan-out stops being visible only to
        # whoever happened to be watching for those twenty-five seconds.
        "flow_events": _script_json(_flow_events(run)) if run else None,
        # The answer's Markdown, as blocks of spans. A callable for the same
        # reason as `readable_data`: the parsing is the boundary and it lives in
        # Python. What comes back is structure with no field a tag could travel
        # in, which is why the template renders it with no `|safe` -- see
        # `station/prose.py`.
        "prose_blocks": prose_blocks,
        # The stage table under the answer: what the page waited at each
        # stage, and what the model reported as its own inference time.
        "timing_rows": timing_view.rows(run.get("timings")) if run else [],
        "waiting": escalations.pending_count(),
    }


@app.get("/ask", response_class=HTMLResponse)
def ask_page(request: Request):
    return templates.TemplateResponse(
        request, "analysis.html", _analysis_context(None, locale_of(request))
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
        analysis_graph(), question.strip(), operator_of(request).name,
        asked_lang=locale_of(request),
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
    # Read here rather than inside `stream()`: the generator outlives the
    # request object's convenient lifetime, and the language a run is recorded
    # under has to be the one the asker was reading.
    asked_lang = locale_of(request)

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
                # `lang` in the initial state, as `service.answer_question`
                # passes it. Until 2026-08-28 this path left it out, so a
                # question asked from the English station was recorded under
                # `en` and planned and written in the default language: the
                # stored run said one thing and every sentence on it another.
                {"question": question.strip(), "results": [], "timings_ms": {},
                 "lang": asked_lang},
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
            run_id = analysis_service.persist_run(
                state, question.strip(), asked_by, asked_lang=asked_lang)
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
        request, "analysis.html", _analysis_context(run, locale_of(request))
    )
