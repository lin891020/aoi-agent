"""A human decision that cannot name who made it is not a record either.

The provenance work put the automated half of this store beyond doubt: a
``model`` or ``agent`` row names the weights, the operating point and the
commit behind it, and ``record_decision`` refuses one that does not. The human
half was untouched, and the reviewer who found the first gap found this one in
the same breath -- all 9,140 decisions read ``reviewer = NULL``, ``reviewer``
was free text anyway, and five escalations stood closed with no human decision
beneath them at all.

That is not an access-control complaint. It is a retraining complaint. This
project's entire feedback story is that operator corrections become the next
round's labels -- it is why the station refuses to show ``ground_truth`` -- and
a label carrying no trustworthy identity is a label nobody can weigh. It has
already cost this store five rows: five regions clicked through by somebody
without the domain knowledge to judge them, four of the five wrong, and no
query anywhere that could separate them from an expert's. They had to be
deleted by hand.

Four properties are held here.

1. **A human decision cannot be written without an attributable reviewer.**
   The same word as the automated rule, for the same reason: cannot, not
   should. ``record_decision`` raises, and ``resume_review`` raises earlier
   still so that a refused answer does not consume the interrupt.
2. **The name comes from a mechanism, never from a form.** The station reads
   it off a signed session and ignores anything posted alongside. A field the
   browser can fill is a field that names whoever it likes.
3. **Both pages are behind that session.** The queue and ``/ask`` -- the
   second being the one that turned this from a backlog item into a
   precondition, because a query interface over production statistics is a
   different exposure from a list of regions.
4. **The absences are told apart.** ``automated`` is a positive statement that
   no person was involved; ``unrecorded`` is the migration's mark on rows that
   predate the column. Neither is ``NULL``, and the 9,140 rows this store held
   are therefore distinguishable from a model decision written today.

No Ollama and no GPU: the model is stubbed the way the rest of the suite stubs
it, and the credential file is generated per test at a thousand PBKDF2
iterations instead of the shipped six hundred thousand -- the count is stored
in the record, so this is the real code path over a cheap record.
"""

from __future__ import annotations

import time
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, select, text

from aoi_agent.provenance import (
    AUTOMATED,
    HOST_ACCOUNT,
    SIGNED_IN,
    UNRECORDED,
    DecisionProvenance,
    ReviewerIdentity,
)
from aoi_agent.station import app as station_app
from aoi_agent.station import auth, service
from aoi_agent.store import boards, escalations
from aoi_agent.store.models import (
    Board,
    CandidateRecord,
    ReviewDecision,
    create_all,
    make_session_factory,
)
from conftest import TEST_OPERATOR, TEST_SECRET, sign_in
from test_graph import STUB_DIGEST, StubClient, stub_tools  # noqa: F401  (fixture)
from test_station import REFERENCE, STEM, graph, store  # noqa: F401  (fixtures)

PROVENANCE = DecisionProvenance(
    model_digest=STUB_DIGEST, thresholds={"dismiss": 0.915}, code_version="test"
)


@pytest.fixture
def anonymous(store, graph, monkeypatch, operators):  # noqa: F811
    """A client that knows the station exists and has not signed in."""
    monkeypatch.setattr(station_app, "_graph", graph)
    return TestClient(station_app.app)


@pytest.fixture
def client(anonymous):
    """The same client, signed in as the suite's operator."""
    return sign_in(anonymous)


# ---- what makes a decision attributable ---------------------------------


def test_a_human_decision_cannot_be_written_without_an_identity(store):  # noqa: F811
    with pytest.raises(ValueError, match="attributable reviewer"):
        boards.record_decision(REFERENCE, "short", "human")


@pytest.mark.parametrize(
    "identity",
    [
        ReviewerIdentity(name=None, method=SIGNED_IN),
        ReviewerIdentity(name="   ", method=SIGNED_IN),
        ReviewerIdentity(name="mike", method=AUTOMATED),
        ReviewerIdentity(name="mike", method=UNRECORDED),
        ReviewerIdentity(name="mike", method="trust me"),
    ],
)
def test_a_name_with_nothing_behind_it_is_not_an_identity(store, identity):  # noqa: F811
    """A method is half the record. ``mike`` typed into a box and ``mike`` read
    off a signed session are the same four characters and not the same claim,
    and the second half is exactly what a retraining export selects on."""
    with pytest.raises(ValueError, match="attributable reviewer"):
        boards.record_decision(REFERENCE, "short", "human", identity=identity)
    with store() as session:
        assert session.execute(select(ReviewDecision)).first() is None, (
            "the row was refused and must not have been written anyway"
        )


def test_a_signed_in_answer_carries_the_name_and_how_it_was_established(store):  # noqa: F811
    boards.record_decision(
        REFERENCE, "short", "human", identity=ReviewerIdentity.signed_in("mike")
    )

    with store() as session:
        row = session.execute(select(ReviewDecision)).scalar()
    assert row.reviewer == "mike"
    assert row.reviewer_auth == SIGNED_IN


def test_the_cli_names_the_host_account_rather_than_a_literal(store):  # noqa: F811
    """It used to write ``operator``, or ``auto`` for a scripted answer, both of
    which name nobody. The OS account is a weaker claim than a signed-in
    operator and is recorded as a different word rather than as the same one."""
    identity = ReviewerIdentity.host_account()
    assert identity.is_attributable
    boards.record_decision(REFERENCE, "short", "human", identity=identity)

    with store() as session:
        row = session.execute(select(ReviewDecision)).scalar()
    assert row.reviewer == identity.name
    assert row.reviewer_auth == HOST_ACCOUNT


def test_a_host_with_no_account_name_cannot_answer(store):  # noqa: F811
    """Fail closed. A container with no passwd entry gets no name, and a
    nameless answer is refused rather than written as an empty string."""
    nameless = ReviewerIdentity.host_account("")
    assert not nameless.is_attributable
    with pytest.raises(ValueError, match="attributable reviewer"):
        boards.record_decision(REFERENCE, "short", "human", identity=nameless)


def test_an_automated_decision_says_nobody_reviewed_it(store):  # noqa: F811
    boards.record_decision(REFERENCE, "open", "agent", provenance=PROVENANCE)

    with store() as session:
        row = session.execute(select(ReviewDecision)).scalar()
    assert row.reviewer is None
    assert row.reviewer_auth == AUTOMATED, (
        "a model row has no reviewer *by construction*, which is a different "
        "statement from a row whose reviewer was never recorded"
    )


def test_an_automated_decision_cannot_borrow_the_name_of_whoever_ran_it(store):  # noqa: F811
    """Nobody reviewed it. Attributing it to the operator whose session
    happened to start the run would make a model's verdict look like a
    person's, which is the same corruption as the anonymous label, upside
    down."""
    with pytest.raises(ValueError, match="Nobody reviewed it"):
        boards.record_decision(
            REFERENCE, "open", "agent",
            identity=ReviewerIdentity.signed_in("mike"), provenance=PROVENANCE,
        )


def test_a_source_that_is_neither_is_refused(store):  # noqa: F811
    """Which of the two rules applies is decided by the source, so an unknown
    source would slip past both."""
    with pytest.raises(ValueError, match="not a decision source"):
        boards.record_decision(REFERENCE, "short", "script")


def test_an_unattributable_answer_does_not_consume_the_escalation(store, graph):  # noqa: F811
    """Refused in ``resume_review`` as well, and before the graph is resumed.
    Refusing only at the store would leave the interrupt spent and the region
    off the queue with no verdict anywhere -- which is precisely the shape of
    the five rows this project had to mark as ``resolved_unattributed``."""
    service.start_review(graph, REFERENCE)

    with pytest.raises(ValueError, match="unattributable"):
        service.resume_review(
            graph, REFERENCE, "short", ReviewerIdentity(name="", method=SIGNED_IN)
        )

    assert [row["reference"] for row in escalations.pending()] == [REFERENCE]
    assert boards.corrections() == []


def test_a_correction_carries_what_a_retraining_round_would_select_on(store):  # noqa: F811
    boards.record_decision(
        REFERENCE, "short", "human", identity=ReviewerIdentity.signed_in("mike")
    )
    assert boards.corrections()[0]["attribution"] == SIGNED_IN


# ---- the rows that predate the column ------------------------------------


def _store_without_the_column(tmp_path, decisions: int = 1):
    """A store as it stood before ``reviewer_auth``, with a board and rows in it.

    Built forwards and then taken back one column, rather than by hand-writing
    the old ``CREATE TABLE``: what is under test is the migration against a real
    store, and a hand-written table drifts from the model the moment anything
    else changes. The decisions go in through raw SQL for the same reason the
    column is dropped -- the ORM knows about a column this store does not have.
    """
    url = f"sqlite:///{tmp_path / 'old.db'}"
    create_all(url)
    factory = make_session_factory(url)
    with factory() as session:
        _seed_board(session)
        session.commit()

    engine = create_engine(url, future=True)
    with engine.begin() as connection:
        connection.execute(text("DROP INDEX ix_review_decisions_reviewer_auth"))
        connection.execute(text("ALTER TABLE review_decisions DROP COLUMN reviewer_auth"))
        for _ in range(decisions):
            connection.execute(text(
                "INSERT INTO review_decisions (candidate_id, verdict, source)"
                " VALUES (1, 'open', 'model')"
            ))
    return url, factory


def test_the_column_reaches_a_store_that_already_holds_decisions(tmp_path):
    """9,140 of them, in the store this was written against. Rebuilding to gain
    a column would throw away the corrections the column exists to qualify."""
    url, factory = _store_without_the_column(tmp_path)

    create_all(url)

    columns = {
        c["name"] for c in inspect(create_engine(url)).get_columns("review_decisions")
    }
    assert "reviewer_auth" in columns
    with factory() as session:
        assert session.execute(select(ReviewDecision)).scalar().reviewer_auth == UNRECORDED


def test_a_row_written_before_the_column_is_not_a_row_that_names_nobody(tmp_path,
                                                                       monkeypatch):
    """The distinction the whole migration is for.

    ``NULL`` would have to carry two meanings: "written before anyone recorded
    how a reviewer was identified" and "no reviewer, because no person was
    involved". The second is a fact worth stating about a model decision -- and
    every one of the 9,140 rows this store held has ``reviewer`` NULL, so
    without the distinction the two are the same row. The migration stamps the
    first meaning and ``record_decision`` writes the second, which leaves
    ``NULL`` meaning nothing at all.
    """
    url, factory = _store_without_the_column(tmp_path)

    create_all(url)
    monkeypatch.setattr(boards, "_session_factory", factory)
    boards.record_decision(REFERENCE, "short", "model", provenance=PROVENANCE)

    with factory() as session:
        rows = session.execute(
            select(ReviewDecision).order_by(ReviewDecision.id)
        ).scalars().all()
    assert [row.reviewer for row in rows] == [None, None]
    assert [row.reviewer_auth for row in rows] == [UNRECORDED, AUTOMATED]


def test_no_row_is_left_with_a_null_attribution(tmp_path, monkeypatch):
    url, factory = _store_without_the_column(tmp_path, decisions=3)

    create_all(url)
    monkeypatch.setattr(boards, "_session_factory", factory)
    boards.record_decision(
        REFERENCE, "short", "human", identity=ReviewerIdentity.signed_in("mike")
    )

    with factory() as session:
        nulls = session.execute(
            select(ReviewDecision).where(ReviewDecision.reviewer_auth.is_(None))
        ).scalars().all()
    assert nulls == []


def _seed_board(session) -> None:
    board = Board(
        stem=STEM, split="test", lot_id="LOT-2201", line_id="L2",
        machine_id="M22", shift="A", inspected_at=datetime(2026, 8, 20, 9, 0),
    )
    session.add(board)
    session.flush()
    session.add(
        CandidateRecord(
            board_id=board.id, index_on_board=0,
            x1=100, y1=120, x2=140, y2=155, area=1400,
            predicted_class="open", confidence=0.55, false_call_probability=0.45,
        )
    )
    session.flush()


# ---- the credential file -------------------------------------------------


def test_the_file_holds_no_passphrase():
    encoded = auth.hash_secret("hunter2", iterations=1000)
    assert "hunter2" not in encoded
    assert encoded.startswith("pbkdf2_sha256$1000$")


def test_the_same_passphrase_twice_is_two_different_records():
    """Salted per record, so the file does not say which two operators chose
    the same passphrase."""
    first = auth.hash_secret("hunter2", iterations=1000)
    second = auth.hash_secret("hunter2", iterations=1000)
    assert first != second
    assert auth.verify_secret("hunter2", first)
    assert auth.verify_secret("hunter2", second)


def test_the_iteration_count_travels_with_the_record():
    """So the cost can be raised without invalidating a file -- and so this
    suite can use a cheap record without testing a different code path."""
    cheap = auth.hash_secret("hunter2", iterations=1000)
    dearer = auth.hash_secret("hunter2", iterations=2000)
    assert auth.verify_secret("hunter2", cheap)
    assert auth.verify_secret("hunter2", dearer)


@pytest.mark.parametrize(
    "record", ["", "nonsense", "md5$1000$aa$bb", "pbkdf2_sha256$notanumber$aa$bb"]
)
def test_a_malformed_record_locks_that_operator_out_and_nothing_else(record):
    """A typo in the file must not be an exception on the login route: one bad
    line would otherwise take the station down for everyone."""
    assert auth.verify_secret("hunter2", record) is False


def test_an_unknown_operator_and_a_wrong_passphrase_both_fail(operators):
    assert auth.authenticate("nobody", TEST_SECRET) is None
    assert auth.authenticate(TEST_OPERATOR, "not the passphrase") is None
    assert auth.authenticate(TEST_OPERATOR, TEST_SECRET).name == TEST_OPERATOR


def test_a_missing_operator_file_signs_nobody_in(tmp_path, monkeypatch):
    """Fail closed. A station that opened itself when its credential file went
    missing would be one deleted file away from the state this work ends."""
    monkeypatch.setenv(auth.OPERATORS_ENV, str(tmp_path / "not-there"))
    assert auth.load_operators() == {}
    assert auth.authenticate("mike", "anything") is None


def test_a_signed_in_operator_is_what_the_store_records(operators):
    identity = auth.authenticate(TEST_OPERATOR, TEST_SECRET)
    assert identity.method == SIGNED_IN
    assert identity.is_attributable


def test_the_script_writes_a_file_only_its_owner_can_read(tmp_path, monkeypatch):
    """It writes credentials. `0600` is the protection this scheme has left
    once the passphrase is hashed, and the file is gitignored for the same
    reason."""
    import add_operator

    path = tmp_path / "operators"
    monkeypatch.setenv(auth.OPERATORS_ENV, str(path))

    assert add_operator.main(["mike", "--secret", "a passphrase"]) == 0

    assert path.stat().st_mode & 0o777 == 0o600
    assert "a passphrase" not in path.read_text()
    assert auth.authenticate("mike", "a passphrase").name == "mike"


def test_the_script_replaces_and_removes_without_touching_anyone_else(tmp_path,
                                                                     monkeypatch):
    import add_operator

    monkeypatch.setenv(auth.OPERATORS_ENV, str(tmp_path / "operators"))
    add_operator.main(["mike", "--secret", "one"])
    add_operator.main(["sara", "--secret", "two"])
    add_operator.main(["mike", "--secret", "three"])

    assert auth.authenticate("mike", "one") is None
    assert auth.authenticate("mike", "three").name == "mike"

    assert add_operator.main(["mike", "--remove"]) == 0
    assert auth.authenticate("mike", "three") is None
    assert auth.authenticate("sara", "two").name == "sara", (
        "removing one operator must not disturb another"
    )


# ---- the session ---------------------------------------------------------


def test_a_session_names_the_operator_who_signed_in(operators):
    assert auth.operator_from_session(auth.issue_session("mike")) == "mike"


def test_a_session_cannot_be_edited_into_somebody_else(operators):
    """The name on the label comes from this value, so what the signature buys
    is that the value cannot be rewritten -- not that it cannot be read."""
    token = auth.issue_session("mike")
    encoded, expires, signature = token.split(".")
    import base64

    forged = base64.urlsafe_b64encode(b"supervisor").decode()
    assert auth.operator_from_session(f"{forged}.{expires}.{signature}") is None


@pytest.mark.parametrize("token", ["", "a.b.c", "not-a-token", "a.b", "a.b.c.d"])
def test_a_token_that_is_not_one_carries_no_operator(operators, token):
    assert auth.operator_from_session(token) is None


def test_a_session_expires(operators):
    stale = auth.issue_session("mike", now=time.time() - auth.SESSION_MAX_AGE_S - 10)
    assert auth.operator_from_session(stale) is None


def test_a_session_signed_with_another_key_is_refused(operators, monkeypatch):
    token = auth.issue_session("mike")
    monkeypatch.setenv(auth.SECRET_ENV, "a different key entirely")
    assert auth.operator_from_session(token) is None


# ---- the door ------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/",                      # the queue: the regions on one line
        "/c/20085293/0",          # one region, with its images and context
        "/corrections",           # who has been overruling the model
        "/board/20085293",        # a board's quality record
        "/ask",                   # production statistics for the whole plant
        "/queue-count",
    ],
)
def test_no_page_answers_a_visitor_who_has_not_signed_in(anonymous, path):
    """``/ask`` is the one that made this a precondition rather than a backlog
    item -- a query interface over the plant's production data is a different
    exposure from a list of flagged regions -- but the rule is the same rule,
    and it is an allowlist so that a route added tomorrow is behind it too."""
    response = anonymous.get(path, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/login")


def test_the_stream_is_behind_it_too(anonymous):
    response = anonymous.get("/ask/stream?question=anything", follow_redirects=False)
    assert response.status_code == 303
    assert "text/event-stream" not in response.headers.get("content-type", "")


def test_a_verdict_from_nobody_is_refused_and_writes_nothing(anonymous, graph):  # noqa: F811
    """A POST gets a flat 401 rather than a redirect: bouncing it to the login
    page would drop the form body, and an operator who thought they had
    answered a region would have answered nothing."""
    service.start_review(graph, REFERENCE)

    response = anonymous.post(
        f"/c/{STEM}/0/verdict", data={"verdict": "short"}, follow_redirects=False
    )

    assert response.status_code == 401
    assert [row["reference"] for row in escalations.pending()] == [REFERENCE]
    assert boards.corrections() == []


def test_the_login_page_carries_no_queue(anonymous):
    """It is reachable without a session, so nothing on it may say what is
    waiting or which boards exist."""
    body = anonymous.get("/login").text
    assert STEM not in body
    assert "waiting" not in body


def test_signing_in_takes_the_operator_where_they_were_going(anonymous):
    """A bookmarked region survives the sign-in, which is the only reason the
    parameter exists."""
    landing = anonymous.get(f"/c/{STEM}/0", follow_redirects=False)
    assert landing.headers["location"] == f"/login?next=%2Fc%2F{STEM}%2F0"

    response = anonymous.post(
        "/login",
        data={"name": TEST_OPERATOR, "secret": TEST_SECRET, "next": f"/c/{STEM}/0"},
        follow_redirects=False,
    )
    assert response.headers["location"] == f"/c/{STEM}/0"


def test_the_login_form_will_not_bounce_a_visitor_off_this_station(anonymous):
    """An open redirect on a login form is the cheapest phishing primitive
    there is."""
    response = anonymous.post(
        "/login",
        data={"name": TEST_OPERATOR, "secret": TEST_SECRET,
              "next": "//example.invalid/phish"},
        follow_redirects=False,
    )
    assert response.headers["location"] == "/"


def test_a_wrong_passphrase_says_so_without_saying_which_half(anonymous):
    response = anonymous.post(
        "/login", data={"name": TEST_OPERATOR, "secret": "wrong"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert auth.COOKIE_NAME not in anonymous.cookies
    body = anonymous.get(response.headers["location"]).text
    assert "that name and passphrase do not match" in body


def test_signing_out_ends_the_session(client):
    assert client.get("/", follow_redirects=False).status_code == 200
    client.post("/logout", follow_redirects=False)
    assert client.get("/", follow_redirects=False).status_code == 303


def test_the_station_works_with_javascript_off(anonymous):
    """The whole path -- sign in, read the queue, answer a region -- through
    plain form posts. `TestClient` runs no JavaScript, so this is the shop-floor
    browser with scripting locked down, which is the case the station is built
    for."""
    body = anonymous.get("/login").text
    assert '<form class="signin-form" method="post" action="/login">' in body
    assert "<script" not in body

    sign_in(anonymous)
    service.start_review(station_app._graph, REFERENCE)
    response = anonymous.post(
        f"/c/{STEM}/0/verdict", data={"verdict": "short"}, follow_redirects=False
    )
    assert response.status_code == 303
    assert boards.corrections()[0]["reviewer"] == TEST_OPERATOR


# ---- the name is not the form's to choose --------------------------------


def test_the_recorded_reviewer_is_the_session_and_not_the_form(client, graph):  # noqa: F811
    """The station used to take the name off a text input with ``operator``
    prefilled. Posting one now changes nothing: the field is gone from the
    template and the route never reads it."""
    service.start_review(graph, REFERENCE)

    client.post(
        f"/c/{STEM}/0/verdict",
        data={"verdict": "short", "reviewer": "somebody-else", "asked_by": "nobody"},
        follow_redirects=False,
    )

    recorded = boards.corrections()[0]
    assert recorded["reviewer"] == TEST_OPERATOR
    assert recorded["attribution"] == SIGNED_IN


def test_the_verdict_form_offers_no_name_to_fill_in(client, graph):  # noqa: F811
    service.start_review(graph, REFERENCE)
    body = client.get(f"/c/{STEM}/0").text

    assert 'name="reviewer"' not in body
    assert TEST_OPERATOR in body, "the operator should see whose name goes on the label"


def test_the_board_record_names_the_operator_who_settled_it(client, graph):  # noqa: F811
    """The board-level ``decided_by`` was free text for the same reason the
    region-level one was, and it is the field an auditor reads first."""
    from aoi_agent.store import dispositions

    service.start_review(graph, REFERENCE)
    service.start_review(graph, f"{STEM}#1")
    client.post(f"/c/{STEM}/0/verdict", data={"verdict": "short"},
                follow_redirects=False)
    client.post(f"/c/{STEM}/1/verdict", data={"verdict": "false_call"},
                follow_redirects=False)

    assert dispositions.history(STEM)[-1]["decided_by"] == TEST_OPERATOR


def test_a_question_is_attributed_to_whoever_asked_it(client, monkeypatch):
    """``asked_by`` was a form default and a query parameter, which meant the
    name beside a stored question was whatever the caller typed into a URL."""
    from aoi_agent.store import analysis as analysis_store

    monkeypatch.setattr(
        station_app, "analysis_graph",
        lambda: _StubAnalysisGraph(),
    )
    client.post("/ask", data={"question": "what is the defect rate?",
                              "asked_by": "somebody-else"},
                follow_redirects=False)

    assert analysis_store.recent_runs(1)[0]["asked_by"] == TEST_OPERATOR


class _StubAnalysisGraph:
    """Answers without a model, because no test may call one."""

    def invoke(self, state, **kwargs):
        return {
            "question": state["question"],
            "plan": {"interpretation": "stub", "calls": []},
            "results": [],
            "answer": "stub answer",
            "timings_ms": {},
            "refused": False,
        }
