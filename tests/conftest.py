import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from aoi_agent.aoi.simulator import Candidate
from aoi_agent.data.deeppcb import Annotation


@pytest.fixture
def blank_board() -> np.ndarray:
    """A 128x128 board with some copper on it."""
    board = np.zeros((128, 128), dtype=np.uint8)
    board[20:100, 30:40] = 255   # a vertical trace
    board[20:30, 30:100] = 255   # a horizontal trace
    return board


def make_candidate(x1, y1, x2, y2, area=None) -> Candidate:
    return Candidate(x1, y1, x2, y2, area if area is not None else (x2 - x1) * (y2 - y1))


def make_annotation(x1, y1, x2, y2, class_id=1) -> Annotation:
    return Annotation(x1, y1, x2, y2, class_id)


# ---- the station's sign-in ----------------------------------------------
#
# Every page on the station is behind a session since 2026-08-23, so a
# `TestClient` that has not signed in gets a redirect to `/login` rather than
# the page under test. These two are how the page tests get past that in one
# line, without any test having to know how the cookie is built and without the
# guard being switchable off for the suite -- a test that can turn the check
# off is a test that stops noticing when the check goes away.
#
# The credential file is written at a thousand PBKDF2 iterations rather than the
# shipped six hundred thousand. The count lives in each record, so this is the
# real code path over a cheap record, not a second path.

from aoi_agent import i18n
from aoi_agent.station import auth  # noqa: E402

TEST_OPERATOR = "mike"
TEST_SECRET = "the-suite-signs-in-with-this"


@pytest.fixture
def operators(tmp_path, monkeypatch) -> str:
    """One operator, in a credential file this test owns."""
    path = tmp_path / "operators"
    path.write_text(f"{TEST_OPERATOR}:{auth.hash_secret(TEST_SECRET, iterations=1000)}\n")
    monkeypatch.setenv(auth.OPERATORS_ENV, str(path))
    # Pinned, so a session issued in one request verifies in the next even
    # though the process-wide fallback would too. An explicit key is also what
    # a deployment is told to set, so the suite exercises that arrangement.
    monkeypatch.setenv(auth.SECRET_ENV, "a-signing-key-for-the-suite")
    return TEST_OPERATOR


@pytest.fixture
def senior(operators, tmp_path) -> str:
    """The same operator, promoted.

    Rewrites the credential file rather than patching `auth`, so the tests that
    depend on it exercise the real parse of the real format -- including the
    third field, which is where a role actually comes from. A fixture that
    monkeypatched `role_of` would pass on a file format that never worked.
    """
    path = tmp_path / "operators"
    path.write_text(
        f"{TEST_OPERATOR}:{auth.hash_secret(TEST_SECRET, iterations=1000)}:senior\n"
    )
    return TEST_OPERATOR


@pytest.fixture
def senior_elsewhere(operators, tmp_path) -> str:
    """A senior exists, and it is not the operator signing in.

    The case the two notices on the deferred page have to tell apart: "nobody
    can answer these" is a configuration fault and "you cannot answer these" is
    a permission. With one operator promoted in place there is no way to
    produce the second, so a test written that way asserts the first while
    claiming the second.
    """
    path = tmp_path / "operators"
    secret = auth.hash_secret(TEST_SECRET, iterations=1000)
    path.write_text(f"{TEST_OPERATOR}:{secret}:operator\nsandy:{secret}:senior\n")
    return "sandy"


def read_in(client, locale: str):
    """Read the station in one language for the rest of this test.

    Tests that assert on wording have to say which language they mean. Before
    the station was bilingual they did not have to, and the ones that assert
    English now say so here -- an English assertion against a station whose
    default is Traditional Chinese is an assertion about an accident.
    """
    client.cookies.set(i18n.LOCALE_COOKIE, locale)
    return client


def sign_in(client, name: str = TEST_OPERATOR, secret: str = TEST_SECRET):
    """Post the login form and keep the cookie. Returns the same client."""
    response = client.post(
        "/login", data={"name": name, "secret": secret}, follow_redirects=False
    )
    assert response.status_code == 303, response.text
    assert auth.COOKIE_NAME in client.cookies, "signing in set no session cookie"
    return client


# ---- where a test sits against the flow's thresholds ---------------------
#
# Five test files used to write `confidence = 0.93` to mean "high enough to be
# dispositioned, low enough that the LLM is still asked for a rationale". That
# was true of 0.93 while the band was [0.915, 0.95), and it stopped being true
# on 2026-08-24 when a retrain moved `ESCALATE_BELOW` to 0.961: eleven tests
# went red naming a number whose meaning had moved out from under them.
#
# Deriving the input from the constants is safe here, and the distinction is
# worth stating because the opposite mistake has already been made twice in
# this suite: what these tests assert is a *route* -- which nodes ran, what got
# written -- and a route is not computed from a threshold. A test that derived
# its expected output from the constant under test would assert nothing, which
# is the degenerate shape `tests/test_registration.py` was caught in.
#
# `tests/test_threshold_citations.py` holds the band open. Without it a
# collapsed band would put both of these on the same value and every test
# below would pass by saying nothing.

from aoi_agent.graph.flow import CONFIDENT, ESCALATE_BELOW  # noqa: E402

#: Dispositioned automatically, and the LLM is still asked to explain it.
IN_THE_EXPLANATION_BAND = (ESCALATE_BELOW + CONFIDENT) / 2

#: Below the line: this one goes to a person, whatever else the test does.
BELOW_ESCALATION = ESCALATE_BELOW - 0.05

#: Above the cost gate: dispositioned, and the LLM is never asked.
ABOVE_CONFIDENT = (CONFIDENT + 1.0) / 2
