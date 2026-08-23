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


def sign_in(client, name: str = TEST_OPERATOR, secret: str = TEST_SECRET):
    """Post the login form and keep the cookie. Returns the same client."""
    response = client.post(
        "/login", data={"name": name, "secret": secret}, follow_redirects=False
    )
    assert response.status_code == 303, response.text
    assert auth.COOKIE_NAME in client.cookies, "signing in set no session cookie"
    return client
