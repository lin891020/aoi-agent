"""Who is at the station, and how the station knows.

This is deliberately the smallest mechanism that makes an operator's name mean
something, and it is worth saying plainly what it is *for*. It is not here to
keep an attacker out of a network -- a re-verification station sits on a line
behind whatever the plant's network already is. It is here so that a label
written into ``review_decisions`` names somebody, because that label is the
next training round's input, and a store where an expert's judgement and an
anonymous click are the same row shape is a store nobody can retrain from. The
authentication exists to make the attribution real; the attribution is the
point.

**What it is.** A file of operators, one per line, each with a passphrase
stored as a PBKDF2-HMAC-SHA256 record with its own salt and its iteration
count written beside it. Signing in exchanges the passphrase for an HMAC-signed
cookie carrying the operator's name and an expiry; every request reads the
name off that signature. No user management, no registration, no roles, no
reset flow: a line has a fixed set of operators and a supervisor who can run
one script.

**What it does not protect against**, stated here rather than left for someone
to discover:

* **Sharing.** A passphrase two people know produces one name on both their
  labels. Nothing here can tell them apart, and any scheme short of a badge
  reader has this property.
* **A hostile network.** The cookie is a bearer token. Over plain HTTP anyone
  who can read the traffic can replay it, and this process speaks plain HTTP --
  the cookie is marked ``Secure`` only when the request that set it arrived
  over TLS, which on a laptop it does not. Put the station behind TLS if the
  network between it and the shop floor is not trusted. Nothing in this module
  can make that decision for you and nothing in it pretends to have.
* **Anyone with the machine.** The store is a SQLite file. Shell access to the
  host beats every check in this module by opening the database, which is why
  the CLI's ``host_account`` identity is recorded as a weaker word rather than
  pretended to be equal to a signed-in one.
* **Guessing.** There is no rate limit and no lockout. PBKDF2 at the default
  iteration count makes an offline attack on the file expensive and does
  nothing about an online one against a weak passphrase.
* **Repudiation.** A signed-in name is a claim by this process, not a
  signature by the operator. It is good enough to weigh a training label and
  not good enough to hold somebody to in a dispute.

Sessions do not survive a restart unless ``AOI_AGENT_SESSION_SECRET`` is set,
because an unset secret is generated per process. That is the right default for
a laptop and the wrong one for a service, and the login page says so where a
supervisor will see it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from pathlib import Path

from aoi_agent.provenance import ReviewerIdentity

#: Where the operator file lives. One line per operator; ``#`` comments and
#: blank lines are ignored. Written by ``scripts/add_operator.py``.
OPERATORS_ENV = "AOI_AGENT_OPERATORS"
DEFAULT_OPERATORS_PATH = "data/operators"

#: The key the session cookie is signed with. Unset means a per-process random
#: one, which signs every operator out on restart -- fine on a laptop, wrong
#: for a service, and said out loud on the login page rather than only here.
SECRET_ENV = "AOI_AGENT_SESSION_SECRET"

COOKIE_NAME = "aoi_operator"

#: One shift. An operator who walks away from a shared terminal at the end of
#: theirs does not leave their name usable by whoever sits down next morning.
SESSION_MAX_AGE_S = 12 * 60 * 60

#: OWASP's floor for PBKDF2-HMAC-SHA256 at the time of writing. The count is
#: stored in each record rather than assumed, so raising it later does not
#: invalidate the file -- and so the suite can generate cheap records without
#: the production value being a guess about what the suite needs.
DEFAULT_ITERATIONS = 600_000

_SCHEME = "pbkdf2_sha256"
_process_secret: bytes | None = None
_dummy_record: str | None = None


# ---- credentials ---------------------------------------------------------


def hash_secret(secret: str, iterations: int = DEFAULT_ITERATIONS) -> str:
    """Encode a passphrase as ``pbkdf2_sha256$iterations$salt$hash``.

    Salted per record, so two operators who choose the same passphrase do not
    have the same line in the file, and iteration-tagged, so the cost can be
    raised without a migration.
    """
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", secret.encode(), salt, iterations)
    return f"{_SCHEME}${iterations}${salt.hex()}${digest.hex()}"


def verify_secret(secret: str, encoded: str) -> bool:
    """Constant-time check of a passphrase against one stored record.

    A malformed record is ``False`` rather than an exception: a typo in the
    operator file must lock that operator out, not take the station down for
    everyone else.
    """
    try:
        scheme, iterations, salt, expected = encoded.strip().split("$")
        if scheme != _SCHEME:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", secret.encode(), bytes.fromhex(salt), int(iterations)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), expected)


def operators_path() -> Path:
    return Path(os.getenv(OPERATORS_ENV) or DEFAULT_OPERATORS_PATH)


def load_operators() -> dict[str, str]:
    """The operator file, as ``name -> encoded secret``.

    A missing file is an empty mapping, and an empty mapping means nobody can
    sign in and therefore nobody can answer the queue. That is the correct
    failure: a station that opens itself when its credential file is absent
    would be one deleted file away from the state this work exists to end.
    """
    path = operators_path()
    if not path.exists():
        return {}
    operators: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        name, _, encoded = line.partition(":")
        if name.strip():
            operators[name.strip()] = encoded.strip()
    return operators


def authenticate(name: str, secret: str) -> ReviewerIdentity | None:
    """Check a name and passphrase, and return the identity a decision carries.

    An unknown name is still hashed, against a throwaway record built at the
    default cost, so that the two failures do not differ by a visible factor
    and a caller cannot ask this function which names exist. Not a constant-time
    guarantee -- a record written at a lower iteration count than the default
    still answers faster -- and said that way rather than claimed.
    """
    global _dummy_record
    operators = load_operators()
    encoded = operators.get((name or "").strip())
    if encoded is None:
        if _dummy_record is None:
            _dummy_record = hash_secret(secrets.token_hex(16))
        verify_secret(secret or "", _dummy_record)
        return None
    if not verify_secret(secret or "", encoded):
        return None
    return ReviewerIdentity.signed_in(name.strip())


# ---- sessions ------------------------------------------------------------


def session_secret() -> bytes:
    """The signing key, from the environment or generated for this process."""
    global _process_secret
    configured = os.getenv(SECRET_ENV)
    if configured:
        return configured.encode()
    if _process_secret is None:
        _process_secret = secrets.token_bytes(32)
    return _process_secret


def _sign(payload: str) -> str:
    return hmac.new(session_secret(), payload.encode(), hashlib.sha256).hexdigest()


def issue_session(name: str, now: float | None = None) -> str:
    """A cookie value naming one operator until it expires.

    ``name.expiry.signature``, with the name base64'd so a colon or a dot in it
    cannot move the field boundaries. Nothing here is encrypted: the value is
    readable by whoever holds it, and what the signature buys is that it cannot
    be *edited* -- which is the property the store needs, since the name on the
    label comes from this and never from a form.
    """
    expires = int((now if now is not None else time.time()) + SESSION_MAX_AGE_S)
    encoded = base64.urlsafe_b64encode(name.encode()).decode()
    payload = f"{encoded}.{expires}"
    return f"{payload}.{_sign(payload)}"


def operator_from_session(token: str | None, now: float | None = None) -> str | None:
    """The name a valid, unexpired, untampered cookie carries, or ``None``."""
    if not token:
        return None
    try:
        encoded, expires, signature = token.split(".")
        payload = f"{encoded}.{expires}"
        if not hmac.compare_digest(_sign(payload), signature):
            return None
        if int(expires) < (now if now is not None else time.time()):
            return None
        name = base64.urlsafe_b64decode(encoded.encode()).decode()
    except (ValueError, TypeError, UnicodeDecodeError):
        return None
    return name.strip() or None


def identity_from_session(token: str | None) -> ReviewerIdentity | None:
    """What a decision written under this session names."""
    name = operator_from_session(token)
    return ReviewerIdentity.signed_in(name) if name else None


def secure_cookie_for(scheme: str) -> bool:
    """``Secure`` when the request arrived over TLS, and not otherwise.

    Read off the request rather than configured. A ``Secure`` cookie set over
    plain HTTP is never sent back, so a fixed ``True`` would lock out the
    laptop case this station is mostly run in, and a fixed ``False`` would
    quietly strip the flag from a deployment that had earned it.
    """
    return scheme == "https"
