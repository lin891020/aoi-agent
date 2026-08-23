"""What produced a decision -- which weights, which thresholds, which code --
and, for the half of them a person made, who made it and how that name was
established.

A quality record that cannot name the model behind a disposition cannot be
revisited. When a metric moves next month, "which model produced these
dispositions, and under what thresholds" is the first question, and until
2026-08-23 this store could not answer it for any of its 9,140 rows.

The same sentence is true one column over. A human decision is the next
training round's label, and a label that cannot name who made it is a label
nobody can weigh: an expert's judgement and a click looked identical in this
store, which is not a hypothetical -- five regions were clicked through without
the domain knowledge to judge them, four of the five were wrong, and the only
way anyone could act on that was to delete all five by hand. So
``ReviewerIdentity`` sits beside ``DecisionProvenance`` here rather than in a
module of its own: they are one question asked of the two kinds of decider, and
they share the vocabulary of absences below.

Three things identify an automated decision, and each one is *derived* rather
than declared:

* **The weights**, by the SHA-256 of the checkpoint file. Not its path --
  ``models/reverifier.pt`` is overwritten by every training run, so a filename
  identifies a slot and not a set of weights. Not a ``model_version`` string
  either: a field somebody has to remember to bump is worse than no field,
  because the day it is forgotten it is wrong and looks right. A digest cannot
  be forgotten, because nobody writes it.
* **The thresholds in force**, as a small JSON object, because the same weights
  disposition differently at a different operating point and the operating
  point is the half of the decision that moves most often.
* **The code**, by the commit the process is running, with ``+dirty`` when the
  tree has uncommitted changes. Also derived: read from git at first use rather
  than kept in a constant.

Three absences are told apart, and none of them is ``NULL``:

``unrecorded``   the row predates these columns. Backfilled by the migration in
                 ``store.models``, so a decision whose provenance was never
                 captured cannot be mistaken for one whose provenance is
                 genuinely nothing.
``unavailable``  the row was written after the columns existed and the model
                 identity still could not be determined -- an operator's answer
                 resumed from a run checkpointed before digests were carried in
                 the graph state. Their verdict is worth more than the field, so
                 it is recorded and the gap is named.
``unknown``      the code version could not be read: no git, no build stamp. It
                 says so rather than reporting a stale value.

This module imports nothing from the rest of the package on purpose. The store,
the vision layer and the graph all need it, and a provenance record that can
introduce an import cycle will eventually be dropped from one of them.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: Written by the migration onto rows that existed before the columns did.
UNRECORDED = "unrecorded"

#: Written when a value genuinely could not be determined at write time.
UNAVAILABLE = "unavailable"

#: Written when the code version cannot be read from the environment.
UNKNOWN = "unknown"

#: Values that are a statement about the absence of provenance rather than
#: provenance. An automated decision carrying one of these is not attributable
#: and ``store.boards.record_decision`` refuses to write it.
ABSENCES = (UNRECORDED, UNAVAILABLE)

#: How a reviewer's name was established. Stored in ``reviewer_auth`` beside
#: the name itself, because the name alone says nothing about what stands
#: behind it -- ``mike`` typed into a text box and ``mike`` taken off a signed
#: session cookie are the same eight characters and are not the same claim.
#:
#: ``signed_in``     the station authenticated the operator against its
#:                   credential file and read the name off the session, not off
#:                   the form. The only method a browser can produce.
#: ``host_account``  a decision written by the CLI, attributed to the operating
#:                   system account running it. Derived rather than typed, and
#:                   no weaker than the store's own file permissions: anyone who
#:                   can run the CLI can already write the database directly.
#: ``automated``     no person was involved. Not an absence -- it is the
#:                   positive statement that this row has no reviewer by
#:                   construction, and the provenance columns carry the answer
#:                   instead. This is what distinguishes a model decision
#:                   written today from the 9,140 rows that predate the column
#:                   and read ``unrecorded``.
SIGNED_IN = "signed_in"
HOST_ACCOUNT = "host_account"
AUTOMATED = "automated"

#: The two methods that name a person. A ``human`` decision carrying anything
#: else is not attributable and ``store.boards.record_decision`` refuses it.
ATTRIBUTED_METHODS = (SIGNED_IN, HOST_ACCOUNT)

#: How much of the SHA-256 is kept. Sixteen hex characters is 64 bits: enough
#: that two checkpoints colliding is not a thing that happens, short enough to
#: read out loud on a call with an auditor.
DIGEST_CHARS = 16

_ALGORITHM = "sha256"

_digest_cache: dict[tuple[str, int, int], str] = {}
_code_version: str | None = None


def checkpoint_digest(path: Path | str) -> str:
    """Identify a set of weights by its bytes.

    Cached on ``(path, size, mtime)`` so a retrain in the same process is seen.
    A 43MB checkpoint hashes in about 40ms and this runs once per process, at
    model load, off the per-candidate path.
    """
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _digest_cache:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        _digest_cache[key] = f"{_ALGORITHM}:{digest.hexdigest()[:DIGEST_CHARS]}"
    return _digest_cache[key]


def code_version() -> str:
    """The commit this process is running, ``+dirty`` if the tree has changes.

    ``AOI_AGENT_CODE_VERSION`` wins where it is set, which is how a container
    that ships no ``.git`` states its build. Where neither is available the
    answer is ``unknown``: a wrong version on a quality record is worse than an
    absent one, and a hand-maintained constant is the wrong-version machine.
    """
    global _code_version
    if _code_version is not None:
        return _code_version

    stamped = os.getenv("AOI_AGENT_CODE_VERSION")
    if stamped:
        _code_version = stamped.strip()[:64]
        return _code_version

    _code_version = _from_git() or UNKNOWN
    return _code_version


def _from_git() -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not commit:
        return None
    return f"{commit}+dirty" if dirty else commit


@dataclass(frozen=True)
class DecisionProvenance:
    """What one decision was produced by.

    Carried through the graph state as a plain dict so the checkpointer can
    serialise it, and written to three columns so an auditor can group by any
    of them without parsing prose.
    """

    model_digest: str
    thresholds: dict[str, float]
    code_version: str

    @property
    def is_attributable(self) -> bool:
        """Does this name a specific set of weights?"""
        return bool(self.model_digest) and self.model_digest not in ABSENCES

    def thresholds_json(self) -> str:
        return json.dumps(self.thresholds, sort_keys=True, separators=(",", ":"))

    def as_dict(self) -> dict:
        return {
            "model_digest": self.model_digest,
            "thresholds": dict(self.thresholds),
            "code_version": self.code_version,
        }

    def columns(self) -> dict:
        """The three column values, ready to hand to a row."""
        return {
            "model_digest": self.model_digest,
            "thresholds_json": self.thresholds_json(),
            "code_version": self.code_version,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "DecisionProvenance | None":
        """Rebuild one from graph state, or ``None`` if the state carries none.

        A run checkpointed before this existed resumes with no provenance in
        its state. That is the ``unavailable`` case, and the caller decides
        what to do about it -- this returns ``None`` rather than inventing a
        record of something that was not recorded.
        """
        if not payload or not payload.get("model_digest"):
            return None
        return cls(
            model_digest=str(payload["model_digest"]),
            thresholds=dict(payload.get("thresholds") or {}),
            code_version=str(payload.get("code_version") or UNKNOWN),
        )


@dataclass(frozen=True)
class ReviewerIdentity:
    """Who made a decision, and what stands behind that name.

    The name is half a record. ``reviewer`` was free text from the first commit
    of this project until 2026-08-23, which meant the corrections feeding the
    next training round carried a string an operator typed and nothing else --
    an expert's label and an anonymous click were the same row shape. The
    method is the other half, and it is what makes the pair worth keeping: a
    retraining export can select on it, which it could not do on a name.

    Constructed through the three classmethods rather than by hand, so the
    method can never be asserted by whoever is writing the row. ``signed_in``
    is issued by ``station.auth`` after it has verified a credential;
    ``host_account`` reads the OS; ``automated`` names no one on purpose.
    """

    name: str | None
    method: str

    @classmethod
    def signed_in(cls, name: str) -> "ReviewerIdentity":
        """An operator the station authenticated. See ``station.auth``."""
        return cls(name=(name or "").strip()[:64] or None, method=SIGNED_IN)

    @classmethod
    def host_account(cls, name: str | None = None) -> "ReviewerIdentity":
        """Whoever is running the CLI, according to the operating system.

        Derived, not typed, for the same reason the model digest is: a value
        somebody supplies is a value somebody can supply wrongly. It is a
        weaker claim than ``signed_in`` and says so by being a different word --
        it names an account on one machine, not a person who proved anything.
        That is honest rather than generous: the CLI runs beside the SQLite
        file, and anyone who can run it can write the table with ``sqlite3``.
        """
        if name is None:
            try:
                name = getpass.getuser()
            except Exception:  # noqa: BLE001 -- no passwd entry, no environment
                name = ""
        return cls(name=(name or "").strip()[:64] or None, method=HOST_ACCOUNT)

    @classmethod
    def automated(cls) -> "ReviewerIdentity":
        """No person. The provenance columns are this row's answer."""
        return cls(name=None, method=AUTOMATED)

    @property
    def is_attributable(self) -> bool:
        """Does this name a person some mechanism stands behind?

        Stripped rather than taken as given: the classmethods normalise, but a
        record built directly is still a record this has to answer for, and a
        name of three spaces is not a name.
        """
        return bool((self.name or "").strip()) and self.method in ATTRIBUTED_METHODS

    def columns(self) -> dict:
        """The two column values, ready to hand to a row."""
        return {
            "reviewer": (self.name or "").strip()[:64] or None,
            "reviewer_auth": self.method,
        }
