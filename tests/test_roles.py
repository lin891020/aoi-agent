"""Two roles, and the questions an auditor asks about them.

The station had none until 2026-08-25: every operator could answer every
region, so "who is authorised to judge what" had one answer and it was
"everybody". That was tolerable while every region was the same kind of
question. It stopped being tolerable the moment a region could be *handed
back* -- because handing one to the next ordinary operator hands it to the
same judgement that already failed on it, and what comes out is a guess
recorded as a training label, which is the exact failure the deferral path was
built to prevent.

So the model is deliberately two words wide. `senior` may answer a handed-back
region; `operator` may not; nothing else is gated, because on a line every
trained operator answers every ordinary region and a permission grid over
defect classes would encode a policy no work instruction states.

These tests are mostly about the ways a role system fails *open* -- a file that
does not parse, a role that is not real, a rotation that quietly demotes.
"""

from __future__ import annotations

import pytest

from aoi_agent.station import auth

SECRET = "the-suite-signs-in-with-this"


def write(path, *lines: str) -> None:
    path.write_text("".join(line + "\n" for line in lines))


@pytest.fixture
def operators_file(tmp_path, monkeypatch):
    path = tmp_path / "operators"
    monkeypatch.setenv(auth.OPERATORS_ENV, str(path))
    return path


def record(secret: str = SECRET) -> str:
    return auth.hash_secret(secret, iterations=1000)


# ---- the old format has to keep working ----------------------------------


def test_a_file_written_before_roles_still_parses(operators_file):
    """A credential file is not a thing a line rewrites for a feature release.
    If the old format stopped loading, the station would lock out every
    operator on upgrade -- which is worse than having no roles."""
    write(operators_file, f"mike:{record()}")

    assert auth.authenticate("mike", SECRET) is not None
    assert set(auth.load_operators()) == {"mike"}


def test_silence_about_a_role_is_read_as_the_lower_one(operators_file):
    """The direction matters more than the value. Reading silence as `senior`
    would grant every existing operator a permission nobody decided to give
    them, on the very file whose purpose is to make that decidable."""
    write(operators_file, f"mike:{record()}")

    assert auth.role_of("mike") == auth.OPERATOR
    assert auth.seniors() == []


def test_a_stated_role_is_honoured(operators_file):
    write(operators_file, f"mike:{record()}", f"sandy:{record()}:senior")

    assert auth.role_of("sandy") == auth.SENIOR
    assert auth.seniors() == ["sandy"]


def test_the_encoded_secret_survives_the_extra_field(operators_file):
    """The secret separates its own fields with `$`, which is why `:` is free
    to mean something here. If that were ever not true, sign-in would break for
    everyone with a role and the parse would look fine."""
    write(operators_file, f"sandy:{record()}:senior")

    assert auth.authenticate("sandy", SECRET) is not None
    assert auth.authenticate("sandy", "wrong") is None


# ---- the ways it could fail open -----------------------------------------


def test_a_role_that_is_not_real_does_not_become_a_permission(operators_file):
    """A typo must not grant anything."""
    write(operators_file, f"mike:{record()}:supervisor")

    assert auth.role_of("mike") == auth.OPERATOR
    assert auth.seniors() == []


def test_a_role_that_is_not_real_does_not_lock_everyone_out_either(operators_file):
    """The other failure, and the reason the parse does not raise: refusing to
    load the file over one bad word would take the whole station down, and the
    person who made the typo was trying to grant a permission, not remove
    everyone's."""
    write(operators_file, f"mike:{record()}:supervisor")

    assert auth.authenticate("mike", SECRET) is not None


def test_a_role_that_is_not_real_is_reported_rather_than_swallowed(operators_file):
    """Downgraded silently, somebody has written a permission that is not in
    force and nothing anywhere would ever say so."""
    write(operators_file, f"mike:{record()}:supervisor", f"sandy:{record()}:senior")

    assert auth.unknown_roles() == {"mike": "supervisor"}


def test_an_unknown_name_gets_the_lower_role_rather_than_raising(operators_file):
    """A session outliving its operator should lose privileges, not crash the
    page it is on."""
    write(operators_file, f"mike:{record()}")

    assert auth.role_of("ghost") == auth.OPERATOR
    assert auth.role_of(None) == auth.OPERATOR


def test_a_missing_file_grants_nothing(operators_file):
    """A station that opens itself when its credential file is absent is one
    deleted file away from the state this work exists to end."""
    assert auth.load_operators() == {}
    assert auth.seniors() == []
    assert auth.role_of("mike") == auth.OPERATOR


# ---- the role is not in the cookie ---------------------------------------


def test_a_role_is_read_from_the_file_and_not_frozen_at_sign_in(operators_file):
    """The distinction between a revocation and a wish.

    A role baked into the session at sign-in outlives the file that granted it,
    so revoking one would take effect whenever the operator next happened to
    log out. Here the file is the authority and the change is immediate.
    """
    write(operators_file, f"sandy:{record()}:senior")
    identity = auth.authenticate("sandy", SECRET)
    assert auth.role_of(identity.name) == auth.SENIOR

    write(operators_file, f"sandy:{record()}")

    assert auth.role_of(identity.name) == auth.OPERATOR


def test_the_identity_carries_attribution_and_not_authority(operators_file):
    """Two different questions, kept in two different places. `ReviewerIdentity`
    answers "whose label is this", which goes on the record; the role answers
    "may they do this", which does not. Putting the role on the identity would
    have written an authorisation into every training label."""
    write(operators_file, f"sandy:{record()}:senior")

    identity = auth.authenticate("sandy", SECRET)

    assert identity.method == "signed_in"
    assert not hasattr(identity, "role")


# ---- the script that writes the file -------------------------------------


def test_the_script_writes_a_role_it_can_read_back(operators_file, monkeypatch):
    import add_operator

    add_operator.main(["sandy", "--secret", SECRET, "--role", "senior"])

    assert auth.role_of("sandy") == auth.SENIOR


def test_the_script_refuses_a_role_that_is_not_real(operators_file):
    """The supported path cannot produce the typo `load_operator_records` has to
    tolerate. Tolerated on read, refused on write."""
    import add_operator

    with pytest.raises(SystemExit):
        add_operator.main(["sandy", "--secret", SECRET, "--role", "supervisor"])


def test_rotating_a_passphrase_does_not_demote(operators_file):
    """How a station ends up with nobody who can clear the handed-back queue:
    somebody rotates the one senior's passphrase and does not think to restate
    a role they were not changing."""
    import add_operator

    add_operator.main(["sandy", "--secret", SECRET, "--role", "senior"])
    add_operator.main(["sandy", "--secret", "a-new-one"])

    assert auth.role_of("sandy") == auth.SENIOR
    assert auth.authenticate("sandy", "a-new-one") is not None


def test_the_default_role_is_written_out_rather_than_left_implicit(operators_file):
    """An absent third field would then mean two things -- "written before
    roles" on an old line and "deliberately not senior" on a new one -- and
    this project has paid enough for that distinction elsewhere."""
    import add_operator

    add_operator.main(["mike", "--secret", SECRET])

    assert operators_file.read_text().strip().endswith(f":{auth.OPERATOR}")


def test_a_role_can_be_changed_without_resetting_the_passphrase(operators_file):
    """Otherwise granting a permission means changing somebody's credential,
    which means either knowing their passphrase or changing it under them --
    and the supervisor who cannot reach the operator grants it to themselves
    instead. A permission model nobody can use correctly is not one."""
    import add_operator

    add_operator.main(["sandy", "--secret", SECRET])
    add_operator.main(["sandy", "--role", "senior"])

    assert auth.role_of("sandy") == auth.SENIOR
    assert auth.authenticate("sandy", SECRET) is not None, (
        "the passphrase was reset by a role change"
    )


def test_granting_a_role_to_a_name_that_does_not_exist_still_needs_a_secret(
    operators_file, monkeypatch
):
    """The shortcut is a *change*, not a create. Creating an operator with no
    passphrase would be an account anybody can sign in as."""
    import add_operator

    monkeypatch.setattr("getpass.getpass", lambda *a: "")

    assert add_operator.main(["ghost", "--role", "senior"]) == 1
    assert auth.load_operators() == {}
