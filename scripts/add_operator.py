"""Add or replace one operator in the station's credential file.

The whole of user management, on purpose. A re-verification station serves a
fixed set of people on a line, and the alternative -- registration, roles,
password reset, an admin page -- is several hundred lines of surface area whose
only customer is a portfolio screenshot. A supervisor runs this once per
operator:

    uv run python scripts/add_operator.py mike
    uv run python scripts/add_operator.py mike --remove
    uv run python scripts/add_operator.py --list

The passphrase is read from a prompt rather than from an argument, because an
argument ends up in the shell history and in the process table. There is a
hidden ``--secret`` for the suite, which needs to drive this without a terminal;
it is undocumented in ``--help`` rather than removed, and using it by hand puts
the passphrase in both of those places. The value is stored as a salted
PBKDF2-HMAC-SHA256 record, so the file is not a list of passphrases, and the
file is written ``0600`` because the protection it has left is the filesystem's.

This adds a name that answers for a label. It does not add a role, because
there are none, and it grants nothing an existing operator does not have.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.station import auth  # noqa: E402

HEADER = (
    "# Operators for the re-verification station: <name>:<pbkdf2 record>.\n"
    "# Written by scripts/add_operator.py. A name here is a name that can go\n"
    "# on a training label -- see src/aoi_agent/station/auth.py for what this\n"
    "# scheme does and does not protect against.\n"
)


def write_operators(path: Path, records: dict[str, auth.Operator]) -> None:
    """Rewrite the file, roles included.

    The role is always written, even when it is the default. A file where the
    ordinary operators carry no third field and the seniors do would make the
    absence mean two things -- "written before roles" on an old line and
    "deliberately not senior" on a new one -- and this project has spent enough
    on the difference between those to not introduce another instance of it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        f"{name}:{record.encoded}:{record.role}\n"
        for name, record in sorted(records.items())
    )
    path.write_text(HEADER + body)
    os.chmod(path, 0o600)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", help="the operator's name, as it will "
                                                "appear on every label they write")
    parser.add_argument("--remove", action="store_true", help="delete this operator")
    parser.add_argument("--list", action="store_true", help="list configured operators")
    parser.add_argument("--role", default=None, choices=auth.ROLES,
                        help="what they may do. `senior` is the only role that "
                             "may answer a region another operator handed back. "
                             "Defaults to `operator` for a new name, and leaves "
                             "an existing name's role alone.")
    parser.add_argument("--secret", default=None,
                        help=argparse.SUPPRESS)  # tests only; see the module docstring
    args = parser.parse_args(argv)

    path = auth.operators_path()
    operators = auth.load_operator_records()

    if args.list:
        print(f"{path}: {len(operators)} operator(s)")
        for name in sorted(operators):
            print(f"  {name:<20} {operators[name].role}")
        # An empty seniority is a real and bad state: regions handed back have
        # nobody who may take them, and the queue grows with no error anywhere.
        if operators and not any(r.role == auth.SENIOR for r in operators.values()):
            print(f"\n  no {auth.SENIOR} is configured. Regions handed back with "
                  f"'I can\'t tell' can then be answered by nobody -- "
                  f"give one operator --role {auth.SENIOR}.")
        stated = auth.unknown_roles()
        for name, role in stated.items():
            print(f"\n  {name!r} states role {role!r}, which is not one of "
                  f"{auth.ROLES} -- it is in force as {auth.DEFAULT_ROLE!r}.")
        return 0

    if not args.name:
        parser.error("a name is required unless --list is given")

    if args.remove:
        if args.name not in operators:
            print(f"{args.name!r} is not in {path}")
            return 1
        del operators[args.name]
        write_operators(path, operators)
        print(f"removed {args.name!r}. Their existing decisions keep their name: "
              "a record of who answered is not undone by the account going away.")
        return 0

    # A role change on its own must not touch the passphrase.
    #
    # It did for the length of one commit, because the only path to writing a
    # record also hashed a secret -- so granting somebody `senior` meant
    # resetting their credential, which means either knowing their passphrase
    # or changing it under them. A permission model whose only grant mechanism
    # is a credential reset is one nobody will use correctly: the supervisor
    # who cannot reach the operator will grant it to themselves instead.
    if args.role and args.secret is None and args.name in operators:
        existing = operators[args.name]
        if existing.role == args.role:
            print(f"{args.name!r} is already {args.role}")
            return 0
        operators[args.name] = auth.Operator(
            name=args.name, encoded=existing.encoded, role=args.role
        )
        write_operators(path, operators)
        print(f"{args.name!r} is now {args.role} in {path} "
              f"(passphrase unchanged)")
        return 0

    secret = args.secret
    if secret is None:
        secret = getpass.getpass(f"passphrase for {args.name!r}: ")
        if secret != getpass.getpass("again: "):
            print("the two entries do not match")
            return 1
    if not secret:
        print("an empty passphrase is not a passphrase")
        return 1

    replacing = args.name in operators
    # An unstated role on a replacement keeps what they had: rotating somebody's
    # passphrase must not quietly demote them, which is how a station ends up
    # with nobody who can clear the handed-back queue.
    role = args.role or (operators[args.name].role if replacing else auth.DEFAULT_ROLE)
    operators[args.name] = auth.Operator(
        name=args.name, encoded=auth.hash_secret(secret), role=role
    )
    write_operators(path, operators)
    print(f"{'replaced' if replacing else 'added'} {args.name!r} in {path} "
          f"as {role}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
