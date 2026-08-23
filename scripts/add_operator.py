"""Add or replace one operator in the station's credential file.

The whole of user management, on purpose. A re-verification station serves a
fixed set of people on a line, and the alternative -- registration, roles,
password reset, an admin page -- is several hundred lines of surface area whose
only customer is a portfolio screenshot. A supervisor runs this once per
operator:

    uv run python scripts/add_operator.py mike
    uv run python scripts/add_operator.py mike --remove
    uv run python scripts/add_operator.py --list

The passphrase is read from a prompt, never from an argument: an argument ends
up in the shell history and in the process table. It is stored as a salted
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


def write_operators(path: Path, operators: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{name}:{record}\n" for name, record in sorted(operators.items()))
    path.write_text(HEADER + body)
    os.chmod(path, 0o600)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", help="the operator's name, as it will "
                                                "appear on every label they write")
    parser.add_argument("--remove", action="store_true", help="delete this operator")
    parser.add_argument("--list", action="store_true", help="list configured operators")
    parser.add_argument("--secret", default=None,
                        help=argparse.SUPPRESS)  # tests only; see the module docstring
    args = parser.parse_args(argv)

    path = auth.operators_path()
    operators = auth.load_operators()

    if args.list:
        print(f"{path}: {len(operators)} operator(s)")
        for name in sorted(operators):
            print(f"  {name}")
        return 0

    if not args.name:
        parser.error("a name is required unless --list is given")

    if args.remove:
        if args.name not in operators:
            print(f"{args.name!r} is not in {path}")
            return 1
        del operators[args.name]
        write_operators(path, operators)
        print(f"removed {args.name!r}; their existing decisions keep their name, "
              "because a record of who answered is not revoked by an account being")
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
    operators[args.name] = auth.hash_secret(secret)
    write_operators(path, operators)
    print(f"{'replaced' if replacing else 'added'} {args.name!r} in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
