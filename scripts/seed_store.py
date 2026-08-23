"""Build the local store from DeepPCB boards plus simulated production context.

Usage::

    uv run python scripts/seed_store.py --split test --limit 200
    uv run python scripts/seed_store.py --migrate-only

``--migrate-only`` brings an existing store up to the current schema. A store
carrying a queue and a season of operator corrections must not have to be
rebuilt to gain a nullable column -- the corrections are the next training
round's labels. It adds the declared columns in place and stamps the rows that
predate them, so a value that was never recorded cannot later be read as a
value that is missing, and it says which columns it added.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.store.models import (  # noqa: E402
    Base,
    _add_missing_columns,
    make_engine,
    make_session_factory,
)
from aoi_agent.store.seed import seed  # noqa: E402


def _add_missing_columns_reporting(url: str | None) -> list[str]:
    """``create_all``, and what it had to change to get there."""
    engine = make_engine(url)
    Base.metadata.create_all(engine)
    return _add_missing_columns(engine)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["trainval", "test"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--url", default=None)
    parser.add_argument("--reset", action="store_true", help="drop existing tables first")
    parser.add_argument("--migrate-only", action="store_true",
                        help="add missing columns to an existing store and stop")
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)
    if args.reset:
        Base.metadata.drop_all(make_engine(args.url))
    added = _add_missing_columns_reporting(args.url)

    if args.migrate_only:
        if added:
            # Said out loud, because a migration that also *writes* to rows --
            # stamping the ones that predate a column, so their absence cannot
            # be read as "written without one" -- is not something to do
            # silently on a store holding operator corrections.
            print("added: " + ", ".join(added))
        print("schema is up to date")
        return 0

    with make_session_factory(args.url)() as session:
        counts = seed(session, split=args.split, limit=args.limit)

    print(f"\nseeded {counts['boards']} boards, {counts['candidates']} candidates, "
          f"{counts['decisions']} model decisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
