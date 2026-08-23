"""Build the local store from DeepPCB boards plus simulated production context.

Usage::

    uv run python scripts/seed_store.py --split test --limit 200
    uv run python scripts/seed_store.py --migrate-only

``--migrate-only`` brings an existing store up to the current schema without
touching a row. A store carrying a queue and a season of operator corrections
must not have to be rebuilt to gain a nullable column -- the corrections are the
next training round's labels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.store.models import Base, create_all, make_engine, make_session_factory  # noqa: E402
from aoi_agent.store.seed import seed  # noqa: E402


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
    create_all(args.url)

    if args.migrate_only:
        print("schema is up to date")
        return 0

    with make_session_factory(args.url)() as session:
        counts = seed(session, split=args.split, limit=args.limit)

    print(f"\nseeded {counts['boards']} boards, {counts['candidates']} candidates, "
          f"{counts['decisions']} model decisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
