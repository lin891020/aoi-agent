"""Verify each MCP server starts and advertises its tools over stdio.

Acceptance criterion: the tools must be usable by any MCP client, not only by
this project's own flow.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp import ClientSession, StdioServerParameters, stdio_client  # noqa: E402

SERVERS = ["classify", "production", "standards", "sql_readonly"]


async def check(module: str) -> tuple[str, list[str]]:
    # Inherit the environment. Passing only PYTHONPATH strips PATH, and the
    # interpreter then cannot find the shared libraries its own extensions need.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", f"aoi_agent.mcp_servers.{module}"],
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            tools = await session.list_tools()
            return info.server_info.name, [t.name for t in tools.tools]


async def main() -> int:
    failures = 0
    for module in SERVERS:
        try:
            name, tools = await asyncio.wait_for(check(module), timeout=60)
            print(f"  OK   {name:<16} {', '.join(tools)}")
        except Exception as error:
            failures += 1
            print(f"  FAIL {module:<16} {type(error).__name__}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
