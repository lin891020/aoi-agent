"""MCP server exposing the acceptance-criteria documents."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from aoi_agent.store import standards

mcp = MCPServer("aoi-standards")


@mcp.tool()
def search_standards(query: str, top_k: int = 3) -> dict:
    """Find the acceptance criteria that apply to a defect or a decision.

    Use this before dispositioning a board, and whenever a class has
    conditional limits (mousebite, spur, spurious copper, pin hole) where
    whether it is acceptable depends on measurements rather than presence.

    Args:
        query: What you need to know, in plain language.
        top_k: How many passages to return.
    """
    try:
        passages = standards.search(query, top_k=top_k)
    except Exception as error:  # collection missing
        return {
            "error": f"{error}. Build the index first: "
            "uv run python -c 'from aoi_agent.store.standards import build_index; build_index()'"
        }

    return {
        "query": query,
        "passages": [
            {
                "document": p.document,
                "heading": p.heading,
                "text": p.text,
                "distance": round(p.distance, 4),
            }
            for p in passages
        ],
    }


if __name__ == "__main__":
    mcp.run()
