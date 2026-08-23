"""MCP server exposing the acceptance-criteria documents."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from aoi_agent.store import standards

mcp = MCPServer("aoi-standards")


@mcp.tool()
def search_standards(
    query: str, top_k: int = 3, defect_class: str | None = None
) -> dict:
    """Find the acceptance criteria that apply to a defect or a decision.

    Use this before dispositioning a board, and whenever a class has
    conditional limits (mousebite, spur, spurious copper, pin hole) where
    whether it is acceptable depends on measurements rather than presence.

    Args:
        query: What you need to know, in plain language.
        top_k: How many passages to return.
        defect_class: The class the question is about, if it is about one.
            Scopes the search to that class's work instruction plus the
            class-agnostic policy documents, so a limit belonging to another
            class cannot come back as evidence about this one. Leave it unset
            for a question that spans classes -- when to stop the line, what
            the escape budget is -- and set it whenever a class is in hand.
    """
    try:
        passages = standards.search(query, top_k=top_k, defect_class=defect_class)
    except standards.UnknownDefectClass as error:
        return {"error": str(error)}
    except Exception as error:  # collection missing
        return {
            "error": f"{error}. Build the index first: "
            "uv run python -c 'from aoi_agent.store.standards import build_index; build_index()'"
        }

    return {
        "query": query,
        "defect_class": defect_class,
        "passages": [
            {
                "document": p.document,
                "heading": p.heading,
                "text": p.text,
                "distance": round(p.distance, 4),
                "governs": p.defect_class,
            }
            for p in passages
        ],
    }


if __name__ == "__main__":
    mcp.run()
