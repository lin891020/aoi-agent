"""MCP server exposing the re-verification model.

Runs the model in-process rather than proxying to an HTTP backend. The point of
putting the tools behind MCP is to measure what that layer costs; adding a
network hop underneath would hide it.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from aoi_agent.aoi.simulator import Candidate
from aoi_agent.store.boards import load_board_images, resolve_candidate
from aoi_agent.vision.inference import get_reverifier

mcp = MCPServer("aoi-classify")


@mcp.tool()
def classify_defect(candidate_ref: str) -> dict:
    """Re-inspect one AOI-flagged region and say what it actually is.

    Use this first for any question about a specific flagged region. It returns
    the defect class the model believes is present, how confident it is, and
    whether the region is confident enough to drop from the review queue.

    Args:
        candidate_ref: Region identifier in the form ``<board>#<index>``,
            for example ``12000001#3``.
    """
    record = resolve_candidate(candidate_ref)
    if record is None:
        return {"error": f"no candidate {candidate_ref!r}; expected '<board>#<index>'"}

    template, test = load_board_images(record["board_stem"])
    candidate = Candidate(
        x1=record["x1"], y1=record["y1"], x2=record["x2"], y2=record["y2"],
        area=record["area"],
    )
    reverifier = get_reverifier()
    verdict = reverifier.classify(template, test, candidate)

    return {
        "candidate_ref": candidate_ref,
        "board": record["board_stem"],
        # Which weights said so. It travels with the reading rather than being
        # looked up when the decision is written, so the digest on the record is
        # the one that produced the number beside it even if the checkpoint is
        # replaced between the two.
        "model_digest": reverifier.checkpoint_digest,
        "box": [record["x1"], record["y1"], record["x2"], record["y2"]],
        "predicted_class": verdict.predicted_class,
        "confidence": round(verdict.confidence, 4),
        "false_call_probability": round(verdict.false_call_probability, 4),
        "recommendation": (
            "dismiss" if verdict.is_dismissed
            else "escalate" if verdict.is_uncertain
            else "review"
        ),
        "probabilities": {k: round(v, 4) for k, v in verdict.probabilities.items()},
    }


@mcp.tool()
def list_candidates(board: str) -> dict:
    """List every region the AOI flagged on one board.

    Args:
        board: Board identifier, for example ``12000001``.
    """
    from aoi_agent.store.boards import candidates_for_board

    records = candidates_for_board(board)
    if not records:
        return {"error": f"no board {board!r} in the store"}
    return {
        "board": board,
        "candidate_count": len(records),
        "candidates": [
            {
                "candidate_ref": r["reference"],
                "box": [r["x1"], r["y1"], r["x2"], r["y2"]],
                "predicted_class": r["predicted_class"],
                "confidence": round(r["confidence"], 4),
            }
            for r in records
        ],
    }


if __name__ == "__main__":
    mcp.run()
