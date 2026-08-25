"""Run the trained re-verification model on AOI candidates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from aoi_agent.aoi.simulator import Candidate
from aoi_agent.provenance import checkpoint_digest
from aoi_agent.vision.model import build_model, select_device
from aoi_agent.vision.patches import PATCH_SIZE, build_patch

DEFAULT_CHECKPOINT = Path("models/reverifier.pt")

#: Dismissal threshold from the operating-point sweep on the held-out test
#: split: the highest review reduction available while keeping the
#: re-verification escape rate under 0.5%. See docs/benchmarks.md.
#:
#: The sweep's own answer is 0.9609377384185791, rounded **up**. That direction
#: is the whole of the note: higher dismisses less, so up is the safe side and
#: rounding to the nearest is a coin. It was 0.915 until 2026-08-24 -- the
#: previous model's value rounded to the *nearest*, one notch below its sweep,
#: at which the split escaped 15 defects rather than 14: 0.5005%, over the
#: budget this line is cited to satisfy. The citation had been false since it
#: was written.
#:
#: Moved to 0.961 the same day, when registration was turned on in the
#: detector. That changed the candidate population -- 11.5% fewer, the easy
#: false calls residual misalignment was producing -- so the model was retrained
#: and the threshold re-swept, which is the chain
#: `.claude/skills/retraining-the-reverifier` exists to make unskippable.
#: `test_the_shipped_threshold_meets_the_budget_it_cites` re-derives this from
#: the stored predictions rather than trusting either number.
DEFAULT_DISMISS_THRESHOLD = 0.961

#: Below this the model is not confident enough about *any* class for the
#: verdict to stand on its own, and the case is worth escalating.
LOW_CONFIDENCE = 0.70


@dataclass(frozen=True)
class Verdict:
    """What the model thinks one candidate is."""

    predicted_class: str
    confidence: float
    false_call_probability: float
    probabilities: dict[str, float]

    @property
    def is_dismissed(self) -> bool:
        """True when the model is confident enough to drop this from the queue."""
        return self.false_call_probability >= DEFAULT_DISMISS_THRESHOLD

    @property
    def is_uncertain(self) -> bool:
        """True when neither dismissing nor accepting is well supported."""
        return (
            not self.is_dismissed
            and self.confidence < LOW_CONFIDENCE
        )


class ReVerifier:
    """Loads the checkpoint once and classifies candidates."""

    def __init__(self, checkpoint: Path | str = DEFAULT_CHECKPOINT, device=None):
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"{checkpoint} not found. Train the model first:\n"
                "  uv run python scripts/train.py"
            )
        payload = torch.load(checkpoint, map_location="cpu")
        self.checkpoint = checkpoint
        self.checkpoint_digest = checkpoint_digest(checkpoint)
        """Which weights these are, by the SHA-256 of the file.

        Computed once, here, because ``models/reverifier.pt`` is a slot rather
        than an identity: every training run overwrites it, and a decision
        recorded against the path is a decision recorded against whatever is
        there when someone next looks. Every verdict this object produces
        carries this string onto the quality record."""

        self.label_names: list[str] = payload["label_names"]
        self.device = select_device(device)
        self.model = build_model(len(self.label_names), pretrained=False)
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def classify(
        self, template: np.ndarray, test: np.ndarray, candidate: Candidate
    ) -> Verdict:
        patch = build_patch(template, test, candidate, PATCH_SIZE)
        batch = torch.from_numpy(patch).float().div_(255.0).unsqueeze(0)
        probabilities = (
            torch.softmax(self.model(batch.to(self.device)), dim=1)[0].cpu().numpy()
        )
        return self._verdict(probabilities)

    @torch.no_grad()
    def classify_batch(
        self, template: np.ndarray, test: np.ndarray, candidates: list[Candidate]
    ) -> list[Verdict]:
        if not candidates:
            return []
        patches = np.stack(
            [build_patch(template, test, c, PATCH_SIZE) for c in candidates]
        )
        batch = torch.from_numpy(patches).float().div_(255.0)
        probabilities = (
            torch.softmax(self.model(batch.to(self.device)), dim=1).cpu().numpy()
        )
        return [self._verdict(row) for row in probabilities]

    def _verdict(self, probabilities: np.ndarray) -> Verdict:
        by_name = {
            name: float(value)
            for name, value in zip(self.label_names, probabilities, strict=True)
        }
        best = int(probabilities.argmax())
        return Verdict(
            predicted_class=self.label_names[best],
            confidence=float(probabilities[best]),
            false_call_probability=by_name["false_call"],
            probabilities=by_name,
        )


@lru_cache(maxsize=1)
def get_reverifier(checkpoint: str = str(DEFAULT_CHECKPOINT)) -> ReVerifier:
    """Process-wide singleton so MCP servers load the weights once."""
    return ReVerifier(checkpoint)
