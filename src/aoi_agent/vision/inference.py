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

#: Dismissal threshold, chosen **out-of-fold on trainval and never on the split
#: it is reported against** -- `scripts/threshold_cv.py`, five folds by image,
#: 6,569 defects behind the choice, taking the lowest threshold whose 95%
#: interval upper bound clears the 0.5% budget rather than its point estimate.
#: A budget is a promise about defects nobody has seen, so it has to be cleared
#: with the interval; on the single validation split one defect is 0.10% and the
#: point-estimate rule picked 0.610, which escapes 0.93% on test.
#:
#: **It was 0.961 until 2026-08-31, swept on `test_predictions.npz` -- the same
#: split every published figure is read from.** As a comparison between engines
#: that is fair, each getting its own oracle; as a deployment number it had seen
#: the answers, and the ≤0.5% compliance the README led with was bought with
#: them. Before that it was 0.915, the previous model's sweep rounded to the
#: nearest rather than up, which escaped 0.5005% against the same budget.
#:
#: What the honest choice costs is stated rather than hidden: at 0.912 the
#: held-out split escapes **0.663%, over QP-110's 0.5%**, while the procedure
#: that chose it predicted 0.320% out-of-fold. The escape rate on unseen boards
#: is about twice the selection estimate at every threshold measured, the class
#: mix of the two populations is the same, and the excess sits in `open` and
#: `short` -- the two classes the work instructions admit no acceptable size
#: for. So this checkpoint, honestly configured, does not meet the budget, and
#: no threshold on this model does without a selection set that predicts unseen
#: boards better than trainval does. See docs/benchmarks.md, 2026-08-31.
DEFAULT_DISMISS_THRESHOLD = 0.912

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
