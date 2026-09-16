"""Deep prompt tuning trains the prompts and the head, and the backbone not at all.

Five ways the VPT arm could report something other than frozen prompt tuning,
none of which raises:

* a backbone parameter receives a gradient, and the run is partial fine-tuning;
* the prompts are appended at every block instead of replacing the last block's,
  so the sequence grows and block i does not see prompt i;
* the head reads a prompt token or a patch token instead of the class token;
* positions are interpolated after the prompts are in, so
  `interpolate_pos_encoding` counts prompts as patches;
* the input differs from the probe's, and the comparison is two preprocessings.

No weights are downloaded: the backbone is a stub with DINOv2's interface.
"""

from __future__ import annotations

from pathlib import Path

import dinov2_probe as probe
import dinov2_vpt as vpt
import numpy as np
import pytest
import torch
import train
from torch import nn

from aoi_agent.vision.dataset import CandidateDataset
from aoi_agent.vision.patches import PatchSet

ROOT = Path(__file__).resolve().parents[1]
DIM = 8
K = 3


class _Recorder(nn.Module):
    """A block that remembers what it was given and what it returned."""

    def __init__(self):
        super().__init__()
        self.mix = nn.Linear(DIM, DIM)

    def forward(self, x):
        self.seen = x.detach().clone()
        out = x + self.mix(x)
        self.returned = out.detach().clone()
        return out


class _StubBackbone(nn.Module):
    embed_dim = DIM
    patch_size = 14

    def __init__(self, depth: int = 3):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, DIM, kernel_size=14, stride=14)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, 257, DIM))
        self.blocks = nn.ModuleList([_Recorder() for _ in range(depth)])
        self.norm = nn.LayerNorm(DIM)

    def prepare_tokens_with_masks(self, x, masks=None):
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls_token.expand(len(tokens), -1, -1), tokens), dim=1)
        # Positions are added to exactly CLS + patches, as the hub does.
        assert tokens.shape[1] == self.pos_embed.shape[1]
        self.prepared_tokens = tokens.shape[1]
        return tokens + self.pos_embed


def _model(depth: int = 3) -> vpt.VPTDinov2:
    torch.manual_seed(0)
    return vpt.VPTDinov2(_StubBackbone(depth), n_classes=7, prompts=K)


def _batch(n: int = 2) -> torch.Tensor:
    return torch.rand(n, 3, 64, 64)


# ---- frozen means frozen -------------------------------------------------------


def test_the_backbone_receives_no_gradient():
    model = _model()
    model.train()
    model(_batch()).sum().backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.prompts.grad is not None and model.head.weight.grad is not None


def test_only_the_prompts_and_the_head_are_trainable():
    model = _model()
    assert {n for n, p in model.named_parameters() if p.requires_grad} == {
        "prompts", "head.weight", "head.bias"}
    assert vpt.trainable_parameters(model) == 3 * K * DIM + DIM * 7 + 7


def test_training_mode_leaves_the_backbone_in_eval():
    model = _model()
    model.train()
    assert model.training and model.head.training
    assert not model.backbone.training


def test_the_backbone_digest_sees_a_changed_tensor():
    model = _model()
    before = vpt.backbone_digest(model.backbone)
    with torch.no_grad():
        model.backbone.norm.weight[0] += 1.0
    assert vpt.backbone_digest(model.backbone) != before


# ---- where the prompts go ------------------------------------------------------


def test_every_block_sees_its_own_prompts_in_place_of_the_last():
    model = _model(depth=4)
    model.eval()
    model(_batch())
    for depth, block in enumerate(model.backbone.blocks):
        assert block.seen.shape[1] == 1 + K + 256, "prompts must replace, not accumulate"
        expected = model.prompts[depth].detach().unsqueeze(0).expand(2, -1, -1)
        assert torch.allclose(block.seen[:, 1:1 + K], expected)


def test_the_class_token_and_patches_pass_through_between_blocks():
    model = _model()
    model.eval()
    model(_batch())
    blocks = model.backbone.blocks
    for earlier, later in zip(blocks, blocks[1:], strict=False):
        assert torch.allclose(later.seen[:, :1], earlier.returned[:, :1])
        assert torch.allclose(later.seen[:, 1 + K:], earlier.returned[:, 1 + K:])


def test_positions_are_added_before_the_prompts_go_in():
    model = _model()
    model.eval()
    model(_batch())
    assert model.backbone.prepared_tokens == 1 + 256


def test_the_head_reads_the_class_token_after_the_final_norm():
    model = _model()
    model.eval()
    captured = {}
    model.head.register_forward_hook(lambda _m, inputs, _o: captured.setdefault("x", inputs[0]))
    model(_batch())
    last = model.backbone.blocks[-1].returned
    assert torch.allclose(captured["x"], model.backbone.norm(last)[:, 0])


def test_the_input_is_prepared_as_the_probe_prepared_it():
    raw = np.random.default_rng(0).integers(0, 256, size=(2, 3, 64, 64), dtype=np.uint8)
    ours = _model().prepare(torch.from_numpy(raw).float() / 255.0)
    assert torch.allclose(ours, probe.prepare(raw, "stacked"), atol=1e-5)


# ---- train.fit drives it unchanged ---------------------------------------------


class _TinyFrozen(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(3 * 64 * 64, 16).requires_grad_(False)
        self.head = nn.Linear(16, 7)

    def forward(self, x):
        return self.head(self.body(x.flatten(1)))


def test_fit_trains_a_passed_model_and_only_what_requires_a_gradient(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("fit built a ResNet when it was handed a model")

    monkeypatch.setattr(train, "build_model", refuse)
    rng = np.random.default_rng(0)
    n = 48
    patch_set = PatchSet(
        patches=rng.integers(0, 256, size=(n, 3, 64, 64), dtype=np.uint8),
        labels=np.array([0, 1] * (n // 2), dtype=np.int64),
        label_names=["false_call", "open", "short", "mousebite", "spur", "copper", "pin-hole"],
        image_index=np.repeat(np.arange(8), n // 8), boxes=np.zeros((n, 4), dtype=np.int64),
    )
    data = CandidateDataset(patch_set, augment=False)
    model = _TinyFrozen()
    frozen_before = model.body.weight.detach().clone()
    head_before = model.head.weight.detach().clone()

    trained, history, _ = train.fit(
        data, data, patch_set, patch_set.label_names, epochs=1, batch_size=8, lr=1e-2,
        seed=0, device=torch.device("cpu"), escape_budget=0.5, model=model, log=lambda *_: None)

    assert trained is model and len(history) == 1
    assert torch.equal(model.body.weight, frozen_before)
    assert not torch.equal(model.head.weight, head_before)


# ---- the verdict was written before the run ------------------------------------


@pytest.mark.parametrize(("median", "label"), [
    (0.60, "exceeds_resnet"),
    (0.5280, "exceeds_resnet"),
    (0.5279, "matches_resnet"),
    (0.51, "matches_resnet"),
    (0.4973, "matches_resnet"),
    (0.4972, "beats_probe"),
    (0.45, "beats_probe"),
    (0.4204, "no_better_than_probe"),
    (0.30, "no_better_than_probe"),
])
def test_the_verdict_is_the_pre_registered_ladder(median, label):
    assert vpt.verdict(median)[0] == label


def test_the_comparators_are_the_published_figures():
    benchmarks = (ROOT / "docs" / "benchmarks.md").read_text()
    assert f"{vpt.RESNET_ORACLE_MIN:.2%}–{vpt.RESNET_ORACLE_MAX:.2%}" in benchmarks
    assert f"{vpt.PROBE_ORACLE_MIN:.2%}–{vpt.PROBE_ORACLE_MAX:.2%}" in benchmarks
    assert f"{vpt.RESNET_ESCAPE_MAX:.3%}" in benchmarks
    assert vpt.RESNET_ORACLE_MIN == probe.RESNET_ORACLE_MIN
    assert vpt.RESNET_ORACLE_MAX == probe.RESNET_ORACLE_MAX


def test_only_the_pre_registered_configuration_may_append():
    assert vpt.publishable([2, 1, 0], 10, 10, 1e-3, 64)
    assert not vpt.publishable([0], 10, 10, 1e-3, 64)
    assert not vpt.publishable([0, 1, 2], 5, 10, 1e-3, 64)
    assert not vpt.publishable([0, 1, 2], 10, 20, 1e-3, 64)
    assert not vpt.publishable([0, 1, 2], 10, 10, 3e-4, 64)
    assert not vpt.publishable([0, 1, 2], 10, 10, 1e-3, 128)
