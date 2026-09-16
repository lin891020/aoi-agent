"""Visual prompt tuning on DINOv2: does adapting the backbone close the gap the probe left?

`scripts/dinov2_probe.py` put a linear probe on frozen DINOv2 ViT-S/14 features
and it was worse than the ResNet-18 on DeepPCB: median 42.47% review removed at
the budget against 49.00%-55.59%, and at each seed's own oracle 39.89%-42.04%
against 49.73%-52.79%. A linear probe is the weakest way to use those features,
and the literature ranks prompt tuning well ahead of it. This asks whether the
gap is the backbone or the probe.

**Deep VPT.** Before each of the twelve blocks, ``PROMPTS`` learnable tokens sit
after the class token, replacing the previous block's; the class token after the
final norm goes to a linear head. The backbone is frozen -- no parameter of it
receives a gradient -- so what trains is 12 x 10 x 384 prompt values and the
head, about 49 thousand numbers against the ResNet's 11.2 million. Positions are
added before the prompts go in, where the hub code inserts its own register
tokens, because `interpolate_pos_encoding` counts every token after the first as
a patch.

**The recipe is the ResNet's, not a new one.** Training is `train.fit`: the same
by-image split, flips and quarter turns, class weights, cosine schedule, ten
epochs and epoch selection on validation review removed at the budget. Two
numbers differ and are fixed, not searched, because a search is an axis chosen
after looking: AdamW at ``LR`` = 1e-3 (the ResNet's 3e-4 is a fine-tuning rate
for eleven million weights) and batch ``BATCH`` = 64, since a ViT at 224 px
carries far more activation memory through its backward than a ResNet at 64 px.
Input is ``stacked`` -- template, test and difference as one RGB image, upsampled
64 -> 224 bilinear, normalised as the probe normalised it. ``per_channel`` is
three backward passes a candidate and does not fit a night.

**Verdict, written before the run.** Three seeds, each trained once; the verdict
reads the median of each seed's oracle -- the best threshold at the <=0.50%
budget on test -- because that is the column both comparators have at this
cost:

* **exceeds the ResNet-18** -- median above 52.79%, its best seed's oracle;
* **matches the ResNet-18** -- inside 49.73%-52.79%, its five seeds' oracle range;
* **beats the probe, short of the ResNet** -- above 42.04%, the probe's best seed;
* **no better than the probe** -- at or below 42.04%.

A deployable column is reported beside it -- the threshold chosen on each seed's
own validation split by the interval's upper bound, test read once -- and its
escape is set against the ResNet's worst, 0.663%. It decides nothing. Three
seeds against the ResNet's five is stated in the entry, and the oracle column is
not a deployment number.

    uv run python scripts/dinov2_vpt.py --max-steps 20       # time real steps, project the run
    uv run python scripts/dinov2_vpt.py --seeds 0 --dry-run
    uv run python scripts/dinov2_vpt.py                      # three seeds, appends
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dinov2_crops_report as crops  # noqa: E402
import dinov2_probe as probe  # noqa: E402
from train import DEFAULTS, fit, predict, split_by_image  # noqa: E402

from aoi_agent.vision.dataset import CandidateDataset  # noqa: E402
from aoi_agent.vision.patches import PatchSet  # noqa: E402

PROMPTS = 10
LR = 1e-3
BATCH = 64
PUBLISHED_SEEDS = [0, 1, 2]
TIMEBOX_HOURS = 5.0

#: The comparators, as published: the ResNet-18's five-seed oracle range
#: (2026-08-31) and the linear probe's three-seed oracle range (2026-09-15).
#: `tests/test_dinov2_vpt.py` holds them against docs/benchmarks.md.
RESNET_ORACLE_MIN = 0.4973
RESNET_ORACLE_MAX = 0.5279
PROBE_ORACLE_MIN = 0.3989
PROBE_ORACLE_MAX = 0.4204
RESNET_ESCAPE_MAX = 0.00663

DEEPPCB_TEST_CANDIDATES = 7322
DEEPPCB_TEST_DEFECTS = 3018


class VPTDinov2(nn.Module):
    """A frozen DINOv2 with a deep prompt at every block and a linear head on CLS.

    Takes the patch batch `CandidateDataset` yields -- float, (n, 3, 64, 64), in
    [0, 1] -- so `train.fit` and `train.predict` drive it unchanged.
    """

    def __init__(self, backbone: nn.Module, n_classes: int, prompts: int = PROMPTS,
                 input_size: int = probe.INPUT_SIZE, log_every: int = 0, log=print):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        dim = backbone.embed_dim
        depth = len(backbone.blocks)
        # VPT's initialisation: uniform, scaled by the patch embedding's fan.
        bound = math.sqrt(6.0 / float(3 * backbone.patch_size ** 2 + dim))
        self.prompts = nn.Parameter(torch.empty(depth, prompts, dim).uniform_(-bound, bound))
        self.head = nn.Linear(dim, n_classes)
        self.input_size = input_size
        self.register_buffer("mean", torch.tensor(probe.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(probe.IMAGENET_STD).view(1, 3, 1, 1))
        self.log_every = log_every
        self.log = log
        self.steps = 0

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()      # frozen means frozen in behaviour too, not only in weights
        return self

    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        """What `dinov2_probe.prepare(..., "stacked")` does, on a float batch."""
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear",
                          align_corners=False, antialias=False)
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.prepare_tokens_with_masks(self.prepare(x))
        k = self.prompts.shape[1]
        for depth, block in enumerate(self.backbone.blocks):
            prompt = self.prompts[depth].unsqueeze(0).expand(tokens.shape[0], -1, -1)
            rest = tokens[:, 1:] if depth == 0 else tokens[:, 1 + k:]
            tokens = block(torch.cat((tokens[:, :1], prompt, rest), dim=1))
        logits = self.head(self.backbone.norm(tokens)[:, 0])
        if self.training and self.log_every:
            self.steps += 1
            if self.steps % self.log_every == 0:
                self.log(f"    step {self.steps}")
        return logits


def backbone_digest(backbone: nn.Module) -> str:
    """SHA-256 over every backbone tensor, so "frozen" is checked after the run, not assumed."""
    digest = hashlib.sha256()
    for name, tensor in sorted(backbone.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verdict(median_oracle: float) -> tuple[str, str]:
    """The pre-registered reading of the three seeds' median oracle review removed."""
    if median_oracle > RESNET_ORACLE_MAX:
        return "exceeds_resnet", (
            f"Exceeds the ResNet-18: median oracle review removed {median_oracle:.2%} is above "
            f"its best seed's {RESNET_ORACLE_MAX:.2%}.")
    if median_oracle >= RESNET_ORACLE_MIN:
        return "matches_resnet", (
            f"Matches the ResNet-18: median oracle review removed {median_oracle:.2%} is inside "
            f"its five seeds' {RESNET_ORACLE_MIN:.2%}-{RESNET_ORACLE_MAX:.2%}.")
    if median_oracle > PROBE_ORACLE_MAX:
        return "beats_probe", (
            f"Beats the linear probe, short of the ResNet-18: median oracle review removed "
            f"{median_oracle:.2%} is above the probe's best seed ({PROBE_ORACLE_MAX:.2%}) and "
            f"below the ResNet's worst ({RESNET_ORACLE_MIN:.2%}).")
    return "no_better_than_probe", (
        f"No better than the linear probe: median oracle review removed {median_oracle:.2%} "
        f"is at or below the probe's best seed, {PROBE_ORACLE_MAX:.2%}.")


def publishable(seeds: list[int], epochs: int, prompts: int, lr: float, batch: int) -> bool:
    """Only the pre-registered configuration may append; anything else is a read."""
    return (sorted(seeds) == PUBLISHED_SEEDS and epochs == DEFAULTS["epochs"]
            and prompts == PROMPTS and lr == LR and batch == BATCH)


def _datasets(trainval: PatchSet, seed: int):
    from torch.utils.data import Subset

    train_idx, val_idx = split_by_image(trainval, DEFAULTS["val_fraction"], seed)
    return (Subset(CandidateDataset(trainval, augment=True), train_idx),
            Subset(CandidateDataset(trainval, augment=False), val_idx))


def speed_test(trainval: PatchSet, args, device, log=print) -> dict:
    """Time real training steps and project the whole run, before committing a night to it."""
    from torch.utils.data import DataLoader

    train_data, val_data = _datasets(trainval, 0)
    model = VPTDinov2(probe.load_backbone(device), len(trainval.label_names), args.prompts).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    loader = iter(DataLoader(train_data, batch_size=args.batch_size, shuffle=True))
    model.train()
    timings = []
    for step in range(args.max_steps + 2):
        batch, target = next(loader)
        began = time.perf_counter()
        optimizer.zero_grad()
        loss = criterion(model(batch.to(device)), target.to(device))
        loss.backward()
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        if step >= 2:                       # the first two pay for kernels and allocation
            timings.append(time.perf_counter() - began)
        log(f"  step {step}: {time.perf_counter() - began:.2f}s")
    step_s = statistics.median(timings)
    steps_per_epoch = math.ceil(len(train_data) / args.batch_size)
    # A forward costs about a third of a training step; validation runs every epoch.
    val_s = math.ceil(len(val_data) / args.batch_size) * step_s / 3
    per_seed_h = args.epochs * (steps_per_epoch * step_s + val_s) / 3600
    projected_h = per_seed_h * len(args.seeds)
    return {"device": str(device), "batch": args.batch_size, "median_step_s": round(step_s, 3),
            "steps_per_epoch": steps_per_epoch, "per_seed_hours": round(per_seed_h, 2),
            "projected_hours": round(projected_h, 2), "timebox_hours": TIMEBOX_HOURS,
            "fits": projected_h <= TIMEBOX_HOURS}


def one_seed(trainval: PatchSet, test: PatchSet, seed: int, args, device, log=print) -> dict:
    from torch.utils.data import DataLoader

    names = list(trainval.label_names)
    false_call_index = names.index("false_call")
    train_data, val_data = _datasets(trainval, seed)
    backbone = probe.load_backbone(device)
    before = backbone_digest(backbone)
    model = VPTDinov2(backbone, len(names), args.prompts, log_every=50, log=log)
    log(f"\n=== seed {seed}: {trainable_parameters(model):,} trainable parameters ===")
    began = time.perf_counter()
    model, history, _best = fit(
        train_data, val_data, trainval, names, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=seed, device=device, escape_budget=args.budget, model=model, log=log)
    minutes = (time.perf_counter() - began) / 60

    val_p, val_y = predict(model, DataLoader(val_data, batch_size=args.batch_size), device)
    test_p, test_y = predict(model, DataLoader(CandidateDataset(test, augment=False),
                                               batch_size=args.batch_size), device)
    row = {
        "seed": seed, "minutes": round(minutes, 1),
        "trainable_parameters": trainable_parameters(model),
        "backbone_unchanged": backbone_digest(model.backbone) == before,
        "best_epoch": max(history, key=lambda h: h["val_review_reduction"])["epoch"],
        "history": history,
        **crops.read_columns(val_p[:, false_call_index], val_y, test_p[:, false_call_index],
                             test_y, false_call_index, args.budget),
    }
    log(f"  seed {seed}: oracle {crops._percent(row['oracle_review_reduction'])}  "
        f"deployable {crops._percent(row['test_review_reduction'])}  ({minutes:.0f} min)")
    return row


def _pct(value) -> str:
    return "—" if value is None else f"{value:.2%}"


def render(record: dict) -> str:
    rows = record["seeds"]
    oracle = [r["oracle_review_reduction"] or 0.0 for r in rows]
    lines = [
        f"## {record['date']} · commit {record['commit']}",
        "",
        "### Stronger pretrained features, adapted: deep visual prompt tuning on DINOv2 ViT-S/14",
        "",
        f"The question the linear probe left open: is DINOv2 worse than the ResNet-18 on DeepPCB "
        f"because of the backbone, or because a linear probe is the weakest way to use it? "
        f"`dinov2_vits14` at `{record['hub_ref'][:12]}`, frozen (every backbone tensor hashed "
        f"before and after each seed), {PROMPTS} prompt tokens before each of its 12 blocks and "
        f"a linear head on the class token -- {rows[0]['trainable_parameters']:,} trainable "
        f"parameters against the ResNet-18's 11.2 million. Trained by `train.fit` with the "
        f"ResNet's split, augmentation, class weights, schedule and epoch rule; AdamW at "
        f"{LR:g} and batch {BATCH}, both fixed before the run. Stacked input, 64 -> 224 "
        f"bilinear. Same {DEEPPCB_TEST_CANDIDATES:,} test candidates, {DEEPPCB_TEST_DEFECTS:,} "
        f"defects. Seeds {', '.join(str(r['seed']) for r in rows)}, "
        f"{record['wall_minutes']:.0f} min on `{record['device']}`. `scripts/dinov2_vpt.py`.",
        "",
        f"**Verdict, by the rule written before the run: {record['verdict_sentence']}**",
        "",
        "| seed | best epoch | oracle review removed | oracle escape | deployable review removed "
        "| deployable escape | backbone unchanged | minutes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        flag = " (above the ResNet's worst)" if (r["test_escape_rate"] or 0) > RESNET_ESCAPE_MAX else ""
        lines.append(
            f"| {r['seed']} | {r['best_epoch']} | **{_pct(r['oracle_review_reduction'])}** | "
            f"{_pct(r['oracle_escape_rate'])} | {_pct(r['test_review_reduction'])} | "
            f"{_pct(r['test_escape_rate'])}{flag} | {'yes' if r['backbone_unchanged'] else '**no**'} | "
            f"{r['minutes']:.0f} |")
    lines += [
        f"| **median** | — | **{statistics.median(oracle):.2%}** | — | — | — | — | — |",
        f"| ResNet-18, five seeds (2026-08-31) | — | {RESNET_ORACLE_MIN:.2%}–{RESNET_ORACLE_MAX:.2%} "
        f"| — | — | — | — | — |",
        f"| DINOv2 linear probe, three seeds (2026-09-15) | — | "
        f"{PROBE_ORACLE_MIN:.2%}–{PROBE_ORACLE_MAX:.2%} | — | — | — | — | — |",
        "",
        "The oracle column is each seed's best threshold at the ≤0.50% budget read on test: the "
        "column both comparators have at this cost, and not a deployment number. The deployable "
        "column is chosen on the seed's own validation split by the interval's upper bound and "
        "decides nothing. Three seeds here against the ResNet's five.",
        "",
        "**What this does not establish.** Deep prompts only, at one prompt length, one learning "
        "rate and one batch size, none searched; shallow prompts, LoRA and full fine-tuning are "
        "untested. Stacked input only -- the probe's better configuration on this data was "
        "per-channel, which is three backward passes a candidate. The smallest backbone. And "
        "DeepPCB only.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patches", type=Path, default=ROOT / "data" / "patches")
    parser.add_argument("--seeds", type=int, nargs="+", default=PUBLISHED_SEEDS)
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--prompts", type=int, default=PROMPTS)
    parser.add_argument("--budget", type=float, default=DEFAULTS["escape_budget"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="time this many training steps, project the run, and stop")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "dinov2_vpt.json")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "docs" / "benchmarks.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from aoi_agent.vision.model import select_device

    device = select_device(args.device)
    trainval = PatchSet.load(args.patches / "trainval.npz")

    if args.max_steps:
        projection = speed_test(trainval, args, device)
        print(json.dumps(projection))
        return 0 if projection["fits"] else 3

    if not args.dry_run and not publishable(args.seeds, args.epochs, args.prompts, args.lr,
                                            args.batch_size):
        print("only the pre-registered configuration (seeds 0 1 2, ten epochs, 10 prompts, "
              "lr 1e-3, batch 64) may append; pass --dry-run", file=sys.stderr)
        return 2

    test = PatchSet.load(args.patches / "test.npz")
    false_calls = int((test.labels == list(test.label_names).index("false_call")).sum())
    if (len(test.labels), len(test.labels) - false_calls) != (DEEPPCB_TEST_CANDIDATES,
                                                              DEEPPCB_TEST_DEFECTS):
        raise SystemExit("the test split is not the one the comparators were read on")

    started = time.perf_counter()
    rows = [one_seed(trainval, test, seed, args, device) for seed in args.seeds]
    if not all(r["backbone_unchanged"] for r in rows):
        print("a backbone tensor changed during training; this is not frozen VPT", file=sys.stderr)
        return 4

    median_oracle = statistics.median(r["oracle_review_reduction"] or 0.0 for r in rows)
    label, sentence = verdict(median_oracle)
    record = {
        "date": datetime.now(UTC).date().isoformat(), "commit": probe.commit(),
        "device": str(device), "hub_ref": probe.HUB_REF,
        "weights_sha256": probe.sha256(probe.weights_path()),
        "prompts": args.prompts, "lr": args.lr, "batch": args.batch_size, "epochs": args.epochs,
        "wall_minutes": (time.perf_counter() - started) / 60,
        "median_oracle": median_oracle, "verdict": label, "verdict_sentence": sentence,
        "seeds": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, default=float) + "\n")

    body = render(record)
    print("\n" + body)
    if args.dry_run:
        print("(dry run -- nothing appended)")
        return 0
    existing = args.benchmarks.read_text()
    args.benchmarks.write_text(existing.rstrip() + "\n\n" + body.rstrip() + "\n")
    print(f"appended to {args.benchmarks.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
