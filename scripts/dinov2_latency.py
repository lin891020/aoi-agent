"""What a DINOv2 probe costs per candidate, beside the ResNet-18 in the same session.

`scripts/dinov2_probe.py` answers whether the stronger features remove more of
the review queue. This answers what they cost to run, which is the other half
of whether anyone would deploy them. The re-verifier's published figure is
2.5 ms per candidate on CPU; that was a different day, so the ResNet-18 is
measured again here, same machine, same session, and the old figure is printed
beside it rather than compared against.

The timed path is `ReVerifier.classify_batch`'s, extended by what DINOv2 needs:
uint8 patches to float, upsample to 224, normalise, forward, pool, linear head,
softmax, back to the host. The linear head is an untrained layer of the right
shape -- its cost is a matrix product of a few thousand numbers against a
forward of 22M parameters, and its weights do not change it.

`per_channel` is three forwards per candidate (template, test and difference
each), `stacked` is one. Both are reported, because the probe entry may choose
either.

Each engine runs in its own process, so peak resident memory belongs to that
engine and not to whichever ran before it. Contention is checked before and
after, the way `reverifier_latency.py` checks it -- `ollama ps` and a process
sweep -- and a contended run prints its table and refuses to append.

    uv run python scripts/dinov2_latency.py --dry-run
    uv run python scripts/dinov2_latency.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dinov2_probe as probe  # noqa: E402
from reverifier_latency import (  # noqa: E402
    WARMUP_ITERATIONS,
    competing_processes,
    ollama_ps,
    peak_rss_mb,
    process_table,
    resident_models,
    summarise,
    synchronise,
)

ENGINES = ("resnet18", "dinov2_per_channel", "dinov2_stacked")
DEVICES = ("cpu", "mps")
CHECKPOINT = ROOT / "models" / "reverifier.pt"
PUBLISHED_RESNET_CPU_MS = 2.5


def _pooling() -> str:
    """The pooling the probe run chose on seed 0, or the widest one if none has run."""
    record = ROOT / "models" / "dinov2_probe.json"
    if record.exists():
        seeds = json.loads(record.read_text()).get("seeds") or []
        if seeds and seeds[0].get("chosen"):
            return seeds[0]["chosen"].split("/")[1]
    return "cls_mean"


def _engine(engine: str, device_name: str):
    """(step, weights MB, load ms): `step(patches)` is one timed classification."""
    import torch

    device = torch.device(device_name)
    began = time.perf_counter()
    if engine == "resnet18":
        from reverifier_latency import classify_once

        from aoi_agent.vision.inference import ReVerifier

        verifier = ReVerifier(CHECKPOINT, device=device_name)
        load_ms = (time.perf_counter() - began) * 1000

        def step(patches):
            return classify_once(verifier.model, verifier.device, patches)

        return step, CHECKPOINT.stat().st_size / 1e6, load_ms

    mode = engine.removeprefix("dinov2_")
    pooling = _pooling()
    backbone = probe.load_backbone(device)
    per_patch = 3 if mode == "per_channel" else 1
    width = backbone.embed_dim * (2 if pooling == "cls_mean" else 1) * per_patch
    head = torch.nn.Linear(width, 7).to(device).eval()
    load_ms = (time.perf_counter() - began) * 1000

    def step(patches):
        with torch.no_grad():
            x = probe.prepare(patches, mode).to(device)
            features = backbone.forward_features(x)
            pooled = probe.pool_torch(features["x_norm_clstoken"],
                                      features["x_norm_patchtokens"], pooling)
            pooled = pooled.reshape(len(patches), -1)
            return torch.softmax(head(pooled), dim=1).cpu().numpy()

    return step, probe.weights_path().stat().st_size / 1e6, load_ms


def _timed(step, device, patches: np.ndarray, batch: int, repeats: int) -> list[float]:
    samples = []
    total = len(patches)
    for index in range(repeats):
        start = (index * batch) % max(1, total - batch)
        chunk = np.ascontiguousarray(patches[start:start + batch])
        synchronise(device)
        began = time.perf_counter()
        step(chunk)
        synchronise(device)
        samples.append((time.perf_counter() - began) * 1000)
    return samples


def measure_one(engine: str, device_name: str, iterations: int, batch_repeats: int) -> dict:
    """One engine on one device, in this process. Printed as JSON by `--one`."""
    import torch

    from aoi_agent.vision.patches import PatchSet

    patches = PatchSet.load(ROOT / "data" / "patches" / "test.npz").patches[:512]
    device = torch.device(device_name)
    step, weights_mb, load_ms = _engine(engine, device_name)

    cold = _timed(step, device, patches, 1, 1)[0]
    _timed(step, device, patches, 1, WARMUP_ITERATIONS)
    single = summarise(_timed(step, device, patches, 1, iterations))
    batch8 = summarise(_timed(step, device, patches, 8, batch_repeats))
    return {
        "engine": engine, "device": device_name, "pooling": None if engine == "resnet18" else _pooling(),
        "load_ms": load_ms, "cold_ms": cold, "single": single,
        "batch8_per_candidate_ms": batch8["p50"] / 8,
        "weights_mb": weights_mb, "peak_rss_mb": peak_rss_mb(),
    }


def contention() -> dict:
    return {"ollama": resident_models(ollama_ps()),
            "processes": competing_processes(process_table(), os.getpid())}


def render(results: list[dict], before: dict, after: dict, commit: str) -> str:
    resnet_cpu = next(r for r in results if r["engine"] == "resnet18" and r["device"] == "cpu")
    lines = [
        f"## {datetime.now(UTC).date().isoformat()} · commit {commit}",
        "",
        "### Stronger pretrained features: what a DINOv2 probe costs per candidate",
        "",
        "The DINOv2 ViT-S/14 probe against the re-verifier, same machine, same session, "
        "each engine in its own process. The timed path is `ReVerifier.classify_batch`'s "
        "plus what DINOv2 needs: upsample 64 -> 224, normalise, forward, pool "
        f"(`{results[-1]['pooling']}`), linear head, softmax, back to the host. "
        "`per_channel` is three forwards per candidate. `scripts/dinov2_latency.py`.",
        "",
        "| engine | device | batch 1 p50 | p90 | p99 | per candidate at batch 8 | "
        "cold first call | weights | peak RSS | × ResNet-18 (batch 1) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        same_device_resnet = next(x for x in results
                                  if x["engine"] == "resnet18" and x["device"] == r["device"])
        ratio = r["single"]["p50"] / same_device_resnet["single"]["p50"]
        lines.append(
            f"| {r['engine']} | {r['device']} | {r['single']['p50']:.2f} ms | "
            f"{r['single']['p90']:.2f} ms | {r['single']['p99']:.2f} ms | "
            f"{r['batch8_per_candidate_ms']:.2f} ms | {r['cold_ms']:.0f} ms | "
            f"{r['weights_mb']:.0f} MB | {r['peak_rss_mb']:.0f} MB | {ratio:.1f}× |")
    lines += [
        "",
        f"The ResNet-18 measured here on CPU is {resnet_cpu['single']['p50']:.2f} ms at "
        f"batch 1; the published figure is {PUBLISHED_RESNET_CPU_MS} ms (2026-08-24). "
        "The ratios above use this session's ResNet, not the published one.",
        "",
        f"Contention before: Ollama {before['ollama'] or 'none'}, processes "
        f"{len(before['processes'])}; after: Ollama {after['ollama'] or 'none'}, "
        f"processes {len(after['processes'])}.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--one", choices=ENGINES)
    parser.add_argument("--device", choices=DEVICES, default="cpu")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--batch-repeats", type=int, default=50)
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "docs" / "benchmarks.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.one:
        print(json.dumps(measure_one(args.one, args.device, args.iterations, args.batch_repeats)))
        return 0

    before = contention()
    results = []
    for device in DEVICES:
        for engine in ENGINES:
            print(f"measuring {engine} on {device} ...", flush=True)
            run = subprocess.run(
                [sys.executable, __file__, "--one", engine, "--device", device,
                 "--iterations", str(args.iterations), "--batch-repeats", str(args.batch_repeats)],
                capture_output=True, text=True, check=False)
            if run.returncode != 0:
                print(run.stderr[-3000:], file=sys.stderr)
                return 1
            results.append(json.loads(run.stdout.strip().splitlines()[-1]))
    after = contention()

    commit = probe.commit()
    body = render(results, before, after, commit)
    print("\n" + body)
    contended = any((before["ollama"], before["processes"], after["ollama"], after["processes"]))
    if contended:
        print("contended: " + json.dumps({"before": before, "after": after}), file=sys.stderr)
    if args.dry_run or contended:
        print("(not appended)" + (" -- the machine was not quiet" if contended else ""))
        return 0 if args.dry_run else 3
    existing = args.benchmarks.read_text()
    args.benchmarks.write_text(existing.rstrip() + "\n\n" + body.rstrip() + "\n")
    print(f"appended to {args.benchmarks.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
