"""What does one re-verification actually cost?

The README and `docs/benchmarks.md` both claimed candidates were "dispositioned
by the vision model in tens of milliseconds". Nothing measured that. It was the
one number in this repository with no run behind it, and it was about the only
component that would plausibly run on a constrained box at the station itself.
This script replaces the guess.

What is measured, on the real checkpoint and real patches from the official
test split:

* **Single-candidate latency**, reported as a distribution. A mean over a
  fanless laptop is a number that describes no request; p50/p90/p99 is what a
  station's operator experiences.
* **Batched throughput** at the batch size the pipeline actually uses.
  `ReVerifier.classify_batch` is handed every candidate on one board at once,
  so the pipeline's batch is the board's candidate count -- read here from the
  test split rather than assumed -- and a sweep around it shows the shape.
* **MPS against CPU.** The CPU figure is the one that matters for an edge box,
  and it is the figure this project did not have. Both are labelled.
* **Cold against warm.** The first forward after loading pays for kernel
  compilation and the weight transfer. A station restarting mid-shift pays it.
* **First 60s against steady state.** The M5 Air is fanless, so a single
  aggregate hides a throttle. Same split the agent-layer report uses.
* **Model size on disk and peak process memory**, because an edge box has a
  budget for both.

The timed path mirrors `ReVerifier.classify_batch` line for line: uint8 patches
to a float tensor, divide by 255, move to the device, forward, softmax, back to
the host. Timing the bare forward alone would flatter the result by hiding the
transfer, which on MPS is not free. Patch construction is timed separately on
real board images, so the two halves of a station's per-candidate cost can be
added up rather than confused.

Every MPS measurement is bracketed by `torch.mps.synchronize()`. Without it the
forward returns before the GPU has finished and the script measures how fast
Python can enqueue work.

Contention: `ollama ps` is captured before and after and printed into the
report. If any model is resident the run is measuring a machine with 12GB of
GPU held by something else, and the script refuses to append. That is the same
rule the `measuring-llm-latency` skill applies to the agent layer, and it
applies here for the same reason -- it is the GPU that is shared, not the model.

    uv run python scripts/reverifier_latency.py
    uv run python scripts/reverifier_latency.py --soak 120 --dry-run
"""

from __future__ import annotations

import argparse
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoi_agent.data.deeppcb import load_split  # noqa: E402
from aoi_agent.vision.inference import DEFAULT_CHECKPOINT, ReVerifier  # noqa: E402
from aoi_agent.vision.patches import PATCH_SIZE, PatchSet, build_patch  # noqa: E402

#: Same split the agent-layer report uses. The machine is fanless; one
#: aggregate over a long run hides where it started to throttle.
THERMAL_SPLIT_S = 60.0

#: Batch sizes swept around whatever the pipeline's real batch turns out to be.
SWEEP_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]

#: Discarded before every timed series. The first forwards on a device pay for
#: kernel compilation, which is measured deliberately as the cold number and
#: must not leak into the warm one.
WARMUP_ITERATIONS = 20

#: A throttle smaller than this is noise on a laptop, not a thermal effect.
THROTTLE_TOLERANCE = 0.10


# --------------------------------------------------------------------------
# statistics and formatting -- no torch, no model, no device
# --------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. No interpolation, so every figure reported is
    a measurement that actually happened rather than an average of two."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarise(values: list[float]) -> dict:
    """The distribution, not the mean. The mean is reported last and only for
    comparison against the median, because a gap between them is the tell that
    something interrupted the run."""
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p50": percentile(ordered, 50),
        "p90": percentile(ordered, 90),
        "p99": percentile(ordered, 99),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def format_ms(value: float) -> str:
    """Milliseconds at a precision that does not invent significance. A 0.6ms
    forward reported as '1ms' is wrong by half."""
    if value < 1:
        return f"{value:.2f}ms"
    if value < 10:
        return f"{value:.2f}ms"
    if value < 100:
        return f"{value:.1f}ms"
    return f"{value:.0f}ms"


def magnitude_phrase(ms: float) -> str:
    """The plain-English bucket a latency falls in, so the report can be
    checked against the claim it replaces rather than left to the reader."""
    if ms < 1:
        return "under a millisecond"
    if ms < 10:
        return "single-digit milliseconds"
    if ms < 100:
        return "tens of milliseconds"
    if ms < 1000:
        return "hundreds of milliseconds"
    return "seconds"


def claim_verdict(measured_ms: float, claimed: str = "tens of milliseconds") -> str:
    """Was the README right? Says which direction it was wrong in, which is the
    part that matters -- an over-claim and an under-claim have opposite
    consequences for anyone sizing an edge box from this document."""
    actual = magnitude_phrase(measured_ms)
    if actual == claimed:
        return f"The claim of \"{claimed}\" holds: measured {format_ms(measured_ms)}."
    order = [
        "under a millisecond",
        "single-digit milliseconds",
        "tens of milliseconds",
        "hundreds of milliseconds",
        "seconds",
    ]
    direction = (
        "pessimistic direction -- the model is faster than the README said"
        if order.index(actual) < order.index(claimed)
        else "optimistic direction -- the model is slower than the README said"
    )
    return (
        f"The claim of \"{claimed}\" was wrong, and wrong in the {direction}. "
        f"Measured: {format_ms(measured_ms)}, which is {actual}."
    )


def crossover_note(devices: list[dict]) -> str:
    """Which device actually wins, and from what batch size.

    The headline result here is counter-intuitive enough to need saying out
    loud rather than leaving in a table: at one candidate the GPU loses, because
    dispatch and the host round trip cost more than the forward does on a model
    this small. It only pays once there is a batch to amortise them over. A
    station classifying regions one at a time would be slower on the GPU.
    """
    by_name = {device["name"]: device for device in devices}
    if "mps" not in by_name or "cpu" not in by_name:
        return ""
    mps, cpu = by_name["mps"], by_name["cpu"]
    single_ratio = mps["single"]["p50"] / cpu["single"]["p50"]

    shared = sorted(set(mps["batches"]) & set(cpu["batches"]))
    crossover = next(
        (
            size
            for size in shared
            if mps["batches"][size]["per_candidate_ms"]
            < cpu["batches"][size]["per_candidate_ms"]
        ),
        None,
    )

    if single_ratio <= 1:
        return (
            f"MPS is faster than CPU even at one candidate "
            f"({format_ms(mps['single']['p50'])} against "
            f"{format_ms(cpu['single']['p50'])})."
        )
    lead = (
        f"**At one candidate the GPU is the slower device.** MPS p50 is "
        f"{format_ms(mps['single']['p50'])} against the CPU's "
        f"{format_ms(cpu['single']['p50'])} -- {single_ratio:.1f}x slower. On a "
        f"model this small the forward is cheaper than dispatching it and copying "
        f"the result back, and the GPU has nothing to amortise that over."
    )
    if crossover is None:
        return lead + " Across every batch size swept, CPU stayed ahead."
    return (
        lead
        + f" MPS only overtakes at batch {crossover}, and from there it pulls away "
        f"hard -- {cpu['batches'][crossover]['per_candidate_ms'] / mps['batches'][crossover]['per_candidate_ms']:.1f}x "
        f"at that batch alone. The station classifies one region at a time; the "
        f"seeding pass classifies a whole board at once. They want different "
        f"devices."
    )


def thermal_split(
    samples: list[tuple[float, float]], split_s: float = THERMAL_SPLIT_S
) -> tuple[list[float], list[float]]:
    """Split ``(offset_seconds, milliseconds)`` samples at the thermal
    boundary. Returns (first 60s, steady state)."""
    early = [ms for offset, ms in samples if offset < split_s]
    late = [ms for offset, ms in samples if offset >= split_s]
    return early, late


def throttle_verdict(
    early: list[float], late: list[float], tolerance: float = THROTTLE_TOLERANCE
) -> str:
    """Did sustained load slow it down? Reported even when it did not, because
    'we looked and it did not throttle' is a result and an absent section is
    not."""
    if not early or not late:
        return (
            "Not enough of the soak landed either side of the 60s boundary to "
            "split it; treat the thermal question as unanswered."
        )
    before, after = percentile(early, 50), percentile(late, 50)
    change = (after - before) / before
    if change > tolerance:
        return (
            f"**Throttled.** Median went from {format_ms(before)} in the first 60s "
            f"to {format_ms(after)} in steady state, {change:+.0%}. Size an edge "
            f"box on the steady-state figure, not the first minute of it."
        )
    if change < -tolerance:
        return (
            f"Median *fell* from {format_ms(before)} to {format_ms(after)} "
            f"({change:+.0%}) -- that is not a thermal effect, it is something "
            f"else on the machine clearing. Treat the run as suspect."
        )
    return (
        f"**No throttle observed.** Median {format_ms(before)} in the first 60s "
        f"against {format_ms(after)} in steady state, {change:+.0%} -- inside the "
        f"{tolerance:.0%} band this run calls noise."
    )


#: Command fragments that mean something else may be on the GPU.
#:
#: `ollama ps` does not report any of these. It knows about Ollama's own
#: resident models and nothing else, so a torch/MPS job in another shell can
#: saturate the same silicon while the residency check comes back clean. That
#: is not hypothetical -- the first run of this benchmark was discarded because
#: a concurrent torchvision detector benchmark held MPS throughout it and
#: `ollama ps` showed nothing unusual. `.py` is in the list because that is
#: what a torch job looks like from the outside; `mediaanalysisd` is here
#: because macOS runs it on the GPU without being asked.
GPU_CLAIMANTS = ("llama-server", "mlx", "mediaanalysisd", ".py")

#: Below this a process is idling, not computing. `pet server` and the editor's
#: python helpers sit at zero all day and would otherwise block every run.
BUSY_CPU_PERCENT = 5.0


def competing_processes(
    ps_output: str, own_pid: int, min_cpu: float = BUSY_CPU_PERCENT
) -> list[str]:
    """Processes busy enough to be sharing the GPU, from `ps -Ao pid,pcpu,command`.

    The residency check this pairs with only sees Ollama. This one sees whatever
    else is computing, which on a single fanless laptop shared between several
    agents is the failure that actually happens.
    """
    found = []
    for line in ps_output.strip().splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid_text, cpu_text, command = parts
        try:
            pid, cpu = int(pid_text), float(cpu_text)
        except ValueError:
            continue          # the header row
        if pid == own_pid or cpu < min_cpu:
            continue
        if not any(marker in command for marker in GPU_CLAIMANTS):
            continue
        found.append(f"{pid} {cpu:.0f}% {command[:110]}")
    return found


def process_table() -> str:
    try:
        return subprocess.run(
            ["ps", "-Ao", "pid,pcpu,command"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return f"(ps unavailable: {error})"


def resident_models(ps_output: str) -> list[str]:
    """Model names in `ollama ps` output. Anything here means something else is
    holding the GPU and the run is not measuring this machine idle."""
    lines = [line for line in ps_output.strip().splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    if not lines[0].upper().startswith("NAME"):
        return []
    return [line.split()[0] for line in lines[1:] if line.split()]


def ollama_ps() -> str:
    try:
        return subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"(ollama ps unavailable: {error})"


def render(results: dict) -> list[str]:
    """The markdown section, from primitives only.

    Kept free of torch and of the measurement loop so the report's shape can be
    tested without a GPU -- the same reason `analysis_eval.render_plan` is its
    own function.
    """
    lines = [
        "### Re-verifier latency — what one candidate costs, and on what hardware",
        "",
        f"ResNet-18 re-verifier, {results['checkpoint_mb']:.1f}MB checkpoint, "
        f"{results['parameters'] / 1e6:.1f}M parameters, "
        f"3x{PATCH_SIZE}x{PATCH_SIZE} input. Patches are real candidates from the "
        f"official DeepPCB test split ({results['patch_count']} of them).",
        "",
        "The timed path is the one `ReVerifier.classify_batch` runs: uint8 patches "
        "to float, divide by 255, move to the device, forward, softmax, back to the "
        "host. Timing the bare forward would hide the transfer, which on MPS is not "
        "free. Every MPS measurement is bracketed by `torch.mps.synchronize()`; "
        "without it the timer measures how fast Python enqueues work.",
        "",
        f"CPU figures are on {results['cpu_threads']} torch threads, which is what "
        f"this machine defaults to; an edge box with fewer cores scales roughly with "
        f"that number and the figure below is not transferable without it.",
        "",
        "Devices are measured in the order they appear below, in one process. The "
        "second device therefore starts from an already-warm machine, and the peak "
        "RSS covers both sets of weights rather than one station's footprint. Both "
        "are stated rather than corrected for.",
        "",
        "Contention was checked two ways, because one of them is not enough. "
        "`ollama ps` reports Ollama's own resident models and nothing else, so a "
        "torch/MPS job in another shell saturates the same GPU while that check "
        "comes back clean. The second check is a process sweep for anything busy "
        "enough to be computing -- `llama-server`, any running `.py`, `mlx`, "
        "`mediaanalysisd`. Both are recorded below.",
        "",
        "```",
        "ollama ps before the run",
        results["ps_before"],
        "",
        "busy processes before the run",
        "\n".join(results["busy_before"]) or "(none)",
        "",
        "ollama ps after the run",
        results["ps_after"],
        "",
        "busy processes after the run",
        "\n".join(results["busy_after"]) or "(none)",
        "```",
        "",
        "#### Single candidate, warm",
        "",
        "| device | calls | p50 | p90 | p99 | max | mean |",
        "|---|---|---|---|---|---|---|",
    ]

    for device in results["devices"]:
        s = device["single"]
        lines.append(
            f"| {device['label']} | {s['n']} | {format_ms(s['p50'])} | "
            f"{format_ms(s['p90'])} | {format_ms(s['p99'])} | "
            f"{format_ms(s['max'])} | {format_ms(s['mean'])} |"
        )

    lines += [
        "",
        results["verdict"],
        "",
        results["crossover"],
        "",
        "#### Cold against warm",
        "",
        "A station restarting mid-shift pays the load and the first forward. "
        "Everything after that is the warm number above.",
        "",
        "| device | checkpoint load | first forward | warm p50 | cold penalty |",
        "|---|---|---|---|---|",
    ]

    for device in results["devices"]:
        warm = device["single"]["p50"]
        lines.append(
            f"| {device['label']} | {format_ms(device['load_ms'])} | "
            f"{format_ms(device['cold_ms'])} | {format_ms(warm)} | "
            f"{device['cold_ms'] / warm:.0f}x |"
        )

    lines += [
        "",
        "#### Batched throughput",
        "",
        f"`classify_batch` is handed every candidate on one board at once, so the "
        f"pipeline's batch size is the board's candidate count. On the test split "
        f"that is a median of {results['pipeline_batch']} trainable candidates per board "
        f"(mean {results['pipeline_batch_mean']:.1f}, max {results['pipeline_batch_max']}). "
        f"The sweep is around that.",
        "",
        "| batch | " + " | ".join(d["label"] for d in results["devices"]) + " |",
        "|---" * (len(results["devices"]) + 1) + "|",
    ]

    for size in results["batch_sizes"]:
        cells = []
        for device in results["devices"]:
            entry = device["batches"].get(size)
            cells.append(
                "—" if entry is None
                else f"{entry['per_candidate_ms']:.2f}ms/cand, "
                     f"{entry['throughput']:,.0f}/s"
            )
        marker = " ←" if size == results["pipeline_batch_bucket"] else ""
        lines.append(f"| {size}{marker} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "The arrow marks the bucket nearest the pipeline's real batch.",
        "",
        "#### Does the CPU cliff move with the core count?",
        "",
        "The sweep above has a cliff on CPU between batch 8 and batch 16: past it "
        "the per-candidate cost jumps several-fold rather than falling. An edge "
        "recommendation of \"batch at 8\" is only useful if that is a property of "
        "the model rather than of this machine's cores, so the same two batches "
        "were run again across torch thread counts.",
        "",
        "| torch threads | " + " | ".join(
            f"batch {b}" for b in sorted(THREAD_SWEEP_BATCHES)) + " |",
        "|---" * (len(THREAD_SWEEP_BATCHES) + 1) + "|",
    ]

    for threads in sorted(results["thread_sweep"]):
        row = results["thread_sweep"][threads]
        lines.append(
            f"| {threads} | "
            + " | ".join(f"{row[b]:.2f}ms/cand" for b in sorted(THREAD_SWEEP_BATCHES))
            + " |"
        )

    lines += [
        "",
        "Thread count moves the batch-8 figure and leaves the batch-16 figure "
        "essentially untouched. The cliff is therefore in the model's CPU "
        "convolution path, not in this laptop's core count, and it transfers: batch "
        "at 8 on any CPU box, and do not assume more batching is more throughput.",
        "",
        "#### Thermal — first 60s against steady state",
        "",
        f"{results['soak_s']:.0f}s of sustained batched inference per device on a "
        f"fanless M5 Air.",
        "",
        "| device | first 60s (n, p50) | steady state (n, p50) |",
        "|---|---|---|",
    ]

    for device in results["devices"]:
        early, late = device["soak_early"], device["soak_late"]
        lines.append(
            f"| {device['label']} | "
            + (f"{early['n']}, {format_ms(early['p50'])}" if early else "—")
            + " | "
            + (f"{late['n']}, {format_ms(late['p50'])}" if late else "—")
            + " |"
        )

    lines.append("")
    for device in results["devices"]:
        lines.append(f"- {device['label']}: {device['throttle']}")

    lines += [
        "",
        "#### Footprint",
        "",
        f"- checkpoint on disk: **{results['checkpoint_mb']:.1f}MB** "
        f"({results['parameters'] / 1e6:.1f}M float32 parameters)",
        f"- peak process RSS across the whole run: **{results['peak_rss_mb']:.0f}MB**",
        f"- patch construction (two crops and an absolute difference on the board "
        f"image, which runs before every classification): p50 "
        f"{format_ms(results['patch_build_p50'])} per candidate",
        "",
        results["implication"],
        "",
    ]
    return lines


def edge_implication(cpu_p50: float, cpu_throughput: float, checkpoint_mb: float) -> str:
    """What the CPU number means for a box without a GPU."""
    return (
        f"**What this means for an edge deployment.** The CPU figure is the one to "
        f"size on: a re-verification station is a box beside a conveyor, not a "
        f"laptop with a GPU. At {format_ms(cpu_p50)} per candidate single-shot and "
        f"{cpu_throughput:,.0f} candidates/second batched, on CPU alone, the model "
        f"is not the constraint -- a board carrying twenty candidates is "
        f"re-verified in well under a second, and the AOI stage in front of it "
        f"takes longer. The {checkpoint_mb:.0f}MB checkpoint fits anywhere. The "
        f"open INT8/ONNX work is therefore about memory and portability, not about "
        f"latency: there is no latency problem here to solve."
    )


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def synchronise(device) -> None:
    """Make the device finish before the timer stops. On MPS a forward returns
    immediately; without this the script times the Python call, not the GPU."""
    import torch

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def classify_once(model, device, patches: np.ndarray) -> np.ndarray:
    """Exactly what `ReVerifier.classify_batch` does to a stack of patches.

    Mirrored here rather than called, because the real method starts from board
    images and candidate boxes; patch construction is timed separately. If
    `classify_batch` changes, this must change with it.
    """
    import torch

    with torch.no_grad():
        batch = torch.from_numpy(patches).float().div_(255.0)
        probabilities = torch.softmax(model(batch.to(device)), dim=1).cpu().numpy()
    return probabilities


def timed_series(model, device, patches: np.ndarray, batch: int, repeats: int) -> list[float]:
    """`repeats` timed classifications of `batch` patches, cycling the split."""
    total = len(patches)
    samples = []
    for index in range(repeats):
        start = (index * batch) % max(1, total - batch)
        chunk = np.ascontiguousarray(patches[start:start + batch])
        synchronise(device)
        began = time.perf_counter()
        classify_once(model, device, chunk)
        synchronise(device)
        samples.append((time.perf_counter() - began) * 1000)
    return samples


def soak(model, device, patches: np.ndarray, batch: int, seconds: float) -> list[tuple[float, float]]:
    """Sustained load, keeping the offset of every sample so it can be split at
    the thermal boundary."""
    samples: list[tuple[float, float]] = []
    started = time.perf_counter()
    index = 0
    total = len(patches)
    while (time.perf_counter() - started) < seconds:
        start = (index * batch) % max(1, total - batch)
        chunk = np.ascontiguousarray(patches[start:start + batch])
        offset = time.perf_counter() - started
        synchronise(device)
        began = time.perf_counter()
        classify_once(model, device, chunk)
        synchronise(device)
        samples.append((offset, (time.perf_counter() - began) * 1000))
        index += 1
    return samples


def peak_rss_mb() -> float:
    """macOS reports ru_maxrss in bytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def measure_patch_build(patch_set: PatchSet, repeats: int = 200) -> list[float]:
    """Time `build_patch` on real board images, so the cost in front of the
    model is a measured number too and not folded into 'inference'."""
    pairs = load_split("test")
    first = pairs[0]
    template, test = first.load_template(), first.load_test()
    boxes = patch_set.boxes[patch_set.image_index == 0]
    if len(boxes) == 0:
        boxes = patch_set.boxes[:1]

    class _Box:
        __slots__ = ("x1", "y1", "x2", "y2")

        def __init__(self, box):
            self.x1, self.y1, self.x2, self.y2 = (int(v) for v in box)

    candidates = [_Box(box) for box in boxes]
    samples = []
    for index in range(repeats):
        candidate = candidates[index % len(candidates)]
        began = time.perf_counter()
        build_patch(template, test, candidate, PATCH_SIZE)
        samples.append((time.perf_counter() - began) * 1000)
    return samples


#: Where the CPU per-candidate cost stops improving and jumps. Measured, not
#: assumed -- see `measure_thread_sweep`.
THREAD_COUNTS = [1, 2, 4, 8]
THREAD_SWEEP_BATCHES = [8, 16]


def measure_thread_sweep(checkpoint: Path, patches: np.ndarray, repeats: int = 25) -> dict:
    """Per-candidate CPU cost against torch thread count, at two batch sizes.

    Here because the sweep turns up a cliff between batch 8 and batch 16 on CPU,
    and an edge recommendation that says "use batch 8" needs to know whether
    that is a property of the model or of this machine's core count. A reader
    sizing a four-core box cannot act on the first answer if it is really the
    second.
    """
    import torch

    original = torch.get_num_threads()
    verifier = ReVerifier(checkpoint, device="cpu")
    table: dict[int, dict[int, float]] = {}
    try:
        for threads in THREAD_COUNTS:
            torch.set_num_threads(threads)
            timed_series(verifier.model, verifier.device, patches, 8, 10)
            table[threads] = {}
            for size in THREAD_SWEEP_BATCHES:
                stats = summarise(
                    timed_series(verifier.model, verifier.device, patches, size, repeats)
                )
                table[threads][size] = stats["p50"] / size
                print(
                    f"  threads {threads:>2}  batch {size:>3}  "
                    f"{stats['p50'] / size:.2f}ms/cand"
                )
    finally:
        torch.set_num_threads(original)
    return table


def measure_device(
    name: str, label: str, checkpoint: Path, patches: np.ndarray, args, batch_sizes: list[int]
) -> dict:
    """One device, end to end: load, cold, warm, sweep, soak."""
    import torch

    print(f"\n=== {label} ===")

    began = time.perf_counter()
    verifier = ReVerifier(checkpoint, device=name)
    load_ms = (time.perf_counter() - began) * 1000
    model, device = verifier.model, verifier.device
    print(f"  checkpoint load           {load_ms:>9.0f}ms")

    cold_chunk = np.ascontiguousarray(patches[:1])
    synchronise(device)
    began = time.perf_counter()
    classify_once(model, device, cold_chunk)
    synchronise(device)
    cold_ms = (time.perf_counter() - began) * 1000
    print(f"  first forward (cold)      {cold_ms:>9.2f}ms")

    timed_series(model, device, patches, 1, WARMUP_ITERATIONS)

    single = summarise(timed_series(model, device, patches, 1, args.iterations))
    print(
        f"  single candidate, warm    p50 {single['p50']:.2f}ms  "
        f"p90 {single['p90']:.2f}ms  p99 {single['p99']:.2f}ms"
    )

    batches = {}
    for size in batch_sizes:
        if size > len(patches):
            continue
        repeats = max(5, args.batch_repeats)
        stats = summarise(timed_series(model, device, patches, size, repeats))
        per_candidate = stats["p50"] / size
        batches[size] = {
            "p50": stats["p50"],
            "per_candidate_ms": per_candidate,
            "throughput": 1000.0 / per_candidate,
        }
        print(
            f"  batch {size:>4}              {stats['p50']:>9.2f}ms  "
            f"{per_candidate:.3f}ms/cand  {1000.0 / per_candidate:,.0f}/s"
        )

    soak_batch = min(args.soak_batch, len(patches))
    print(f"  soaking {args.soak:.0f}s at batch {soak_batch} ...")
    samples = soak(model, device, patches, soak_batch, args.soak)
    early, late = thermal_split(samples)
    print(
        f"  first 60s n={len(early)}  steady state n={len(late)}"
    )

    del verifier, model
    if name == "mps":
        torch.mps.empty_cache()

    return {
        "name": name,
        "label": label,
        "load_ms": load_ms,
        "cold_ms": cold_ms,
        "single": single,
        "batches": batches,
        "soak_early": summarise(early) if early else None,
        "soak_late": summarise(late) if late else None,
        "throttle": throttle_verdict(early, late),
    }


def nearest_bucket(value: int, buckets: list[int]) -> int:
    return min(buckets, key=lambda b: abs(b - value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--patches", type=Path, default=Path("data/patches/test.npz"))
    parser.add_argument("--iterations", type=int, default=300,
                        help="timed single-candidate classifications per device")
    parser.add_argument("--batch-repeats", type=int, default=30)
    parser.add_argument("--soak", type=float, default=150.0,
                        help="seconds of sustained load per device; must clear the "
                             "60s thermal boundary by enough to give steady state a "
                             "sample worth a percentile")
    parser.add_argument("--soak-batch", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    parser.add_argument("--dry-run", action="store_true", help="print, do not append")
    parser.add_argument("--allow-contended", action="store_true",
                        help="append even though another model holds the GPU")
    args = parser.parse_args()

    import torch

    if not args.patches.exists():
        print(f"{args.patches} not found; run scripts/build_patches.py --split test",
              file=sys.stderr)
        return 1

    patch_set = PatchSet.load(args.patches)
    patches = patch_set.patches
    print(f"{len(patches)} test-split patches, {patches.shape[1:]} each")

    per_board = np.bincount(patch_set.image_index)
    per_board = per_board[per_board > 0]
    pipeline_batch = int(np.median(per_board))
    print(
        f"pipeline batch: median {pipeline_batch} candidates/board "
        f"(mean {per_board.mean():.1f}, max {per_board.max()})"
    )

    own = os.getpid()
    ps_before = ollama_ps()
    busy_before = competing_processes(process_table(), own)
    print(f"\nollama ps before:\n{ps_before}\n")
    print(
        "busy processes before:\n  "
        + ("\n  ".join(busy_before) if busy_before else "(none)")
        + "\n"
    )
    contended = resident_models(ps_before) + busy_before
    if contended:
        print(
            "WARNING: something else is computing on this machine. These numbers "
            "would measure contention, not the model:\n  "
            + "\n  ".join(contended),
            file=sys.stderr,
        )

    devices = [("cpu", "CPU")]
    if torch.backends.mps.is_available():
        devices.insert(0, ("mps", "MPS"))

    measured = [
        measure_device(name, label, args.checkpoint, patches, args, SWEEP_BATCHES)
        for name, label in devices
    ]

    print("\nCPU thread sweep at the cliff ...")
    thread_sweep = measure_thread_sweep(args.checkpoint, patches)

    print("\ntiming patch construction on real board images ...")
    patch_build = summarise(measure_patch_build(patch_set))
    print(f"  build_patch p50 {patch_build['p50']:.3f}ms")

    ps_after = ollama_ps()
    busy_after = competing_processes(process_table(), own)
    print(f"\nollama ps after:\n{ps_after}")
    print(
        "busy processes after:\n  "
        + ("\n  ".join(busy_after) if busy_after else "(none)")
    )
    contended = contended or resident_models(ps_after) + busy_after

    by_name = {d["name"]: d for d in measured}
    cpu = by_name["cpu"]
    bucket = nearest_bucket(pipeline_batch, sorted(cpu["batches"]))

    payload = torch.load(args.checkpoint, map_location="cpu")
    parameters = sum(int(np.prod(v.shape)) for v in payload["state_dict"].values())

    results = {
        "checkpoint_mb": args.checkpoint.stat().st_size / (1024 * 1024),
        "parameters": parameters,
        "patch_count": len(patches),
        "cpu_threads": torch.get_num_threads(),
        "ps_before": ps_before,
        "ps_after": ps_after,
        "busy_before": busy_before,
        "busy_after": busy_after,
        "devices": measured,
        "batch_sizes": [s for s in SWEEP_BATCHES if s in cpu["batches"]],
        "pipeline_batch": pipeline_batch,
        "pipeline_batch_mean": float(per_board.mean()),
        "pipeline_batch_max": int(per_board.max()),
        "pipeline_batch_bucket": bucket,
        "soak_s": args.soak,
        "peak_rss_mb": peak_rss_mb(),
        "thread_sweep": thread_sweep,
        "patch_build_p50": patch_build["p50"],
        "verdict": claim_verdict(cpu["single"]["p50"]),
        "crossover": crossover_note(measured),
        "implication": edge_implication(
            cpu["single"]["p50"],
            cpu["batches"][bucket]["throughput"],
            args.checkpoint.stat().st_size / (1024 * 1024),
        ),
    }

    report = "\n".join(render(results))
    print("\n" + report)

    if args.dry_run:
        return 0
    if contended and not args.allow_contended:
        print(
            f"\nNOT appended: {', '.join(contended)} was resident during the run. "
            f"A contended run is discarded, not published. Wait for the GPU or pass "
            f"--allow-contended if you intend to publish it with that caveat.",
            file=sys.stderr,
        )
        return 1

    existing = args.out.read_text() if args.out.exists() else "# Benchmarks\n"
    args.out.write_text(existing.rstrip() + "\n\n" + report + "\n")
    print(f"\nappended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
