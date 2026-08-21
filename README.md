# AOI-Agent

**Every board the AOI flags gets a second look — from a model, not a person.**

Automated optical inspection on a PCB line is tuned for recall, so it over-flags.
Every flagged region goes to a human at a re-verification station, and most of
what they look at turns out to be nothing. This project puts a model in front of
that queue.

Status: work in progress. The vision pipeline and its evaluation are done; the
MCP tool layer and the agent that drives them are next.

## Result

On the official DeepPCB test split — 8,143 AOI candidates from 499 unseen boards:

| escape budget | achieved escape rate | manual review removed |
|---|---|---|
| ≤0.25% | 0.23% | **50.2%** |
| ≤0.50% | 0.47% | **56.2%** |
| ≤1.00% | 0.97% | **60.6%** |

Accuracy is deliberately not the headline. Dismissing a real defect ships a bad
board; keeping a false call costs an operator a few seconds. The two errors are
not interchangeable, so the model is reported as a curve against an escape
budget rather than as a single number.

See [docs/benchmarks.md](docs/benchmarks.md) for the full report.

## How it works

```
template ─┐
          ├─→ difference + threshold + connected components ─→ candidates
test ─────┘         "AOI simulator"                                │
                                                                   ▼
                                                    ResNet-18 re-verifier
                                                                   │
                                              ┌────────────────────┴───────┐
                                         dismissed                    reviewed
                                       (false call)                  (by a human)
```

The dataset contains only real defects, so the false calls had to be *generated*
rather than fabricated: a plain template-differencing detector — the same
principle real AOI uses — produces high-recall candidates, and matching those
against the ground-truth boxes labels each one as a genuine defect or a false
call. The result is a re-verification problem with real inputs at both ends.

## Data

[DeepPCB](https://github.com/tangsanli5201/DeepPCB) (MIT): 1,500 aligned
template/tested image pairs, 640×640, six defect classes. Note that the dataset
authors augmented the tested images with artificial defects on top of the real
ones, and binarised every image to remove illumination variation.

```bash
git clone --depth 1 https://github.com/tangsanli5201/DeepPCB.git data/DeepPCB
```

## Running it

```bash
uv sync
uv run python scripts/gate_check.py        # does differencing produce false calls?
uv run python scripts/build_patches.py --split trainval
uv run python scripts/build_patches.py --split test
uv run python scripts/train.py
uv run python scripts/report.py            # the operating-point table
```

## Known limits

- **The AOI stage dominates the escape rate.** The simulator misses 5.0% of
  defects outright, against 0.47% added by the re-verifier. Improving the model
  past this point buys nothing until the detector's recall improves.
- Escapes concentrate in the `open` class (1.35% at the ≤0.5% budget) — thin
  breaks in a trace, the hardest thing to tell from a registration artefact.
- DeepPCB ships pre-registered and binarised, which removes the two largest
  real-world sources of false calls. Numbers here are optimistic relative to a
  line running on raw camera output.
- The simulator's 3x3 opening kernel erases the thin slivers misregistration
  leaves along trace edges, so the false calls it produces come from other
  causes. Measured: a 2 px template shift changes 456 pixels on a synthetic
  board and yields zero candidates at the default settings. A real AOI is
  noisier than this one.
