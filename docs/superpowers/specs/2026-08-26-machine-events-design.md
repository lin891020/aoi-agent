# Machine events, and the JOIN they force

**Status:** approved in conversation 2026-08-26 (entity, signal shape and tool
shape each chosen explicitly). This document is the design those choices
produce.

## Why this entity and not another

The independent seventy-question set — written by three people who never saw
the planner's prompt — asked the same thing twice and the system could not
answer it either time:

- **S25** «C 班交接說 M32 有動過參數，動完之後出來的結果有沒有差？» — refused,
  correctly, because nothing in the store records anything *happening to* a
  machine. A refusal is the right answer to the wrong question.
- **S32** «最近人在看的量變多了，是不是機器越判越沒把握？» — answered wrongly
  in two consecutive runs by reaching for `query_false_call_rate`, a snapshot
  tool with no time axis.

Both are questions about **change over time on one machine**. The store today
has no such thing: `lot_id`, `line_id`, `machine_id` and `shift` are bare
string columns on `boards`, so every query walks one foreign-key chain
downwards and nothing ever has to look up what a dimension *is*. The schema is
missing a dimension, and the questions that need it are already measured.

This is the same precedent as `query_false_call_rate`: the tool is built
because the eval asked for it, the fixture is not opened, and the rerun is
scored against an answer key its authors wrote before the tool existed.

**What this is not for.** It is not a JOIN exercise. A table added to practise
a query pattern is the shape this project was killed once for — a mechanism
with nothing to run it on — and the commit message would have to say so. The
reason above is the reason; the JOIN is the price.

## The table

```
machine_events
  id            PK
  machine_id    str, indexed     -- no FK, no machines table (below)
  kind          str, indexed     -- parameter_change | maintenance | lamp_replaced | nozzle_cleaned
  happened_at   datetime, indexed
  note          str | None
  recorded_by   str              -- 'seeded' for now; a signed-in name if the station ever writes one
```

**No `machines` table.** `machine_id` is already a string on `boards` and
every existing query reads it. Normalising it into a foreign key is a
non-additive migration, and this store's migration mechanism is deliberately
additive only (`ADDED_COLUMNS` / `BACKFILL_ON_ADD`, "not a migration framework
and not allowed to become one"). The cost is that an event can name a machine
that does not exist — the `line_id="L4"` failure one level down. So the write
path refuses it: `machine_id` must be in `SELECT DISTINCT machine_id FROM
boards`. Derived from the data, not a hard-coded list.

The `kind` vocabulary is a tuple in `store/events.py`, and the planner's
`Domains` reads it from the table's distinct values, not from the tuple — the
same "what the store actually holds" rule `line_id` and `machine_id` already
follow.

## The seed: four events, one with an effect, three controls

| machine | event | effect |
|---|---|---|
| **M32** | `parameter_change` | **yes** — after the event M32 stops receiving open-heavy boards |
| M31 | `lamp_replaced` | none (control) |
| M21 | `maintenance` | none (control) |
| M12 | `nozzle_cleaned` | none (control) |
| M22 | *no event* | carries the existing planted signal; splitting it weakens every measurement that depends on it |

**The effect reuses the existing mechanism, and only that.** `store/seed.py`
does not write defects — the defects are DeepPCB's real annotations. Its one
lever is *assignment*: which board goes to which machine, ranked on open
share. The event effect is the same lever with a time axis: among boards not
already assigned to M22, the open-heavy ones go to M32 **before**
`happened_at` and not after. So the disclosure in CLAUDE.md ("generated with
one planted signal") becomes "two planted signals, both by assignment", and
the invariant is extended by one sentence rather than replaced.

**The controls are the point.** Without them the tool would be scored on
"is there an event", not "did the event matter". A test on the seed itself
holds three things after every reseed: M22's ranking signal is unchanged; M32
differs before/after by more than its Wilson interval; the three controls do
not. If (c) fails the tool has nothing to be wrong about, and a tool that
cannot be wrong is not being measured.

## The tools

```python
query_machine_events(machine_id: str | None = None, kind: str | None = None) -> dict
    # what happened, to which machine, when — a list, newest first

query_defect_history(..., relative_to: str | None = None, side: str | None = None)
    # existing tool, two new parameters
    # relative_to ∈ the kinds the table actually holds (Domains)
    # side        ∈ ("before", "after")
```

A before/after comparison is therefore **three independent `Send` branches**
— the events lookup and two windows — and the fan-out keeps its shape. The
"fan-out is the shape of the work" invariant is not touched.

The anchor date is resolved **inside** the tool: the newest event of that
`kind` on that `machine_id`. The plan never carries a date, so the plan format
needs no chaining, which it cannot express.

**What the tool refuses**, which matters more than what it returns:

- `relative_to` without `machine_id` — an event belongs to one machine; there
  is no fleet-wide "after".
- `relative_to` naming a kind that machine has no event of — refused, not an
  empty series (the missing series reads as a finding).
- `side` without `relative_to`, or the reverse.

The payload carries `event_at`, `boards_in_window`, and the window actually
covered — the same honesty `query_false_call_rate` already has with
`window_days_covered`.

## The interval, which is what makes it not a toy

Each window returns a **Wilson interval** on "share of flagged regions
confirmed as `open`" (a proportion; `aoi_agent.stats.wilson` already exists).
M32 splits roughly 45/45 boards, so the intervals will be wide. That is
correct, and the tool says so rather than hiding it.

**The tool does not say "it improved."** It returns two intervals and
`intervals_overlap: bool`. Whether to draw a conclusion is the synthesis
prose's job, and the causal-disclaimer machinery it already has applies. A
tool that emits a verdict is the LLM's "confident" flag again, one layer
down.

## Chart, i18n, migration

- **Chart:** two-window results become a before/after bar pair, derived from
  the result shape in `analysis/charts.py` — never chosen by the model.
- **i18n:** every new string in both tables; the parity test enforces it.
- **Migration:** `seed_store.py --migrate-only` creates the table (additive).
  The assignment change means the store must be **reseeded** for the effect
  to exist — the user approved this on 2026-08-26; back up first
  (`aoi_agent.db.bak-before-events`).

## The disclosure that would otherwise go false

CLAUDE.md, "Say what is simulated": *"Production metadata is generated with
one planted signal."* This becomes two. It is edited in the same commit,
because this is exactly the kind of sentence that has gone stale five times
today, and this time it is caught before it does.

## Measurement gate — not optional

The precedent is written down: *the next tool added to this registry re-runs
the independent fixture before shipping.* So:

1. `analysis_eval.py --questions tests/fixtures/analysis_questions_independent.json`
   — fixture unopened, ~65 minutes, adjudication published beside the score.
2. The in-house twenty as well.
3. **This time there is a drift baseline** (same prompt twice moved three rows
   and one point of total), so S25 recovering is readable as signal, and S32
   still failing is readable as the tool not being the fix for it.

Expected: S25 answerable (it names the machine and the event); S32 probably
*still* refused-or-wrong, because it asks about the *model's* confidence over
time and no tool carries that even now. That would be a clean result: one
question closed, one shown to need a different tool, and both scored on an
answer key nobody edited.

## Files

| file | change |
|---|---|
| `src/aoi_agent/store/models.py` | `MachineEvent` table |
| `src/aoi_agent/store/events.py` | new: write guard, `events_for`, `anchor_for` |
| `src/aoi_agent/store/seed.py` | four events; time-split assignment for M32 |
| `src/aoi_agent/mcp_servers/production.py` | `query_machine_events`; two params on `query_defect_history` |
| `src/aoi_agent/analysis/plan.py` | registration, `Domains.kinds`, `DOMAIN_OF` |
| `src/aoi_agent/analysis/charts.py` | before/after builder |
| `src/aoi_agent/i18n.py` | both tables |
| `scripts/seed_store.py` | additive create |
| `CLAUDE.md` | "one planted signal" → two; a paragraph under Still open |
| `tests/test_machine_events.py` | new |
| `tests/test_seed_signal.py` | new: the three seed properties |
| `tests/test_analysis_plan.py`, `test_analysis_charts.py`, `test_i18n.py` | extended |

## Out of scope

- A `machines` table (see above).
- Writing events from the station. `recorded_by` exists so that day needs no
  migration, but no route, template or permission is added here.
- Any camera / optical-head dimension. Deferred until a question set written
  by an equipment engineer says it is wanted.
