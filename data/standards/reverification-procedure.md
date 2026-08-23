---
defect_class: any
---
# WI-300 AOI re-verification station — operating procedure

## Purpose
Automated optical inspection is tuned for recall and therefore over-calls. This
procedure covers disposition of AOI-flagged regions before a board is released
or routed to rework.

## Decision authority
1. **Model-dismissed.** Regions the re-verification model assigns to false call
   above the configured threshold are removed from the queue without operator
   review. The threshold is set from the operating-point analysis and may only
   be changed by quality engineering.

   Dismissal is reserved to this threshold. No later stage may take a region
   off the queue: automated analysis downstream of it confirms a defect or
   escalates, and does nothing else. The dismissal threshold is the only place
   in the station where the escape budget is spent, so it is the only place
   where it can be audited.
2. **Model-classified, high confidence.** The verdict stands and the board is
   dispositioned per the relevant work instruction (WI-201 to WI-206). The
   confidence at which analysis is skipped rather than run is a cost setting,
   not a decision: the disposition is the same either way, and what running it
   adds is the written rationale on the record. It shall not be set below the
   escalation threshold in §3.
3. **Low confidence, or a conditional class near its limit.** Escalate to an
   operator. The operator's verdict is final and is recorded against the
   candidate. Like the dismissal threshold in §1, the confidence at which this
   applies is set from the operating-point analysis and not fixed by this
   document; it shall be at or above the floor in the escalation triggers
   below.

## Escalation triggers
Escalate when any of the following holds:
- the model's top class carries confidence below 0.70. This is a floor, not the
  operating threshold: escalation is mandatory below it and the configured
  threshold is normally higher. A station escalating only below 0.70 does not
  meet the escape budget in QP-110 — measured, it dismisses real defects the
  analysis was never confident about
- the top two classes are within 0.15 of each other
- the class is `open` and the evidence is not unambiguous
- the same coordinates were escalated on the previous panel of the same lot
- the automated analysis has not returned a verdict within the response budget

## Response budget
Automated analysis of one escalated region shall return a verdict within **10
seconds**. On expiry the region is escalated to an operator unanswered.

The budget follows from QP-110: retaining a false call costs an operator a few
seconds, so analysis that takes longer than an operator would have taken to
look has already spent the saving it exists to make. A station blocked waiting
on analysis is worse than one that escalates, because the queue behind it
continues to fill.

The budget is a property of the station, not of the analysis method. If the
configured model cannot meet it, the model is the wrong size for this line and
is to be changed; the budget is not to be raised to accommodate it.

## Records
Every disposition is recorded with its source -- model, agent or human -- and a
rationale. Operator corrections are the training data for the next model
revision; an unrecorded correction is a correction that never happened.
