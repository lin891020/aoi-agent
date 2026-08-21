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
2. **Model-classified, high confidence.** The verdict stands and the board is
   dispositioned per the relevant work instruction (WI-201 to WI-206).
3. **Low confidence, or a conditional class near its limit.** Escalate to an
   operator. The operator's verdict is final and is recorded against the
   candidate.

## Escalation triggers
Escalate when any of the following holds:
- the model's top class carries confidence below 0.70
- the top two classes are within 0.15 of each other
- the class is `open` and the evidence is not unambiguous
- the same coordinates were escalated on the previous panel of the same lot

## Records
Every disposition is recorded with its source -- model, agent or human -- and a
rationale. Operator corrections are the training data for the next model
revision; an unrecorded correction is a correction that never happened.
