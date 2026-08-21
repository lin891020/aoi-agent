# QP-110 Escape tolerance and review budget

## Principle
The two errors available to a re-verification system are not symmetric. A
dismissed real defect leaves the plant. A retained false call costs an operator
a few seconds. Policy is therefore written as an escape budget, and review
effort is whatever that budget allows.

## Current budget
- Maximum re-verification escape rate: **0.5%** of defects reaching the station.
- No target is set for review volume. Review volume is the dependent variable.

## Whole-line accounting
The re-verification stage only sees what the detector flagged. Defects the
detector missed are already gone and no threshold at this station recovers
them. Report both numbers: the stage escape rate and the line escape rate. When
the detector dominates the line figure, further tuning at this station is
wasted effort and the corrective action belongs upstream.

## Review triggers outside the budget
Regardless of the model's confidence, escalate to a human when the defect class
is `open`, when the board is Class 3 product, or when the lot has already
produced two or more confirmed criticals.
