# WI-206 Pin hole — acceptance and disposition

## Definition
A small void entirely inside a conductor or pad, where copper is missing but
the conductor remains continuous around it.

## Classification
Conditional. Acceptable when the void's diameter is under 25% of the conductor
width and it does not fall inside a pad's solderable area.

## Disposition
- Within limits and outside pads: release.
- Inside a pad: reject. Voids in solderable area cause open joints at assembly.

## Re-verification notes
Pin holes are small and round, which makes them easy to classify but easy for a
low-resolution scan to miss entirely. A pin-hole call is usually trustworthy;
the risk with this class is at the detection stage, not re-verification.
