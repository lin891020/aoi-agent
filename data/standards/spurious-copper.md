# WI-205 Spurious copper — acceptance and disposition

## Definition
An isolated island of copper in an area the artwork specifies as clear, not
connected to any conductor.

## Classification
Conditional. Isolated copper is acceptable when it maintains full nominal
clearance from every conductor and does not sit under a component footprint or
a solder-mask opening.

## Disposition
- Clear of all conductors and footprints: release, record.
- Under a footprint or within clearance: rework to remove. Loose copper under a
  component can migrate during reflow.

## Re-verification notes
Spurious copper is the class most often confused with contamination or an
imaging artefact, because both appear as added material in a clear area. Debris
moves between inspections; copper does not. A repeat call at the same
coordinates on a second scan indicates real copper.
