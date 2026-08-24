/* A ruler for the three classes that are judged by a ratio.
 *
 * WI-203 accepts a mousebite at >=80% of the nominal conductor width, WI-204 a
 * spur at >=50% of the nominal clearance, WI-206 a pin-hole under 25% of the
 * conductor width. All three are questions the classifier does not answer --
 * WI-203 says so itself, "escalate for measurement" -- and until now the person
 * they were escalated to had nothing on screen to measure with. They are the
 * last stop, so "escalate" had nowhere left to go.
 *
 * **The tool measures a ratio, and that is what makes it honest.** DeepPCB
 * carries no scale: there is no mm-per-pixel anywhere in this project, so a
 * ruler reporting 0.15mm would be inventing the only number that mattered.
 * A ratio needs no calibration -- and all three criteria are already written as
 * ratios, which is not a coincidence but the reason acceptance criteria are
 * usually written that way.
 *
 * What it cannot do is compare across the gap between two panels. The three
 * panels are the same region at the same scale, so a length in one is
 * comparable with a length in another; a single segment spanning the gap is
 * not a length at all, and `segment` refuses one rather than returning a
 * number that looks fine.
 *
 * Geometry only in here, and no DOM: the page draws, this decides. That is
 * what lets `tests/fixtures/measure_harness.js` drive it under node.
 */
(function (global) {
  'use strict';

  /* The panels of the triptych, in the geometry `station/images.py` builds:
   * CONTEXT_SIZE * SCALE per panel, PANEL_GAP between. Kept as fractions of
   * the rendered width so the tool works at whatever size the page displays
   * the image -- and asserted against the server's constants by a test, because
   * two copies of a layout are two things to keep true. */
  var PANELS = 3;

  /* limit: the ratio the work instruction gives. `atLeast` says which side of
   * it is acceptable -- a mousebite must keep *at least* 80% of its width, a
   * pin-hole must stay *under* 25% of it. Getting that backwards would pass
   * every reject, so it is data rather than a branch. */
  var CRITERIA = {
    mousebite: { limit: 0.80, atLeast: true,
                 reference: 'nominal conductor width',
                 measured: 'remaining width at the notch' },
    spur:      { limit: 0.50, atLeast: true,
                 reference: 'nominal clearance',
                 measured: 'remaining clearance' },
    'pin-hole': { limit: 0.25, atLeast: false,
                 reference: 'conductor width',
                 measured: 'void diameter' }
  };

  function criterionFor(defectClass) {
    return CRITERIA[defectClass] || null;
  }

  /* Which panel an x lies in, or -1 for the gap between two. */
  function panelOf(x, width, gapFraction) {
    var panel = (1 - gapFraction * (PANELS - 1)) / PANELS;
    for (var index = 0; index < PANELS; index++) {
      var start = index * (panel + gapFraction);
      if (x >= start && x <= start + panel) return index;
    }
    return -1;
  }

  /* A segment between two points, in fractions of the rendered width.
   *
   * Returns null when the two ends are not in the same panel: the gap is
   * canvas, not board, so a length across it is a number with no meaning, and
   * a tool that returned one anyway would be worse than no tool.
   */
  function segment(a, b, width, height, gapFraction) {
    if (panelOf(a.x, width, gapFraction) === -1) return null;
    if (panelOf(a.x, width, gapFraction) !== panelOf(b.x, width, gapFraction)) {
      return null;
    }
    // Back to pixels before measuring: x and y are fractions of *different*
    // sides, so a length taken in fraction space would be stretched by the
    // aspect ratio -- and the triptych is three times wider than it is tall.
    var dx = (b.x - a.x) * width;
    var dy = (b.y - a.y) * height;
    return { length: Math.sqrt(dx * dx + dy * dy),
             panel: panelOf(a.x, width, gapFraction) };
  }

  /* The reading: the ratio, and what the work instruction makes of it.
   *
   * `verdict` is one of 'within', 'outside', or 'incomparable' -- the last when
   * the two segments were taken in different panels, which compares a length on
   * the template against a length on the test board. Those are the same scale
   * and the comparison is arithmetically fine, which is exactly why it needs
   * refusing explicitly: it is a ratio of two different boards.
   */
  function reading(reference, measured, criterion) {
    if (!reference || !measured || !criterion) return null;
    if (reference.length === 0) return null;
    if (reference.panel !== measured.panel) {
      return { ratio: null, verdict: 'incomparable' };
    }
    var ratio = measured.length / reference.length;
    var within = criterion.atLeast
      ? ratio >= criterion.limit
      : ratio < criterion.limit;
    return { ratio: ratio, verdict: within ? 'within' : 'outside' };
  }

  global.AoiMeasure = {
    CRITERIA: CRITERIA,
    criterionFor: criterionFor,
    panelOf: panelOf,
    segment: segment,
    reading: reading
  };
}(typeof globalThis !== 'undefined' ? globalThis : this));
