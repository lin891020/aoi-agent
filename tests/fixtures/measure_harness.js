/* Drives station/static/measure.js from pytest.
 *
 * The ruler's geometry decides whether a reading is meaningful -- two ends in
 * the same panel, two segments on the same board -- and getting that wrong
 * produces a number that looks right, which on an acceptance decision is the
 * worst available failure. It is JavaScript, so node evaluates it here and
 * pytest asserts on what comes back.
 *
 * No DOM is faked because none is needed: `measure.js` is geometry and holds
 * no reference to a document. That separation is the reason this harness is
 * ten lines rather than a hundred.
 *
 * Usage: node measure_harness.js <path-to-measure.js>  < scenario.json
 */
'use strict';

const fs = require('fs');
const vm = require('vm');

const context = { globalThis: {} };
context.globalThis = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);

const scenario = JSON.parse(fs.readFileSync(0, 'utf8'));
const M = context.AoiMeasure;

const width = scenario.width;
const height = scenario.height;
const gap = scenario.gap;

function seg(pair) {
  return pair ? M.segment(pair[0], pair[1], width, height, gap) : null;
}

const reference = seg(scenario.reference);
const measured = seg(scenario.measured);

process.stdout.write(JSON.stringify({
  reference: reference,
  measured: measured,
  reading: M.reading(reference, measured, M.criterionFor(scenario.defect_class)),
  criterion: M.criterionFor(scenario.defect_class),
  panels: [
    M.panelOf(0.05, width, gap),
    M.panelOf(0.34, width, gap),
    M.panelOf(0.50, width, gap),
    M.panelOf(0.95, width, gap)
  ]
}));
