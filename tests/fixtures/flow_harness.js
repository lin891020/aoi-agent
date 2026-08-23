/* Drives station/static/flow.js from pytest.
 *
 * The flow view derives everything it draws from the SSE events, and that
 * derivation is the part that can be wrong -- a branch that never resolves, a
 * refusal drawn as a run that finished. It is JavaScript, so this is what the
 * Python tests call: node evaluates flow.js in a bare context, this file feeds
 * it a scenario on stdin and prints what came back as JSON.
 *
 * The document handed to `render` is a fake, and deliberately a hostile one:
 * `innerHTML` throws on both read and write, so a future edit that reaches for
 * it -- the obvious thing to reach for when building SVG -- fails a test
 * rather than shipping. Text is only observable through `textContent`, which
 * is the property everything is required to go through.
 *
 * Usage: node flow_harness.js <path-to-flow.js>  < scenario.json
 */
'use strict';

const fs = require('fs');
const vm = require('vm');

function element(ns, tag, doc) {
  const self = {
    ns: ns,
    tag: tag,
    attrs: {},
    children: [],
    text: null,
    ownerDocument: doc
  };
  Object.defineProperty(self, 'firstChild', {
    get: function () { return self.children.length ? self.children[0] : null; }
  });
  Object.defineProperty(self, 'textContent', {
    get: function () { return self.text; },
    set: function (value) { self.text = String(value); }
  });
  Object.defineProperty(self, 'innerHTML', {
    get: function () { throw new Error('innerHTML was read'); },
    set: function () { throw new Error('innerHTML was assigned'); }
  });
  self.setAttribute = function (name, value) { self.attrs[name] = String(value); };
  self.appendChild = function (child) { self.children.push(child); return child; };
  self.removeChild = function (child) {
    self.children = self.children.filter(function (one) { return one !== child; });
    return child;
  };
  return self;
}

function fakeDocument() {
  const doc = {};
  doc.createElementNS = function (ns, tag) { return element(ns, tag, doc); };
  return doc;
}

function plain(node) {
  return {
    tag: node.tag,
    ns: node.ns,
    attrs: node.attrs,
    text: node.text,
    children: node.children.map(plain)
  };
}

const source = fs.readFileSync(process.argv[2], 'utf8');
const context = {};
vm.createContext(context);
vm.runInContext(source, context);
const flow = context.AoiFlow;

const scenario = JSON.parse(fs.readFileSync(0, 'utf8'));
const state = flow.model(scenario.events || []);
const doc = fakeDocument();
const svg = doc.createElementNS('http://www.w3.org/2000/svg', 'svg');
flow.render(svg, state);

process.stdout.write(JSON.stringify({
  phase: state.phase,
  interpretation: state.interpretation,
  branches: state.branches,
  outstanding: state.outstanding,
  failed: state.failed,
  reached: state.reached,
  stopped: state.stopped,
  finished: state.finished,
  running: flow.running(state),
  label: flow.label(state),
  stages: flow.stages(state),
  svg: plain(svg)
}));
