/* The live flow view for /ask: planning, the fan-out, the join, the writing.
 *
 * Enhancement only. Nothing on the page depends on this file -- with scripting
 * off the form posts and the server renders the finished run, and the panel
 * this draws into stays hidden.
 *
 * Three rules this file exists under:
 *
 *   1. Every value that reaches the document goes through `.textContent`.
 *      Nothing here concatenates a string into markup, and `innerHTML` is not
 *      assigned anywhere -- clearing is `removeChild`, and every element is
 *      built with `document.createElementNS`. Tool names arrive from the
 *      server's own registry, but the plan they arrive in is model-authored
 *      and the rule does not get to depend on which half a value came from.
 *
 *   2. It derives everything from the events already on the wire. `plan`
 *      carries the branch count and their names, each `tool` carries one
 *      completion, `synthesising` carries the join. There is no event here
 *      that the progress list did not already need.
 *
 *   3. The fan-out is drawn as parallel because it is parallel -- the facts
 *      are independent. Nothing here says or implies it is faster. The tools
 *      cost milliseconds against two model calls of around eight seconds
 *      each, so a picture claiming a saving would be a picture lying.
 *
 * `model` and `stages` are pure functions of the event list and are what the
 * tests drive; `render` is the only part that touches a document, and it takes
 * the one it draws into rather than reaching for a global.
 */
(function (global) {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';

  /* The four stages, in the order the graph runs them. `join` is a boundary
   * rather than a phase -- collect_node is microseconds -- so it never shows
   * as active on its own; it goes done at the moment the join is announced. */
  var STAGES = [
    { id: 'plan', label: '規劃' },
    { id: 'fan', label: '獨立查詢' },
    { id: 'join', label: '匯整' },
    { id: 'write', label: '撰寫回答' }
  ];

  /* Fold the events received so far into what the page has to show.
   *
   * `reached` is the index into STAGES of the stage the run is in. `stopped`
   * is a run that ended before its last stage -- a refusal, a plan that did
   * not validate, a dropped stream -- and is sticky, so a later `done` cannot
   * repaint a stage that never ran as one that finished.
   */
  function model(events) {
    var state = {
      phase: 'planning',
      interpretation: '',
      branches: [],
      reached: 0,
      stopped: false,
      finished: false,
      errors: []
    };

    (events || []).forEach(function (event) {
      var data = event.data || {};
      if (event.event === 'plan') {
        state.interpretation = data.interpretation || '';
        state.branches = (data.calls || []).map(function (call) {
          return { key: call.key, tool: call.tool, state: 'running', ms: null };
        });
        state.reached = 1;
        if (state.branches.length === 0) {
          // A plan with no calls is the model declining. Nothing fans out,
          // nothing joins, and no answer is written -- the graph routes
          // straight to `report`.
          state.phase = 'refused';
          state.stopped = true;
        } else {
          state.phase = 'running';
        }
      } else if (event.event === 'tool') {
        state.branches.forEach(function (branch) {
          if (branch.key === data.key) {
            branch.state = data.ok ? 'ok' : 'failed';
            branch.ms = data.elapsed_ms;
          }
        });
      } else if (event.event === 'synthesising') {
        // The join happened, so the fan-out and the join are both behind us
        // and the second model call is what the wait is now.
        state.reached = 3;
        state.phase = 'synthesising';
      } else if (event.event === 'error') {
        state.errors.push(data.message || '');
        state.stopped = true;
        if (state.phase !== 'refused') state.phase = 'failed';
      } else if (event.event === 'done') {
        state.finished = true;
        if (!state.stopped) state.phase = 'done';
      }
    });

    state.outstanding = state.branches.filter(function (branch) {
      return branch.state === 'running';
    }).length;
    state.failed = state.branches.filter(function (branch) {
      return branch.state === 'failed';
    }).length;
    return state;
  }

  /* One state per stage: done, active, pending, or skipped. */
  function stages(state) {
    return STAGES.map(function (stage, index) {
      var status;
      if (state.stopped) {
        status = index < state.reached ? 'done' : 'skipped';
      } else if (state.finished) {
        status = 'done';
      } else if (index < state.reached) {
        status = 'done';
      } else if (index === state.reached) {
        status = 'active';
      } else {
        status = 'pending';
      }
      return { id: stage.id, label: stage.label, state: status };
    });
  }

  /* What the status region says. The screen reader gets this; the spinner is
   * decoration over the top of it. */
  function label(state) {
    if (state.phase === 'planning') return '規劃中…';
    if (state.phase === 'running') {
      var done = state.branches.length - state.outstanding;
      return '查詢中…（' + done + '/' + state.branches.length + ' 完成）';
    }
    if (state.phase === 'synthesising') return '撰寫回答中…';
    if (state.phase === 'refused') return '沒有可執行的查詢';
    if (state.phase === 'failed') return '已中止';
    return '完成';
  }

  function running(state) {
    return !(state.stopped || state.finished);
  }

  var ROW = 24;
  var GAP = 8;
  var PAD = 12;
  var COLUMNS = [
    { x: 10, w: 76 },   // plan
    { x: 122, w: 208 }, // the branches
    { x: 366, w: 58 },  // join
    { x: 460, w: 104 }  // write
  ];
  var WIDTH = 574;

  function make(doc, name, attrs) {
    var node = doc.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (key) {
      node.setAttribute(key, String(attrs[key]));
    });
    return node;
  }

  function boxWithText(doc, parent, geom, cls, text) {
    parent.appendChild(make(doc, 'rect', {
      x: geom.x, y: geom.y, width: geom.w, height: geom.h, rx: 5,
      'class': 'flow-box ' + cls
    }));
    var caption = make(doc, 'text', {
      x: geom.x + geom.w / 2, y: geom.y + geom.h / 2 + 4,
      'text-anchor': 'middle', 'class': 'flow-text ' + cls
    });
    // The one place a value from the wire lands in the document.
    caption.textContent = text;
    parent.appendChild(caption);
  }

  function connect(doc, parent, x1, y1, x2, y2, cls) {
    parent.appendChild(make(doc, 'path', {
      d: 'M' + x1 + ' ' + y1 + ' C' + ((x1 + x2) / 2) + ' ' + y1 + ',' +
         ((x1 + x2) / 2) + ' ' + y2 + ',' + x2 + ' ' + y2,
      'class': 'flow-link ' + cls
    }));
  }

  function branchText(branch) {
    var mark = branch.state === 'ok' ? '✓' : (branch.state === 'failed' ? '✗' : '·');
    return mark + ' ' + branch.tool;
  }

  /* The <svg> the diagram is drawn into, created rather than sitting in the
   * template. A page with scripting off must contain no chart markup at all --
   * two tests read "<svg is on the page" as "a chart was rendered", and an
   * empty placeholder only the script ever uses would make both of them lie.
   * Idempotent: a second submit redraws into the element the first created. */
  function mount(container) {
    var existing = container.firstChild;
    if (existing) return existing;
    var svg = container.ownerDocument.createElementNS(NS, 'svg');
    container.appendChild(svg);
    return svg;
  }

  /* Redraw the whole diagram. Cheap: a handful of nodes, a few times a run. */
  function render(svg, state) {
    var doc = svg.ownerDocument;
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    var stageList = stages(state);
    var rows = Math.max(1, state.branches.length);
    var bandHeight = rows * ROW + (rows - 1) * GAP;
    var height = bandHeight + PAD * 2;
    var middle = height / 2;
    svg.setAttribute('viewBox', '0 0 ' + WIDTH + ' ' + height);

    var geoms = COLUMNS.map(function (column) {
      return { x: column.x, y: middle - ROW / 2, w: column.w, h: ROW };
    });

    // Plan, then the branch column, then the join, then the answer.
    boxWithText(doc, svg, geoms[0], 'is-' + stageList[0].state, stageList[0].label);

    var fanState = stageList[1].state;
    var branchGeoms = state.branches.map(function (branch, index) {
      return {
        x: geoms[1].x, y: PAD + index * (ROW + GAP), w: geoms[1].w, h: ROW,
        branch: branch
      };
    });
    if (branchGeoms.length === 0) {
      // Two different nothings, and the difference is the whole point of
      // drawing this before the plan arrives: branches not known yet, against
      // branches that will never exist because the model declined.
      var empty = fanState === 'skipped' ? '沒有查詢' : '尚未規劃';
      boxWithText(doc, svg, geoms[1], 'is-' + fanState, empty);
      connect(doc, svg, geoms[0].x + geoms[0].w, middle, geoms[1].x, middle,
        'is-' + fanState);
    } else {
      branchGeoms.forEach(function (geom) {
        var cls = geom.branch.state === 'ok' ? 'is-done'
          : (geom.branch.state === 'failed' ? 'is-failed'
            : (fanState === 'skipped' ? 'is-skipped' : 'is-active'));
        boxWithText(doc, svg, geom, cls, branchText(geom.branch));
        var rowMiddle = geom.y + ROW / 2;
        connect(doc, svg, geoms[0].x + geoms[0].w, middle, geom.x, rowMiddle,
          cls === 'is-active' ? 'is-active' : cls);
        connect(doc, svg, geom.x + geom.w, rowMiddle, geoms[2].x, middle,
          cls === 'is-active' ? 'is-active' : cls);
      });
    }

    boxWithText(doc, svg, geoms[2], 'is-' + stageList[2].state, stageList[2].label);
    connect(doc, svg, geoms[2].x + geoms[2].w, middle, geoms[3].x, middle,
      'is-' + stageList[3].state);
    boxWithText(doc, svg, geoms[3], 'is-' + stageList[3].state, stageList[3].label);
  }

  global.AoiFlow = {
    model: model,
    stages: stages,
    label: label,
    running: running,
    mount: mount,
    render: render
  };
}(typeof globalThis !== 'undefined' ? globalThis : this));
