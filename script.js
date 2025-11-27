(function () {
  const container = document.getElementById('canvas-container');

  const width = 760;
  const height = 400;

  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Simple neural network diagram with two inputs, hidden layer, and one output');
  svg.setAttribute('aria-label', 'Linear regression diagram with bias and one input');

  // Layout positions
  const colX = {
    input: 160,
    output: 600,
  };
  const rowY = {
    input: [140, 260], // two inputs
    output: [200], // one output
  };

  // Helper to add a line
  function addLine(x1, y1, x2, y2) {
    const line = document.createElementNS(svgNS, 'line');
    line.setAttribute('x1', x1);
    line.setAttribute('y1', y1);
    line.setAttribute('x2', x2);
    line.setAttribute('y2', y2);
    line.setAttribute('class', 'line');
    svg.appendChild(line);
    return line;
  }

  // Helper to add a label near the midpoint of a connection
  function addWeightLabel(x1, y1, x2, y2, textValue, offset = { x: 0, y: -14 }) {
    const mx = (x1 + x2) / 2 + offset.x;
    const my = (y1 + y2) / 2 + offset.y;
    const t = document.createElementNS(svgNS, 'text');
    t.setAttribute('x', mx);
    t.setAttribute('y', my);
    t.setAttribute('class', 'weight-label');
    t.textContent = textValue;
    svg.appendChild(t);
    return t;
  }

  // Helper to add a circle with a label
  function addNode(x, y, label, sublabel, options = {}) {
    const g = document.createElementNS(svgNS, 'g');

    const circle = document.createElementNS(svgNS, 'circle');
    circle.setAttribute('cx', x);
    circle.setAttribute('cy', y);
    circle.setAttribute('r', 26);
    circle.setAttribute('class', 'circle');
    if (options.className) {
      circle.setAttribute('class', `circle ${options.className}`);
    }

    const text = document.createElementNS(svgNS, 'text');
    text.setAttribute('x', x);
    text.setAttribute('y', y);
    text.setAttribute('class', 'label');
    text.textContent = label;

    g.appendChild(circle);
    g.appendChild(text);

    if (sublabel) {
      const sub = document.createElementNS(svgNS, 'text');
      sub.setAttribute('x', x);
      sub.setAttribute('y', y + 38);
      sub.setAttribute('class', 'small-label');
      sub.textContent = sublabel;
      g.appendChild(sub);
    }

    svg.appendChild(g);
    return g;
  }

  // Draw input nodes (top is bias = 1, bottom is x1)
  const inputs = [
    addNode(colX.input, rowY.input[0], '1', 'bias', { className: 'circle--bias' }),
    addNode(colX.input, rowY.input[1], 'x1', 'input'),
  ];

  // Draw output node
  const outputs = rowY.output.map((y) => addNode(colX.output, y, 'y', 'output'));

  // Connect inputs directly to output (linear regression) with initial weights
  const state = {
    w0: 0.5,
    w1: 1.2,
  };
  // bias (top) → output
  const biasLine = addLine(colX.input + 26, rowY.input[0], colX.output - 26, rowY.output[0]);
  const biasLabel = addWeightLabel(colX.input + 26, rowY.input[0], colX.output - 26, rowY.output[0], `w0=${state.w0}`);
  // x1 (bottom) → output
  const x1Line = addLine(colX.input + 26, rowY.input[1], colX.output - 26, rowY.output[0]);
  const x1Label = addWeightLabel(colX.input + 26, rowY.input[1], colX.output - 26, rowY.output[0], `w1=${state.w1}`, { x: 0, y: 16 });

  // Layer titles
  function addTitle(x, y, textValue) {
    const t = document.createElementNS(svgNS, 'text');
    t.setAttribute('x', x);
    t.setAttribute('y', y);
    t.setAttribute('class', 'small-label');
    t.textContent = textValue;
    svg.appendChild(t);
  }

  addTitle(colX.input, 60, 'Inputs');
  addTitle(colX.output, 60, 'Output (ŷ)');
  const eqn = addTitle((colX.input + colX.output) / 2, 360, 'ŷ = w0·1 + w1·x1');

  // Exercise logic: hook inputs and buttons
  const inpX1 = document.getElementById('inp-x1');
  const inpY = document.getElementById('inp-y');
  const inpAlpha = document.getElementById('inp-alpha');
  const btnUpdate = document.getElementById('btn-update');
  const btnReset = document.getElementById('btn-reset');
  const statusYhat = document.getElementById('status-yhat');
  const statusLoss = document.getElementById('status-loss');

  function computeYhat(x1) {
    return state.w0 * 1 + state.w1 * x1;
  }
  function computeLoss(yhat, y) {
    const e = yhat - y;
    return 0.5 * e * e;
  }
  function refreshLabels() {
    biasLabel.textContent = `w0=${round2(state.w0)}`;
    x1Label.textContent = `w1=${round2(state.w1)}`;
  }
  function round2(v) { return Math.round(v * 100) / 100; }

  function updateStatus(x1, y) {
    const yhat = computeYhat(x1);
    const loss = computeLoss(yhat, y);
    statusYhat.textContent = `ŷ: ${round2(yhat)}`;
    statusLoss.textContent = `L: ${round2(loss)}`;
  }

  function doUpdate() {
    const x1 = parseFloat(inpX1.value);
    const y = parseFloat(inpY.value);
    const alpha = parseFloat(inpAlpha.value);
    if (!isFinite(x1) || !isFinite(y) || !isFinite(alpha)) return;
    const yhat = computeYhat(x1);
    const error = yhat - y;
    const dL_dw0 = error * 1;
    const dL_dw1 = error * x1;
    state.w0 = state.w0 - alpha * dL_dw0;
    state.w1 = state.w1 - alpha * dL_dw1;
    refreshLabels();
    updateStatus(x1, y);
  }

  function doReset() {
    state.w0 = 0.5;
    state.w1 = 1.2;
    refreshLabels();
    const x1 = parseFloat(inpX1.value);
    const y = parseFloat(inpY.value);
    updateStatus(x1, y);
  }

  btnUpdate.addEventListener('click', doUpdate);
  btnReset.addEventListener('click', doReset);
  // Initial status
  doReset();

  // Batch exercise logic
  const batchAlpha = document.getElementById('batch-alpha');
  const batchUpdateBtn = document.getElementById('batch-update');
  const batchResetBtn = document.getElementById('batch-reset');
  const batchWSpan = document.getElementById('batch-w');
  const batchLossSpan = document.getElementById('batch-loss');

  // Dataset (n samples)
  const Xdata = [
    [1, 0.2],
    [1, 0.5],
    [1, 0.9],
    [1, 1.4],
    [1, 2.0],
  ];
  const Ydata = [0.7, 1.2, 1.9, 2.6, 3.5];

  const batchState = {
    w0: 0.5,
    w1: 1.2,
  };

  function batchPredictRow(row) {
    return batchState.w0 * row[0] + batchState.w1 * row[1];
  }
  function batchForward() {
    return Xdata.map(r => batchPredictRow(r));
  }
  function batchLoss(preds) {
    const n = preds.length;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const e = preds[i] - Ydata[i];
      sum += e * e;
    }
    return 0.5 * sum / n;
  }
  function batchGradient(preds) {
    const n = preds.length;
    let g0 = 0;
    let g1 = 0;
    for (let i = 0; i < n; i++) {
      const e = preds[i] - Ydata[i];
      g0 += e * 1;      // bias column
      g1 += e * Xdata[i][1];
    }
    return { g0: g0 / n, g1: g1 / n };
  }
  function batchRefresh() {
    const preds = batchForward();
    const loss = batchLoss(preds);
    batchWSpan.textContent = `w = [${round2(batchState.w0)}, ${round2(batchState.w1)}]`;
    batchLossSpan.textContent = `L = ${round2(loss)}`;
  }
  function batchDoUpdate() {
    const alpha = parseFloat(batchAlpha.value);
    if (!isFinite(alpha)) return;
    const preds = batchForward();
    const grad = batchGradient(preds);
    batchState.w0 -= alpha * grad.g0;
    batchState.w1 -= alpha * grad.g1;
    batchRefresh();
  }
  function batchDoReset() {
    batchState.w0 = 0.5;
    batchState.w1 = 1.2;
    batchRefresh();
  }
  if (batchUpdateBtn && batchResetBtn) {
    batchUpdateBtn.addEventListener('click', batchDoUpdate);
    batchResetBtn.addEventListener('click', batchDoReset);
    batchDoReset();
  }

  // Copy buttons
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-copy-target');
      const pre = document.getElementById(targetId);
      if (!pre) return;
      const text = pre.textContent;
      navigator.clipboard.writeText(text).then(() => {
        const orig = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = orig; }, 1200);
      }).catch(() => {
        btn.textContent = 'Copy failed';
        setTimeout(() => { btn.textContent = 'Copy'; }, 1200);
      });
    });
  });

  // Epoch exercise logic (reuses Xdata/Ydata structure but separate state)
  const epochsAlpha = document.getElementById('epochs-alpha');
  const epochsCount = document.getElementById('epochs-count');
  const epochsRunBtn = document.getElementById('epochs-run');
  const epochsResetBtn = document.getElementById('epochs-reset');
  const epochsWSpan = document.getElementById('epochs-w');
  const epochsLossSpan = document.getElementById('epochs-loss');

  const epochsState = { w0: 0.5, w1: 1.2 };

  function epochsForward(w0, w1) {
    return Xdata.map(r => w0 * r[0] + w1 * r[1]);
  }
  function epochsLoss(preds) {
    const n = preds.length;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const e = preds[i] - Ydata[i];
      sum += e * e;
    }
    return 0.5 * sum / n;
  }
  function epochsGrad(preds, w0, w1) {
    const n = preds.length;
    let g0 = 0, g1 = 0;
    for (let i = 0; i < n; i++) {
      const e = preds[i] - Ydata[i];
      g0 += e * 1;
      g1 += e * Xdata[i][1];
    }
    return { g0: g0 / n, g1: g1 / n };
  }
  function epochsRefresh() {
    const preds = epochsForward(epochsState.w0, epochsState.w1);
    const loss = epochsLoss(preds);
    epochsWSpan.textContent = `w: [${round2(epochsState.w0)}, ${round2(epochsState.w1)}]`;
    epochsLossSpan.textContent = `L: ${round2(loss)}`;
  }
  function runEpochs(alpha, count) {
    for (let ep = 0; ep < count; ep++) {
      const preds = epochsForward(epochsState.w0, epochsState.w1);
      const grad = epochsGrad(preds, epochsState.w0, epochsState.w1);
      epochsState.w0 -= alpha * grad.g0;
      epochsState.w1 -= alpha * grad.g1;
    }
    epochsRefresh();
  }
  function epochsDoReset() {
    epochsState.w0 = 0.5;
    epochsState.w1 = 1.2;
    epochsRefresh();
  }
  if (epochsRunBtn && epochsResetBtn) {
    epochsRunBtn.addEventListener('click', () => {
      const alpha = parseFloat(epochsAlpha.value);
      const count = parseInt(epochsCount.value, 10);
      if (!isFinite(alpha) || !isFinite(count) || count < 1) return;
      epochsDoReset(); // start from initial each reveal
      runEpochs(alpha, count);
    });
    epochsResetBtn.addEventListener('click', epochsDoReset);
    epochsDoReset();
  }

  // Append SVG
  container.appendChild(svg);
})();
