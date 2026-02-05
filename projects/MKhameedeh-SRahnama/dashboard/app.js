function lineChart(canvas, label) {
  const ctx = canvas.getContext("2d");
  const state = { xs: [], ys: [] };

  function draw() {
    const w = canvas.width = canvas.clientWidth * devicePixelRatio;
    const h = canvas.height = canvas.clientHeight * devicePixelRatio;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#111";
    ctx.font = `${12 * devicePixelRatio}px system-ui`;
    ctx.fillText(label, 10 * devicePixelRatio, 18 * devicePixelRatio);

    if (state.xs.length < 2) return;
    const xmin = state.xs[0], xmax = state.xs[state.xs.length - 1];
    let ymin = Math.min(...state.ys), ymax = Math.max(...state.ys);
    if (ymin === ymax) { ymin -= 1; ymax += 1; }

    const pad = 26 * devicePixelRatio;
    const xscale = (x) => pad + (x - xmin) * (w - 2 * pad) / (xmax - xmin);
    const yscale = (y) => h - pad - (y - ymin) * (h - 2 * pad) / (ymax - ymin);

    ctx.strokeStyle = "#2b6";
    ctx.lineWidth = 2 * devicePixelRatio;
    ctx.beginPath();
    for (let i = 0; i < state.xs.length; i++) {
      const x = xscale(state.xs[i]);
      const y = yscale(state.ys[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function setData(xs, ys) {
    state.xs = xs;
    state.ys = ys;
    draw();
  }

  window.addEventListener("resize", draw);
  return { setData };
}

async function fetchJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return await r.json();
}

async function tick() {
  const data = await fetchJSON("/api/metrics");
  const steps = data.points.map(p => p.step ?? 0);

  const chartLoss = lineChart(document.getElementById("loss"), "loss");
  const chartAcc = lineChart(document.getElementById("acc"), "acc");
  const chartMem = lineChart(document.getElementById("memFree"), "cuda_mem_free_mb");
  const chartStep = lineChart(document.getElementById("stepS"), "step_s");

  function series(key) {
    const xs = [];
    const ys = [];
    for (const p of data.points) {
      if (p[key] === undefined) continue;
      xs.push(p.step ?? xs.length);
      ys.push(p[key]);
    }
    return [xs, ys];
  }

  chartLoss.setData(...series("loss"));
  chartAcc.setData(...series("acc"));
  chartMem.setData(...series("cuda_mem_free_mb"));
  chartStep.setData(...series("step_s"));

  document.getElementById("runDir").textContent = data.run_dir;
}

async function loop() {
  while (true) {
    try {
      await tick();
    } catch (e) {
      console.error(e);
    }
    await new Promise(r => setTimeout(r, 1000));
  }
}

loop();

