async function fetchJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return await r.json();
}

async function postJSON(path, payload) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload ?? {}),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

function pivotTidy(rows) {
  const byT = new Map();
  for (const r of rows) {
    const t = Number(r.t_ms ?? r.t ?? 0);
    const metric = r.metric;
    const value = Number(r.value);
    if (!isFinite(t) || !metric || !isFinite(value)) continue;
    if (!byT.has(t)) byT.set(t, { t_ms: t });
    byT.get(t)[metric] = value;
  }
  return Array.from(byT.values()).sort((a, b) => a.t_ms - b.t_ms);
}

function seriesFromPoints(points, xKey, yKey) {
  const xs = [];
  const ys = [];
  for (const p of points) {
    const x = Number(p[xKey]);
    const y = Number(p[yKey]);
    if (!isFinite(x) || !isFinite(y)) continue;
    xs.push(x);
    ys.push(y);
  }
  return [xs, ys];
}

function seriesFromRowsRelTime(rows, tKey, yKey) {
  const xs = [];
  const ys = [];
  if (!rows.length) return [xs, ys];
  const t0 = Number(rows[0][tKey]);
  for (const r of rows) {
    const t = Number(r[tKey]);
    const y = Number(r[yKey]);
    if (!isFinite(t) || !isFinite(y)) continue;
    xs.push((t - t0) / 1000.0);
    ys.push(y);
  }
  return [xs, ys];
}

function setOptions(selectEl, values, keepValue = true) {
  const prev = selectEl.value;
  selectEl.innerHTML = "";
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    selectEl.appendChild(opt);
  }
  if (keepValue && values.includes(prev)) selectEl.value = prev;
}

function fmtNum(x, digits = 3) {
  const v = Number(x);
  if (!isFinite(v)) return "-";
  return v.toFixed(digits);
}

function getStyle(prop) {
  return getComputedStyle(document.documentElement).getPropertyValue(prop).trim();
}

function plotLine(divId, title, xs, ys, xTitle, yTitle) {
  const div = document.getElementById(divId);
  if (!div) return;
  if (!window.uPlot) {
    div.textContent = "uPlot failed to load.";
    return;
  }

  if (!div.__uplot) {
    const theme = document.documentElement.getAttribute("data-theme") || "dark";
    const isLight = theme === "light";

    // Use CSS variables for chart colors
    const axisColor = getStyle('--muted') || (isLight ? "rgba(17,24,39,.55)" : "rgba(232,236,255,.55)");
    const gridColor = getStyle('--border') || (isLight ? "rgba(17,24,39,.10)" : "rgba(232,236,255,.10)");
    const textColor = getStyle('--text') || (isLight ? "#111827" : "#e8ecff");
    const lineColor = getStyle('--accent') || (isLight ? "#0ea5e9" : "#58d6ff");

    const opts = {
      title,
      width: div.clientWidth || 600,
      height: div.clientHeight || 320,
      padding: [12, 14, 38, 54],
      cursor: { drag: { x: true, y: false } },
      scales: { x: { time: false }, y: { auto: true } },
      axes: [
        { stroke: axisColor, grid: { stroke: gridColor }, label: xTitle, font: "12px Inter", labelFont: "12px Inter", ticks: { stroke: axisColor } },
        { stroke: axisColor, grid: { stroke: gridColor }, label: yTitle, font: "12px Inter", labelFont: "12px Inter", ticks: { stroke: axisColor } },
      ],
      series: [
        { label: xTitle },
        { label: yTitle, stroke: lineColor, width: 2, points: { show: false } },
      ],
    };
    div.__uplot = new window.uPlot(opts, [xs, ys], div);

    const ro = new ResizeObserver(() => {
      const u = div.__uplot;
      if (!u) return;
      const w = div.clientWidth || 600;
      const h = div.clientHeight || 320;
      u.setSize({ width: w, height: h });
    });
    ro.observe(div);
    div.__ro = ro;
  } else {
    div.__uplot.setData([xs, ys], true);
  }
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem("theme", theme);
  } catch { }
}

function initTheme() {
  let theme = "dark";
  try {
    const saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") theme = saved;
  } catch { }
  applyTheme(theme);
  const btn = document.getElementById("themeToggle");
  if (btn) {
    btn.addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme") || "dark";
      applyTheme(cur === "dark" ? "light" : "dark");
      // Force rebuild plots with new theme colors.
      for (const id of ["lossPlot", "accPlot", "stepSPlot", "gpuPowerPlot", "gpuUtilPlot", "gpuMemUsedPlot", "procCpuPlot", "procRssPlot", "occPlot", "paramNormPlot", "gradNormPlot"]) {
        const el = document.getElementById(id);
        if (el && el.__uplot) {
          try { el.__ro && el.__ro.disconnect(); } catch { }
          try { el.__uplot.destroy(); } catch { }
          el.__uplot = null;
          el.__ro = null;
        }
      }
    });
  }
}

function updateFnTable(rows, q) {
  const tbody = document.getElementById("fnTable");
  const query = (q || "").toLowerCase().trim();
  const filtered = rows
    .map((r) => ({
      kind: r.kind ?? "",
      name: r.name ?? "",
      calls: Number(r.calls ?? 0),
      total_ms: Number(r.total_ms ?? 0),
      avg_ms: Number(r.avg_ms ?? 0),
      max_ms: Number(r.max_ms ?? 0),
      total_bytes: Number(r.total_bytes ?? 0),
      avg_gbps: Number(r.avg_gbps ?? 0),
    }))
    .filter((r) => (query ? (r.kind + " " + r.name).toLowerCase().includes(query) : true))
    .sort((a, b) => b.total_ms - a.total_ms)
    .slice(0, 250);

  tbody.innerHTML = "";
  for (const r of filtered) {
    const total_mb = r.total_bytes > 0 ? r.total_bytes / (1024.0 * 1024.0) : 0;
    const total_mb_str = total_mb > 0 ? fmtNum(total_mb, 2) : "-";
    const avg_gbps_str = r.avg_gbps > 0 ? fmtNum(r.avg_gbps, 2) : "-";
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.kind}</td><td style="font-family:monospace; font-size:12px;">${r.name}</td><td>${r.calls}</td><td>${fmtNum(
      r.total_ms,
      2
    )}</td><td>${fmtNum(r.avg_ms, 4)}</td><td>${fmtNum(r.max_ms, 4)}</td><td>${total_mb_str}</td><td>${avg_gbps_str}</td>`;
    tbody.appendChild(tr);
  }
}

const state = {
  runsRoot: "",
  runDir: "",
  kernelName: "",
  fnFilter: "",
  terminalOffset: 0,
};

async function updateTerminal() {
  const terminal = document.getElementById("terminal");
  if (!terminal) return;
  try {
    const data = await fetchJSON(`/api/terminal?offset=${state.terminalOffset}`);
    if (data.logs && data.logs.length > 0) {
      const atBottom = terminal.scrollHeight - terminal.scrollTop <= terminal.clientHeight + 50;
      for (const line of data.logs) {
        const span = document.createElement("span");
        span.textContent = line;
        terminal.appendChild(span);
        // Ensure separation if lines don't end in newline
        if (!line.endsWith('\n')) {
          terminal.appendChild(document.createTextNode('\n'));
        }
      }
      state.terminalOffset = data.offset;
      if (atBottom) {
        terminal.scrollTop = terminal.scrollHeight;
      }
    }
  } catch (e) {
    console.error("Terminal fetch failed", e);
  }
}

async function loadRuns() {
  const data = await fetchJSON("/api/runs");
  state.runsRoot = data.runs_root;
  document.getElementById("runsRoot").textContent = state.runsRoot;
  const runs = (data.runs || []).map((r) => r.run_dir);
  const sel = document.getElementById("runSelect");
  setOptions(sel, runs, true);

  // Only apply server default on first load; preserve user's selection on refreshes
  if (!state.runDir || !runs.includes(state.runDir)) {
    const info = await fetchJSON("/api/run_info");
    const defaultName = (info.run_dir || "").split(/[\\/]/).pop();
    if (defaultName && runs.includes(defaultName)) sel.value = defaultName;
    state.runDir = sel.value || defaultName || runs[0] || "";
  } else {
    sel.value = state.runDir;
  }
}

async function loadRunInfo() {
  if (!state.runDir) return;
  const info = await fetchJSON(`/api/run_info?run_dir=${encodeURIComponent(state.runDir)}`);
  document.getElementById("runDir").textContent = info.run_dir;
  const cfg = info.config || {};
  const dev = info.device || {};

  const tags = [
    { label: 'arch', val: cfg.arch },
    { label: 'dataset', val: cfg.dataset },
    { label: 'batch', val: cfg.batch_size },
    { label: 'lr', val: cfg.lr },
    { label: 'device', val: dev.name },
    { label: 'sm', val: dev.sm_count }
  ];

  const html = tags.filter(t => t.val !== undefined && t.val !== null)
    .map(t => `<div class="badge">${t.label}: <strong>${t.val}</strong></div>`)
    .join('');

  document.getElementById("runInfo").innerHTML = html;

  // Handle speedup
  const benchCard = document.getElementById("benchmarkCard");
  if (info.speedup && info.speedup.raw) {
    benchCard.style.display = "block";
    const match = info.speedup.raw.match(/\((.*)\% faster\)/);
    if (match) {
      document.getElementById("speedupVal").textContent = match[1] + "%";
    } else {
      document.getElementById("speedupVal").textContent = info.speedup.raw.split("BENCHMARK_SPEEDUP")[1].trim();
    }
  } else {
    benchCard.style.display = "none";
  }
}

async function updateProcStatus() {
  const st = await fetchJSON("/api/proc_status");
  const sidebarEl = document.getElementById("procStatus");
  const headerEl = document.getElementById("headerStatus");

  const statusMap = {
    running: { label: `Running (PID ${st.pid})`, cls: 'status-running', icon: 'play' },
    completed: { label: 'Completed', cls: 'status-completed', icon: 'check-circle' },
    stopped: { label: 'Stopped', cls: 'status-stopped', icon: 'square' },
    error: { label: `Error (exit ${st.exit_code})`, cls: 'status-error', icon: 'alert-triangle' },
    idle: { label: 'Idle', cls: 'status-idle', icon: 'circle' },
  };

  const info = statusMap[st.status] || statusMap.idle;

  const badgeHTML = `<span class="status-badge ${info.cls}"><span class="status-dot"></span><i data-lucide="${info.icon}" size="14"></i> ${info.label}</span>`;

  if (sidebarEl) sidebarEl.innerHTML = badgeHTML;
  if (headerEl) headerEl.innerHTML = badgeHTML;

  // Re-render lucide icons in the newly added HTML
  try { lucide.createIcons(); } catch { }
}

async function tick() {
  if (!state.runDir) return;

  await updateProcStatus();
  await updateTerminal();

  const q = `run_dir=${encodeURIComponent(state.runDir)}`;
  const [train, gpu, sys, fns, kern, kmet] = await Promise.all([
    fetchJSON(`/api/csv?file=train_metrics&${q}&tail=6000`),
    fetchJSON(`/api/csv?file=gpu_metrics&${q}&tail=6000`),
    fetchJSON(`/api/csv?file=system_metrics&${q}&tail=6000`),
    fetchJSON(`/api/csv?file=functions_summary&${q}&tail=4000`),
    fetchJSON(`/api/csv?file=kernel_launches&${q}&tail=6000`).catch(() => ({ rows: [] })),
    fetchJSON(`/api/csv?file=kernel_metrics&${q}&tail=6000`).catch(() => ({ rows: [] })),
  ]);

  const trainPts = pivotTidy(train.rows || []);
  const lossPts = trainPts.filter((p) => p.loss !== undefined && p.step !== undefined);
  const accPts = trainPts.filter((p) => p.acc !== undefined && p.step !== undefined);
  const stepPts = trainPts.filter((p) => p.step_s !== undefined && p.step !== undefined);
  const pNormPts = trainPts.filter((p) => p.param_l2 !== undefined && p.step !== undefined);
  const gNormPts = trainPts.filter((p) => p.grad_l2 !== undefined && p.step !== undefined);

  plotLine("lossPlot", "Loss", ...seriesFromPoints(lossPts, "step", "loss"), "step", "loss");
  plotLine("accPlot", "Accuracy", ...seriesFromPoints(accPts, "step", "acc"), "step", "acc");
  plotLine("stepSPlot", "Step Time", ...seriesFromPoints(stepPts, "step", "step_s"), "step", "s");
  plotLine("paramNormPlot", "Parameter L2 Norm", ...seriesFromPoints(pNormPts, "step", "param_l2"), "step", "L2");
  plotLine("gradNormPlot", "Gradient L2 Norm", ...seriesFromPoints(gNormPts, "step", "grad_l2"), "step", "L2");

  const gpuRows = (gpu.rows || []).map((r) => ({
    t_ms: Number(r.t_ms),
    nvml_power_w: Number(r.nvml_power_w),
    nvml_util_gpu_pct: Number(r.nvml_util_gpu_pct),
    cuda_mem_used_mb: Number(r.cuda_mem_used_mb),
  }));
  plotLine("gpuPowerPlot", "GPU Power", ...seriesFromRowsRelTime(gpuRows, "t_ms", "nvml_power_w"), "t_s", "W");
  plotLine("gpuUtilPlot", "GPU Utilization", ...seriesFromRowsRelTime(gpuRows, "t_ms", "nvml_util_gpu_pct"), "t_s", "%");
  plotLine("gpuMemUsedPlot", "GPU Memory Used", ...seriesFromRowsRelTime(gpuRows, "t_ms", "cuda_mem_used_mb"), "t_s", "MB");

  const sysRows = (sys.rows || []).map((r) => ({
    t_ms: Number(r.t_ms),
    process_cpu_pct: Number(r.process_cpu_pct),
    process_rss_mb: Number(r.process_rss_mb),
  }));
  plotLine("procCpuPlot", "Process CPU", ...seriesFromRowsRelTime(sysRows, "t_ms", "process_cpu_pct"), "t_s", "%");
  plotLine("procRssPlot", "Process RSS", ...seriesFromRowsRelTime(sysRows, "t_ms", "process_rss_mb"), "t_s", "MB");

  updateFnTable(fns.rows || [], state.fnFilter);

  const kernRows = (kern.rows || []).map((r) => ({
    t_ms: Number(r.t_ms),
    name: r.name ?? "",
    occupancy_pct: Number(r.occupancy_pct),
  }));
  const kmetRows = (kmet.rows || []).map((r) => ({
    name: r.name ?? "",
    achieved_occupancy_pct: Number(r.achieved_occupancy_pct ?? r.achieved_occupancy ?? r.occupancy_pct),
  }));
  const kmetMap = new Map();
  for (const r of kmetRows) {
    if (r.name && isFinite(r.achieved_occupancy_pct)) {
      kmetMap.set(r.name, r.achieved_occupancy_pct);
    }
  }
  const kernelNames = Array.from(
    new Set([...kernRows.map((r) => r.name), ...kmetRows.map((r) => r.name)].filter(Boolean))
  ).sort();
  const kernelSel = document.getElementById("kernelSelect");
  if (kernelNames.length) {
    if (!state.kernelName) state.kernelName = kernelNames[0];
    setOptions(kernelSel, kernelNames, true);
    if (!kernelSel.value) kernelSel.value = state.kernelName;
    state.kernelName = kernelSel.value;
    const occPts = kernRows.filter((r) => r.name === state.kernelName);
    const achieved = kmetMap.get(state.kernelName);
    if (occPts.length && isFinite(achieved)) {
      const constPts = occPts.map((r) => ({ t_ms: r.t_ms, occupancy_pct: achieved }));
      plotLine(
        "occPlot",
        `Achieved Occupancy: ${state.kernelName}`,
        ...seriesFromRowsRelTime(constPts, "t_ms", "occupancy_pct"),
        "t_s",
        "%"
      );
    } else {
      plotLine(
        "occPlot",
        `Occupancy: ${state.kernelName}`,
        ...seriesFromRowsRelTime(occPts, "t_ms", "occupancy_pct"),
        "t_s",
        "%"
      );
    }
  } else {
    plotLine("occPlot", "Kernel Occupancy", [], [], "t_ms", "%");
  }

  await loadRunInfo();
}

async function wireLaunchControls() {
  const btnStart = document.getElementById("btnStart");
  const btnStop = document.getElementById("btnStop");
  const sel = document.getElementById("runSelect");
  if (btnStart) {
    btnStart.addEventListener("click", async () => {
      console.log("Start button clicked. Collecting payload...");
      let payload = {};
      try {
        const getVal = (id, def) => {
          const el = document.getElementById(id);
          if (!el) console.warn(`Missing element: ${id}`);
          return (el ? el.value : "") || def;
        };
        const getNum = (id, def) => Number(getVal(id, def));
        const getCheck = (id) => {
          const el = document.getElementById(id);
          return el ? el.checked : false;
        };

        payload = {
          exe: getVal("cfgExe", ""),
          run_name: getVal("cfgRunName", ""),
          arch: getVal("cfgArch", "lenet"),
          dataset: getVal("cfgDataset", "mnist"),
          epochs: getNum("cfgEpochs", 2),
          batch: getNum("cfgBatch", 64),
          lr: getNum("cfgLr", 0.01),
          weight_decay: getNum("cfgWeightDecay", 0.0),
          log_every: getNum("cfgLogEvery", 50),
          eval_every: getNum("cfgEvalEvery", 200),
          max_steps: getNum("cfgMaxSteps", 0),
          norm_log_multiplier: getNum("cfgNormMult", 5),
          profile_interval_ms: getNum("cfgProfileInterval", 200),

          enable_h2d_pipeline: getCheck("cfgH2DPipeline"),
          enable_log_sync_optimizations: getCheck("cfgLogSync"),
          enable_async_checkpoint: getCheck("cfgAsyncCkpt"),
          enable_cuda_graph_sgd: getCheck("cfgCudaGraph"),
          enable_async_eval: getCheck("cfgAsyncEval"),
          shuffle_train: !getCheck("cfgNoShuffle"),

          benchmark_compare: getCheck("cfgBenchmark"),
          benchmark_steps: getNum("cfgBenchSteps", 200),
        };
      } catch (err) {
        console.error("Failed to collect payload", err);
        alert("Configuration Error: " + err.message);
        return;
      }

      console.log("Sending payload:", payload);

      // Clear terminal for new run
      const terminal = document.getElementById("terminal");
      if (terminal) terminal.innerHTML = "";
      state.terminalOffset = 0;

      try {
        const btn = document.getElementById("btnStart");
        const originalText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = "Starting...";

        const r = await postJSON("/api/start", payload);
        console.log("Start response:", r);

        await loadRuns();
        if (r.run_dir) {
          sel.value = r.run_dir;
          state.runDir = r.run_dir;
          await loadRunInfo();
        }
      } catch (e) {
        console.error("Start failed", e);
        alert("Launch Failed: " + String(e));
      } finally {
        const btn = document.getElementById("btnStart");
        btn.disabled = false;
        btn.innerHTML = '<i data-lucide="play" size="16"></i> Start';
        lucide.createIcons();
      }
    });
  }
  if (btnStop) {
    btnStop.addEventListener("click", async () => {
      try {
        await postJSON("/api/stop", {});
      } catch (e) {
        console.error(e);
      }
    });
  }
}

async function main() {
  window.addEventListener("error", (e) => {
    console.error("Global JS Error:", e.message, e.filename, e.lineno);
  });
  window.addEventListener("unhandledrejection", (e) => {
    console.error("Unhandled Promise Rejection:", e.reason);
  });
  initTheme();
  await loadRuns();

  const sel = document.getElementById("runSelect");
  sel.addEventListener("change", () => {
    state.runDir = sel.value;
  });
  document.getElementById("kernelSelect").addEventListener("change", (e) => {
    state.kernelName = e.target.value;
  });
  document.getElementById("fnFilter").addEventListener("input", (e) => {
    state.fnFilter = e.target.value || "";
  });
  const btnRefresh = document.getElementById("btnRefresh");
  if (btnRefresh) {
    btnRefresh.addEventListener("click", async () => {
      const btn = document.getElementById("btnRefresh");
      btn.disabled = true;
      try {
        await loadRuns();
        await tick();
      } catch (e) {
        console.error("Manual refresh failed", e);
      } finally {
        btn.disabled = false;
      }
    });
  }

  await wireLaunchControls();

  let tickCounter = 0;
  while (true) {
    const autoRefresh = document.getElementById("cfgAutoRefresh")?.checked ?? true;
    if (autoRefresh) {
      try {
        await tick();

        // Refresh run list every 10 ticks (approx 10s)
        tickCounter++;
        if (tickCounter >= 10) {
          await loadRuns();
          tickCounter = 0;
        }
      } catch (e) {
        console.error(e);
      }
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
}

main();
