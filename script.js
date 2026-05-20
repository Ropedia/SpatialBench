const data = window.SPATIALBENCH_DATA;

const lowerIsBetter = new Set(["avgAbsRel", "mediumAbsRel", "denseAbsRel", "time"]);
const columns = [
  "method",
  "paradigm",
  "params",
  "time",
  "singleAbsRel",
  "sparseAbsRel",
  "sparseAuc30",
  "mediumAbsRel",
  "mediumAuc30",
  "mediumAte",
  "mediumFScore",
  "denseAbsRel",
  "denseAuc30",
  "denseAte",
  "denseFScore",
  "avgAbsRel",
  "avgAuc30",
  "avgAte",
  "avgFScore"
];

const labels = {
  avgAuc30: "Avg AUC@30",
  avgAbsRel: "Avg AbsRel",
  avgFScore: "Avg F-Score",
  mediumAuc30: "Medium AUC@30",
  denseAuc30: "Dense AUC@30",
  time: "Time"
};

const statsGrid = document.querySelector("#statsGrid");
const paradigmFilter = document.querySelector("#paradigmFilter");
const sortMetric = document.querySelector("#sortMetric");
const searchInput = document.querySelector("#searchInput");
const leaderboardBody = document.querySelector("#leaderboardTable tbody");
const topModels = document.querySelector("#topModels");
const datasetBody = document.querySelector("#datasetTable tbody");

function isNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function cleanNumber(value) {
  return isNumber(value) ? value : null;
}

function decimalPlaces(value) {
  if (!isNumber(value)) return 0;
  const text = String(value);
  if (!text.includes(".")) return 0;
  return text.split(".")[1].length;
}

const leaderboardDecimalPlaces = Object.fromEntries(columns.map((key) => [
  key,
  Math.max(0, ...data.leaderboard.map((row) => decimalPlaces(row[key])))
]));

function formatValue(value, row, key) {
  if (value === "OOM" || value === "T.O") {
    return `<span class="bad-cell">${value}</span>`;
  }
  if (!isNumber(value)) {
    return value ?? "--";
  }
  const fixed = key === "params" || key === "time" || key.includes("Ate") ? 2 : 3;
  const text = value.toFixed(fixed).replace(/\.?0+$/, "");
  if (row?.incompleteDense && key.startsWith("avg")) {
    return `<span class="incomplete">(${text})</span>`;
  }
  return text;
}

function formatLeaderboardValue(value, row, key) {
  if (value === "OOM" || value === "T.O") {
    return `<span class="bad-cell">${value}</span>`;
  }
  if (!isNumber(value)) {
    return value ?? "--";
  }
  const text = value.toFixed(leaderboardDecimalPlaces[key] ?? 0);
  if (row?.incompleteDense && key.startsWith("avg")) {
    return `<span class="incomplete">(${text})</span>`;
  }
  return text;
}

function slug(text) {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

function renderStats() {
  statsGrid.innerHTML = data.stats.map((item) => `
    <article class="stat">
      <strong>${item.value}</strong>
      <b>${item.label}</b>
      <span>${item.detail}</span>
    </article>
  `).join("");
}

function setupFilters() {
  const paradigms = ["All paradigms", ...new Set(data.leaderboard.map((row) => row.paradigm))];
  paradigmFilter.innerHTML = paradigms.map((name) => `<option value="${name}">${name}</option>`).join("");
}

function sortedRows() {
  const q = searchInput.value.trim().toLowerCase();
  const selected = paradigmFilter.value;
  const metric = sortMetric.value;
  const direction = lowerIsBetter.has(metric) ? 1 : -1;

  return data.leaderboard
    .filter((row) => selected === "All paradigms" || row.paradigm === selected)
    .filter((row) => !q || row.method.toLowerCase().includes(q))
    .sort((a, b) => {
      const av = cleanNumber(a[metric]);
      const bv = cleanNumber(b[metric]);
      if (av === null && bv === null) return a.method.localeCompare(b.method);
      if (av === null) return 1;
      if (bv === null) return -1;
      return (av - bv) * direction;
    });
}

function renderTopCards(rows) {
  const metric = sortMetric.value;
  const usable = rows.filter((row) => isNumber(row[metric])).slice(0, 3);
  const values = usable.map((row) => row[metric]);
  const max = Math.max(...values, 0.001);
  const min = Math.min(...values, 0);

  topModels.innerHTML = usable.map((row, index) => {
    const raw = row[metric];
    const width = lowerIsBetter.has(metric)
      ? Math.max(10, ((max - raw) / Math.max(max - min, 0.001)) * 90 + 10)
      : Math.max(10, (raw / max) * 100);
    return `
      <article class="top-card">
        <span class="rank">Rank ${index + 1}</span>
        <strong>${row.method}</strong>
        <span class="tag ${slug(row.paradigm)}">${row.paradigm}</span>
        <p>${labels[metric]}: <b>${formatValue(raw, row, metric)}</b></p>
        <div class="bar" aria-hidden="true"><span style="width:${width}%"></span></div>
      </article>
    `;
  }).join("");
}

function renderLeaderboard() {
  const rows = sortedRows();
  renderTopCards(rows);
  leaderboardBody.innerHTML = rows.map((row) => {
    const cells = columns.map((key) => {
      if (key === "method") {
        const badge = row.ours ? ' <span class="tag slam-based">Ours</span>' : "";
        return `<td class="method-name sticky-col">${row.method}${badge}</td>`;
      }
      if (key === "paradigm") {
        return `<td><span class="tag ${slug(row.paradigm)}">${row.paradigm}</span></td>`;
      }
      return `<td>${formatLeaderboardValue(row[key], row, key)}</td>`;
    }).join("");
    return `<tr class="${row.ours ? "ours-row" : ""}">${cells}</tr>`;
  }).join("");
}

const datasetTagFamilies = ["env", "dyn", "view", "src"];

function datasetTagClass(family, value) {
  const key = String(value).toLowerCase();
  return `tag tag-${family}-${slug(key)}`;
}

function renderDensityCount(value) {
  if (value === "--" || value === null || value === undefined) {
    return `<span class="density-cell empty">&mdash;</span>`;
  }
  return `<span class="density-cell">${value}</span>`;
}

function renderDatasets() {
  datasetBody.innerHTML = data.datasets.map((row) => {
    const [name, env, dyn, view, src, single, sparse, medium, dense, frames] = row;
    const tagCells = [env, dyn, view, src].map((v, i) =>
      `<td><span class="${datasetTagClass(datasetTagFamilies[i], v)}">${v}</span></td>`
    ).join("");
    const densityCells = [single, sparse, medium, dense].map((v) => `<td>${renderDensityCount(v)}</td>`).join("");
    return `
      <tr>
        <td class="dataset-name">${name}</td>
        ${tagCells}
        ${densityCells}
        <td class="dataset-frames">${frames}</td>
      </tr>
    `;
  }).join("");
}

[paradigmFilter, sortMetric, searchInput].forEach((el) => {
  el.addEventListener("input", renderLeaderboard);
});

function renderModelChart(targetId, valueKey, { ascending = true, scale = "linear", unit = "" } = {}) {
  const target = document.querySelector(`#${targetId}`);
  if (!target) return;
  const rows = data.leaderboard
    .filter((row) => isNumber(row[valueKey]))
    .slice()
    .sort((a, b) => {
      const diff = a[valueKey] - b[valueKey];
      return ascending ? diff : -diff;
    });
  if (!rows.length) {
    target.innerHTML = "";
    return;
  }
  const values = rows.map((row) => row[valueKey]);
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const project = (v) => {
    if (scale === "log") {
      const lo = Math.log10(Math.max(minVal, 1e-3));
      const hi = Math.log10(Math.max(maxVal, lo + 1e-3));
      return ((Math.log10(Math.max(v, 1e-3)) - lo) / Math.max(hi - lo, 1e-6)) * 100;
    }
    return (v / Math.max(maxVal, 1e-6)) * 100;
  };
  const fixed = valueKey === "time" ? 2 : (rows[0][valueKey] >= 100 ? 0 : 1);
  target.innerHTML = rows.map((row) => {
    const pct = Math.max(2, project(row[valueKey])); // floor so tiny bars stay visible
    const valueText = row[valueKey].toFixed(fixed) + unit;
    const oursClass = row.ours ? " is-ours" : "";
    return `
      <li class="chart-row${oursClass}" title="${row.method} — ${row.paradigm}: ${valueText}">
        <span class="chart-name">${row.method}${row.ours ? ' <span class="chart-ours-tag">Ours</span>' : ""}</span>
        <span class="chart-bar-track" aria-hidden="true">
          <span class="chart-bar-fill ${slug(row.paradigm)}" style="width:${pct.toFixed(2)}%"></span>
        </span>
        <span class="chart-value">${valueText}</span>
      </li>
    `;
  }).join("");
}

function renderModelCharts() {
  renderModelChart("paramsChart", "params", { ascending: true, scale: "linear", unit: " M" });
  renderModelChart("timeChart",   "time",   { ascending: true, scale: "log",    unit: " s" });
}

renderStats();
setupFilters();
renderLeaderboard();
renderModelCharts();
renderDatasets();

// ===== Per-dataset detail results =====
const datasetResultsPicker = document.querySelector("#datasetResultsPicker");
const datasetResultsContainer = document.querySelector("#datasetResultsContainer");
const datasetResultsTable = document.querySelector("#datasetResultsTable");
const datasetResultsEmpty = document.querySelector("#datasetResultsEmpty");
let datasetResultsData = null;

const METRIC_ARROW = { AbsRel: "↓", ATE: "↓", "AUC@30": "↑", "F-Score": "↑" };

function detailGroupHeader(layout) {
  const groups = [];
  let last = null;
  for (const col of layout) {
    if (col.regime !== last) {
      groups.push({ name: col.regime, span: 1 });
      last = col.regime;
    } else {
      groups[groups.length - 1].span += 1;
    }
  }
  return groups;
}

function fmtNumber(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return String(value);
  return value.toFixed(3).replace(/\.?0+$/, "");
}

function detailCell(cell) {
  if (cell.oom) {
    return `<td><span class="bad-cell">${cell.value}</span></td>`;
  }
  if (cell.value === null || cell.value === undefined) {
    return `<td class="muted">&mdash;</td>`;
  }
  const num = fmtNumber(cell.value);
  const text = cell.incomplete ? `(${num})` : num;
  const classes = [];
  if (cell.rank) classes.push(cell.rank);
  if (cell.bold) classes.push("bold");
  if (cell.incomplete) classes.push("incomplete");
  const cls = classes.length ? ` class="${classes.join(" ")}"` : "";
  return `<td${cls}>${text}</td>`;
}

function renderDatasetResults(allData) {
  const key = datasetResultsPicker.value;
  if (!key || !allData[key]) {
    datasetResultsContainer.hidden = true;
    datasetResultsEmpty.hidden = false;
    return;
  }
  const ds = allData[key];
  const groups = detailGroupHeader(ds.layout);
  const groupHeader = `<tr>
    <th rowspan="2" class="sticky-col">Method</th>
    <th rowspan="2">Params (M)</th>
    ${groups.map((g) => `<th colspan="${g.span}" class="detail-group">${g.name}</th>`).join("")}
  </tr>`;
  const metricHeader = `<tr>${ds.layout
    .map((col) => `<th class="detail-metric">${col.metric}<span>${METRIC_ARROW[col.metric] ?? ""}</span></th>`)
    .join("")}</tr>`;

  const ncols = 2 + ds.layout.length;
  const body = [];
  let lastCat = null;
  for (const row of ds.rows) {
    if (row.category && row.category !== lastCat) {
      body.push(`<tr class="detail-cat"><td colspan="${ncols}">${row.category}</td></tr>`);
      lastCat = row.category;
    }
    const params = typeof row.params === "number" ? row.params.toFixed(1) : (row.params ?? "");
    const oursBadge = row.ours ? ' <span class="tag slam-based">Ours</span>' : "";
    const cells = row.cells.map(detailCell).join("");
    body.push(`<tr class="${row.ours ? "ours-row" : ""}">
      <td class="method-name sticky-col">${row.method}${oursBadge}</td>
      <td>${params}</td>
      ${cells}
    </tr>`);
  }
  datasetResultsTable.innerHTML = `<thead>${groupHeader}${metricHeader}</thead><tbody>${body.join("")}</tbody>`;
  datasetResultsContainer.hidden = false;
  datasetResultsEmpty.hidden = true;
}

fetch("data/dataset_results.json")
  .then((r) => {
    if (!r.ok) throw new Error(`dataset_results.json fetch failed (${r.status})`);
    return r.json();
  })
  .then((all) => {
    datasetResultsData = all;
    const entries = Object.entries(all).sort((a, b) => a[1].name.localeCompare(b[1].name));
    datasetResultsPicker.innerHTML = [
      `<option value="">Select a dataset…</option>`,
      ...entries.map(([k, d]) => `<option value="${k}">${d.name}</option>`)
    ].join("");
    datasetResultsPicker.addEventListener("change", () => renderDatasetResults(all));
    renderTagLeaderboard();
  })
  .catch((err) => {
    datasetResultsEmpty.textContent = `Could not load per-dataset results: ${err.message}`;
  });

// ===== Scene viewer =====
// GLBs are hosted on Hugging Face Datasets so we don't bundle ~1 GB into the GitHub repo.
// CORS-enabled; cache-control: public, max-age=31536000 (browser caches aggressively).
const GLB_BASE_URL = "https://huggingface.co/datasets/HarrisonPENG/SpatialBenchGLBs/resolve/main/";
const MANIFEST_URL = "data/glb_manifest.json";

const viewerFilters = {
  search: document.querySelector("#viewerSearch"),
  density: document.querySelector("#viewerDensity"),
  dataset: document.querySelector("#viewerDataset"),
  environment: document.querySelector("#viewerEnvironment"),
  dynamics: document.querySelector("#viewerDynamics"),
  viewType: document.querySelector("#viewerViewType"),
  dataType: document.querySelector("#viewerDataType")
};
const viewerListEl = document.querySelector("#viewerSceneList");
const viewerCountEl = document.querySelector("#viewerCount");
const viewerResetBtn = document.querySelector("#viewerReset");
const viewerStageTitle = document.querySelector("#viewerStageTitle");
const viewerStageEyebrow = document.querySelector("#viewerStageEyebrow");
const viewerDownload = document.querySelector("#viewerDownload");
const viewerEmpty = document.querySelector("#viewerEmpty");
const viewerStatus = document.querySelector("#viewerStatus");
const viewerMeta = document.querySelector("#viewerMeta");
const sceneViewer = document.querySelector("#sceneViewer");

const densityOrder = ["single", "sparse", "medium", "dense"];
let viewerScenes = [];
let activeScene = null;

function titleCase(text) {
  return text.replace(/[_-]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function uniqueValues(rows, key) {
  return [...new Set(rows.map((row) => key.includes(".") ? key.split(".").reduce((acc, k) => acc?.[k], row) : row[key]))].filter(Boolean);
}

function populateSelect(select, label, values, sorter) {
  const sorted = sorter ? [...values].sort(sorter) : [...values].sort();
  select.innerHTML = [`<option value="">${label}</option>`, ...sorted.map((v) => `<option value="${v}">${titleCase(v)}</option>`)].join("");
}

// ===== Scene-tag leaderboard =====
const TAG_SCENES_URL = "all_scenes.json";
const tagRankFilters = {
  density: document.querySelector("#tagRankDensity"),
  environment: document.querySelector("#tagRankEnvironment"),
  dynamics: document.querySelector("#tagRankDynamics"),
  viewType: document.querySelector("#tagRankViewType"),
  dataType: document.querySelector("#tagRankDataType"),
  sort: document.querySelector("#tagRankSort")
};
const tagLeaderboardTableBody = document.querySelector("#tagLeaderboardTable tbody");
const tagLeaderboardContainer = document.querySelector("#tagLeaderboardContainer");
const tagLeaderboardEmpty = document.querySelector("#tagLeaderboardEmpty");
const tagLeaderboardCount = document.querySelector("#tagLeaderboardCount");
const tagLeaderboardReset = document.querySelector("#tagLeaderboardReset");
const tagRankMetrics = ["AbsRel", "AUC@30", "ATE", "F-Score"];
const tagLowerIsBetter = new Set(["AbsRel", "ATE"]);
const sceneDatasetToResultKey = {
  ropedia: "xperience"
};
let tagRankScenes = [];

function sceneDensity(scene) {
  return scene.view_density ?? scene.tags?.view_density;
}

function sceneResultDatasetKey(scene) {
  return sceneDatasetToResultKey[scene.source_dataset] ?? scene.source_dataset;
}

function hasTagRankSelection() {
  return ["density", "environment", "dynamics", "viewType", "dataType"].some((key) => tagRankFilters[key]?.value);
}

function matchesTagRankFilters(scene) {
  if (tagRankFilters.density.value && sceneDensity(scene) !== tagRankFilters.density.value) return false;
  if (tagRankFilters.environment.value && scene.tags.environment !== tagRankFilters.environment.value) return false;
  if (tagRankFilters.dynamics.value && scene.tags.dynamics !== tagRankFilters.dynamics.value) return false;
  if (tagRankFilters.viewType.value && scene.tags.view_type !== tagRankFilters.viewType.value) return false;
  if (tagRankFilters.dataType.value && scene.tags.data_type !== tagRankFilters.dataType.value) return false;
  return true;
}

function buildTagSelection(rows) {
  const buckets = new Map();
  const datasetKeys = new Set();
  let scoreableScenes = 0;
  let unscoredScenes = 0;

  for (const scene of rows) {
    const datasetKey = sceneResultDatasetKey(scene);
    const density = sceneDensity(scene);
    if (!density || !datasetResultsData?.[datasetKey]) {
      unscoredScenes += 1;
      continue;
    }
    const key = `${datasetKey}|${density}`;
    buckets.set(key, (buckets.get(key) ?? 0) + 1);
    datasetKeys.add(datasetKey);
    scoreableScenes += 1;
  }

  return { buckets, datasetKeys, scoreableScenes, unscoredScenes };
}

function leaderboardMetaByMethod() {
  return new Map(data.leaderboard.map((row) => [row.method, row]));
}

function addTagMetric(agg, metric, value, weight) {
  if (!isNumber(value)) return;
  if (!agg.metrics[metric]) {
    agg.metrics[metric] = { sum: 0, weight: 0, value: null };
  }
  agg.metrics[metric].sum += value * weight;
  agg.metrics[metric].weight += weight;
}

function buildTagLeaderboardRows(selection) {
  const metaByMethod = leaderboardMetaByMethod();
  const rowsByMethod = new Map();

  for (const [bucketKey, weight] of selection.buckets.entries()) {
    const [datasetKey, density] = bucketKey.split("|");
    const ds = datasetResultsData?.[datasetKey];
    if (!ds) continue;

    for (const row of ds.rows) {
      const meta = metaByMethod.get(row.method);
      if (!rowsByMethod.has(row.method)) {
        rowsByMethod.set(row.method, {
          method: row.method,
          paradigm: meta?.paradigm ?? row.category ?? "",
          ours: Boolean(meta?.ours || row.ours),
          metrics: {},
          ranks: {}
        });
      }
      const agg = rowsByMethod.get(row.method);

      ds.layout.forEach((col, index) => {
        if (col.regime.toLowerCase() !== density) return;
        const cell = row.cells[index];
        const value = cell && !cell.oom ? cell.value : null;
        addTagMetric(agg, col.metric, value, weight);
      });
    }
  }

  const rows = [...rowsByMethod.values()].map((row) => {
    for (const metric of tagRankMetrics) {
      const metricAgg = row.metrics[metric];
      if (metricAgg?.weight) {
        metricAgg.value = metricAgg.sum / metricAgg.weight;
      }
    }
    row.sceneCount = row.metrics.AbsRel?.weight ?? Math.max(0, ...tagRankMetrics.map((metric) => row.metrics[metric]?.weight ?? 0));
    return row;
  }).filter((row) => tagRankMetrics.some((metric) => isNumber(row.metrics[metric]?.value)));

  for (const metric of tagRankMetrics) {
    rows
      .filter((row) => isNumber(row.metrics[metric]?.value))
      .sort((a, b) => {
        const diff = a.metrics[metric].value - b.metrics[metric].value;
        return tagLowerIsBetter.has(metric) ? diff : -diff;
      })
      .slice(0, 3)
      .forEach((row, index) => {
        row.ranks[metric] = `rank-${index + 1}`;
      });
  }

  const sortMetric = tagRankFilters.sort.value;
  return rows.sort((a, b) => {
    const av = a.metrics[sortMetric]?.value;
    const bv = b.metrics[sortMetric]?.value;
    if (isNumber(av) && isNumber(bv)) {
      const diff = av - bv;
      return tagLowerIsBetter.has(sortMetric) ? diff : -diff;
    }
    if (isNumber(av)) return -1;
    if (isNumber(bv)) return 1;
    return a.method.localeCompare(b.method);
  });
}

function renderTagMetricCell(row, metric) {
  const metricAgg = row.metrics[metric];
  if (!isNumber(metricAgg?.value)) {
    return `<td class="muted">&mdash;</td>`;
  }
  const rank = row.ranks[metric] ? ` class="${row.ranks[metric]}"` : "";
  const title = `${metricAgg.weight} matched scene${metricAgg.weight === 1 ? "" : "s"}`;
  return `<td${rank} title="${title}">${fmtNumber(metricAgg.value)}</td>`;
}

function renderTagLeaderboard() {
  if (!tagLeaderboardTableBody || !datasetResultsData || !tagRankScenes.length) return;
  if (!hasTagRankSelection()) {
    tagLeaderboardContainer.hidden = true;
    tagLeaderboardEmpty.hidden = false;
    tagLeaderboardEmpty.textContent = "Select scene tags above to display the leaderboard.";
    tagLeaderboardCount.textContent = "No scene-tag selection";
    tagLeaderboardTableBody.innerHTML = "";
    return;
  }
  const matchedScenes = tagRankScenes.filter(matchesTagRankFilters);
  const selection = buildTagSelection(matchedScenes);
  const rows = buildTagLeaderboardRows(selection);
  const selectedText = `${matchedScenes.length} scene${matchedScenes.length === 1 ? "" : "s"}`;
  const scoredText = `${selection.scoreableScenes} scored`;
  const datasetText = `${selection.datasetKeys.size} dataset${selection.datasetKeys.size === 1 ? "" : "s"}`;
  const missingText = selection.unscoredScenes ? ` · ${selection.unscoredScenes} without per-dataset scores` : "";
  tagLeaderboardCount.textContent = `${selectedText} · ${scoredText} · ${datasetText}${missingText}`;

  if (!rows.length) {
    tagLeaderboardContainer.hidden = true;
    tagLeaderboardEmpty.hidden = false;
    tagLeaderboardEmpty.textContent = matchedScenes.length
      ? "No numeric per-dataset scores are available for this scene-tag selection."
      : "No scenes match this scene-tag selection.";
    return;
  }

  tagLeaderboardTableBody.innerHTML = rows.map((row) => {
    const oursBadge = row.ours ? ' <span class="tag slam-based">Ours</span>' : "";
    const paradigm = row.paradigm ? `<span class="tag ${slug(row.paradigm)}">${row.paradigm}</span>` : "";
    return `<tr class="${row.ours ? "ours-row" : ""}">
      <td class="method-name sticky-col">${row.method}${oursBadge}</td>
      <td>${paradigm}</td>
      <td class="scene-count">${row.sceneCount}/${selection.scoreableScenes}</td>
      ${tagRankMetrics.map((metric) => renderTagMetricCell(row, metric)).join("")}
    </tr>`;
  }).join("");
  tagLeaderboardContainer.hidden = false;
  tagLeaderboardEmpty.hidden = true;
}

Object.values(tagRankFilters).forEach((el) => {
  if (el) el.addEventListener("input", renderTagLeaderboard);
});

tagLeaderboardReset?.addEventListener("click", () => {
  ["density", "environment", "dynamics", "viewType", "dataType"].forEach((key) => {
    tagRankFilters[key].value = "";
  });
  tagRankFilters.sort.value = "AbsRel";
  renderTagLeaderboard();
});

function matchesFilters(scene) {
  const q = viewerFilters.search.value.trim().toLowerCase();
  if (q && !scene.scene_id.toLowerCase().includes(q)) return false;
  if (viewerFilters.density.value && scene.view_density !== viewerFilters.density.value) return false;
  if (viewerFilters.dataset.value && scene.source_dataset !== viewerFilters.dataset.value) return false;
  if (viewerFilters.environment.value && scene.tags.environment !== viewerFilters.environment.value) return false;
  if (viewerFilters.dynamics.value && scene.tags.dynamics !== viewerFilters.dynamics.value) return false;
  if (viewerFilters.viewType.value && scene.tags.view_type !== viewerFilters.viewType.value) return false;
  if (viewerFilters.dataType.value && scene.tags.data_type !== viewerFilters.dataType.value) return false;
  return true;
}

function renderViewerList() {
  const rows = viewerScenes.filter(matchesFilters);
  viewerCountEl.textContent = `${rows.length} scene${rows.length === 1 ? "" : "s"}`;
  viewerListEl.innerHTML = rows.map((scene) => `
    <li>
      <button type="button" data-scene-id="${scene.scene_id}" class="viewer-item ${activeScene?.scene_id === scene.scene_id ? "active" : ""}">
        <span class="viewer-item-name">${scene.scene_id}</span>
        <span class="viewer-item-meta">
          <span class="tag density-${scene.view_density}">${scene.view_density}</span>
          <span>${scene.size_mb.toFixed(1)} MB</span>
        </span>
      </button>
    </li>
  `).join("");
}

function renderMeta(scene) {
  const entries = [
    ["Dataset", scene.source_dataset],
    ["Density", scene.view_density],
    ["Environment", scene.tags.environment],
    ["Dynamics", scene.tags.dynamics],
    ["View type", scene.tags.view_type],
    ["Source", scene.tags.data_type],
    ["Frames", scene.n_frames?.toLocaleString?.() ?? scene.n_frames],
    ["Points", scene.num_points?.toLocaleString?.() ?? scene.num_points],
    ["Size", `${scene.size_mb.toFixed(1)} MB`]
  ];
  viewerMeta.innerHTML = entries.map(([k, v]) => `<div><dt>${k}</dt><dd>${titleCase(String(v))}</dd></div>`).join("");
  viewerMeta.hidden = false;
}

let viewerStatusTimer = null;

function setViewerStatus(text) {
  if (text === null) {
    viewerStatus.hidden = true;
    viewerStatus.textContent = "";
  } else {
    viewerStatus.hidden = false;
    viewerStatus.textContent = text;
  }
}

// Some source datasets export GLBs in OpenCV camera conventions (Y-down),
// which loads upside-down under model-viewer's Y-up assumption. Apply a
// per-dataset orientation override so the natural "up" of the scene maps
// to +Y, otherwise the useful viewing angles fall onto the phi=0/180 poles
// where the orbit camera is locked. Attribute is "roll pitch yaw"; pitch
// 180° rotates around X, flipping Y-down to Y-up.
const DATASET_ORIENTATION = {
  "7scenes": "0deg 180deg 0deg",
  "omniworld": "0deg 180deg 0deg"
};

// Datasets whose per-frame camera frustums dominate the bounding box (driving /
// outdoor captures with very wide FOV). Hide them at view time — the GLBs pack
// frustums as triangle meshes and the point cloud as a single Points node, so
// we can drop visibility of every Mesh in the scene tree.
const HIDE_FRUSTUM_DATASETS = new Set(["waymo"]);

function findThreeScene(el) {
  for (const sym of Object.getOwnPropertySymbols(el)) {
    const v = el[sym];
    if (v && typeof v.traverse === "function") return v;
  }
  return null;
}

function applyFrustumVisibility(scene) {
  if (!scene) return;
  const root = findThreeScene(sceneViewer);
  if (!root) return;
  const hide = HIDE_FRUSTUM_DATASETS.has(scene.source_dataset);
  root.traverse((obj) => {
    if (obj.isMesh) obj.visible = !hide;
  });
}

function loadScene(scene) {
  activeScene = scene;
  const url = GLB_BASE_URL + scene.glb_path;
  viewerStageEyebrow.textContent = `${titleCase(scene.source_dataset)} · ${scene.view_density}`;
  viewerStageTitle.textContent = scene.scene_id;
  viewerDownload.href = url;
  viewerDownload.hidden = false;
  viewerEmpty.hidden = true;
  setViewerStatus("Loading…");
  sceneViewer.classList.add("active");

  const orientation = DATASET_ORIENTATION[scene.source_dataset] ?? "0deg 0deg 0deg";
  sceneViewer.setAttribute("orientation", orientation);
  sceneViewer.removeAttribute("camera-orbit");
  sceneViewer.removeAttribute("camera-target");

  sceneViewer.src = url;
  renderMeta(scene);
  renderViewerList();

  clearTimeout(viewerStatusTimer);
  viewerStatusTimer = setTimeout(() => setViewerStatus(null), 12000);
}

sceneViewer.addEventListener("progress", (event) => {
  const pct = event.detail?.totalProgress ?? 0;
  if (pct >= 1) {
    setViewerStatus(null);
  } else if (pct > 0) {
    setViewerStatus(`Loading ${Math.round(pct * 100)}%`);
  }
});
sceneViewer.addEventListener("load", () => {
  setViewerStatus(null);
  applyFrustumVisibility(activeScene);
});
sceneViewer.addEventListener("model-visibility", (event) => {
  if (event.detail?.visible) {
    setViewerStatus(null);
    applyFrustumVisibility(activeScene);
  }
});
sceneViewer.addEventListener("error", (event) => {
  setViewerStatus(`Failed: ${event.detail?.sourceError?.message ?? "check GLB URL"}`);
});

viewerListEl.addEventListener("click", (event) => {
  const btn = event.target.closest("[data-scene-id]");
  if (!btn) return;
  const scene = viewerScenes.find((s) => s.scene_id === btn.dataset.sceneId);
  if (scene) loadScene(scene);
});

Object.values(viewerFilters).forEach((el) => el.addEventListener("input", renderViewerList));

viewerResetBtn.addEventListener("click", () => {
  Object.values(viewerFilters).forEach((el) => { el.value = ""; });
  renderViewerList();
});

fetch(MANIFEST_URL)
  .then((r) => {
    if (!r.ok) throw new Error(`Manifest fetch failed (${r.status})`);
    return r.json();
  })
  .then((scenes) => {
    viewerScenes = scenes;
    populateSelect(viewerFilters.density, "All densities", uniqueValues(scenes, "view_density"),
      (a, b) => densityOrder.indexOf(a) - densityOrder.indexOf(b));
    populateSelect(viewerFilters.dataset, "All datasets", uniqueValues(scenes, "source_dataset"));
    populateSelect(viewerFilters.environment, "All environments", uniqueValues(scenes, "tags.environment"));
    populateSelect(viewerFilters.dynamics, "All dynamics", uniqueValues(scenes, "tags.dynamics"));
    populateSelect(viewerFilters.viewType, "All view types", uniqueValues(scenes, "tags.view_type"));
    populateSelect(viewerFilters.dataType, "All sources", uniqueValues(scenes, "tags.data_type"));
    renderViewerList();
  })
  .catch((err) => {
    viewerCountEl.textContent = "Manifest unavailable";
    viewerListEl.innerHTML = `<li class="viewer-empty-row">Could not load manifest: ${err.message}</li>`;
  });

fetch(TAG_SCENES_URL)
  .then((r) => {
    if (!r.ok) throw new Error(`all_scenes.json fetch failed (${r.status})`);
    return r.json();
  })
  .then((scenes) => {
    tagRankScenes = scenes;
    populateSelect(tagRankFilters.density, "All densities", uniqueValues(scenes, "tags.view_density"),
      (a, b) => densityOrder.indexOf(a) - densityOrder.indexOf(b));
    populateSelect(tagRankFilters.environment, "All environments", uniqueValues(scenes, "tags.environment"));
    populateSelect(tagRankFilters.dynamics, "All dynamics", uniqueValues(scenes, "tags.dynamics"));
    populateSelect(tagRankFilters.viewType, "All view types", uniqueValues(scenes, "tags.view_type"));
    populateSelect(tagRankFilters.dataType, "All sources", uniqueValues(scenes, "tags.data_type"));
    renderTagLeaderboard();
  })
  .catch((err) => {
    tagLeaderboardCount.textContent = "Scene tags unavailable";
    tagLeaderboardEmpty.hidden = false;
    tagLeaderboardEmpty.textContent = `Could not load scene-tag leaderboard: ${err.message}`;
  });
