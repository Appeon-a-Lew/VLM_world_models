"use strict";

/* Experiment dashboard.  No external dependencies, no CDNs.  Geometry
 * mirrors physwm/geometry.py and the box-corner formula in
 * scripts/viz_samples.py.
 *
 * Deployment contract: every network access goes through fetchJSON() and
 * fileURL(), driven by window.DASH_CONFIG:
 *
 *   api_base        prefix for JSON endpoints, trailing "/" ("/api/")
 *   file_template   episode image URL with {split}, {id}, {name}
 *   paginate        true when the backend pages /api/episodes (server mode)
 *
 * dashboard_export.py writes a config.js that selects static mode
 * ("api/", "files/{split}/{id}/{name}", paginate:false); ?base=<url-prefix>
 * overrides the bundle location at runtime. */

const DASH_DEFAULTS = {
  api_base: "/api/",
  file_template: "/file/{split}/{id}/{name}",
  paginate: true,
};

function withTrailingSlash(value) {
  return String(value).replace(/\/+$/, "") + "/";
}

const PAGES_REMOTE = {
  api_base: "https://huggingface.co/datasets/Appeon-a-Lew/physwm-tabletop-pushes/resolve/main/site/v3/api/",
  file_template: "https://huggingface.co/datasets/Appeon-a-Lew/physwm-tabletop-pushes/resolve/main/v3/{split}/{id}/{name}",
  paginate: false,
};

const DASH_CONFIG = (() => {
  const defaults = (!window.DASH_CONFIG && window.location.hostname.endsWith(".github.io"))
    ? PAGES_REMOTE : DASH_DEFAULTS;
  const config = Object.assign({}, defaults, window.DASH_CONFIG || {});
  let base = null;
  try {
    base = new URLSearchParams(window.location.search).get("base");
  } catch (_) {
    base = null;
  }
  if (base) {
    const root = String(base).replace(/\/+$/, "");
    config.api_base = `${root}/api/`;
    config.file_template = `${root}/files/{split}/{id}/{name}`;
    config.paginate = false;
  }
  config.api_base = withTrailingSlash(config.api_base);
  return config;
})();

const TASK_TIME = { A: 0.0, B: 1.0 };
const TASK_TOL = 0.051;

/* Edge index order follows viz_samples.box_corners: sx/sy/sz with sz innermost. */
const BOX_EDGES = [
  [0, 1], [1, 3], [3, 2], [2, 0],
  [4, 5], [5, 7], [7, 6], [6, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];
const PALETTE = ["#e65050", "#50a0f0", "#6ec86e", "#f0c85a", "#c878e6", "#78dcd2"];
const ERROR_COLOUR = "#f472b6";
const SEG_COLOURS = ["#e65050", "#50a0f0", "#6ec86e", "#f0c85a", "#c878e6", "#78dcd2"];

const $ = (selector) => document.querySelector(selector);

const els = {
  status: $("#status"),
  splitList: $("#split-list"),
  episodeList: $("#episode-list"),
  epCount: $("#ep-count"),
  epFilter: $("#ep-filter"),
  emptyState: $("#empty-state"),
  emptyMsg: $("#empty-msg"),
  episodeView: $("#episode-view"),
  epTitle: $("#ep-title"),
  epSubtitle: $("#ep-subtitle"),
  epBadges: $("#ep-badges"),
  epAction: $("#ep-action"),
  stage: $("#stage"),
  rgb: $("#rgb-img"),
  seg: $("#seg-img"),
  overlay: $("#overlay"),
  frameBadge: $("#frame-badge"),
  viewHint: $("#view-hint"),
  frameSlider: $("#frame-slider"),
  frameReadout: $("#frame-readout"),
  viewWrap: $("#view-wrap"),
  viewSelect: $("#view-select"),
  toggleGt: $("#toggle-gt"),
  toggleLabels: $("#toggle-labels"),
  togglePreds: $("#toggle-preds"),
  toggleErrors: $("#toggle-errors"),
  toggleSeg: $("#toggle-seg"),
  predSource: $("#pred-source"),
  taskA: $("#task-A"),
  taskB: $("#task-B"),
  predNote: $("#pred-note"),
  legend: $("#legend"),
  predTable: $("#pred-table"),
  rawContainer: $("#raw-container"),
  viewerPanel: $("#viewer-panel"),
  metricsPanel: $("#metrics-panel"),
  metricsBody: $("#metrics-body"),
};

const state = {
  splits: [],
  split: null,
  episodes: [],
  episode: null,
  episodeId: null,
  frameIdx: 0,
  viewIdx: 0,
  preds: [],
  predName: "",
  records: [],
  show: { gt: true, labels: true, preds: true, errors: true, seg: false },
  tasks: { A: true, B: true },
  metrics: null,
  navSeq: 0,
  loadToken: 0,
};

/* ------------------------------------------------------------------ */
/* small helpers                                                       */
/* ------------------------------------------------------------------ */

function encode(value) {
  return encodeURIComponent(String(value));
}

/* Every JSON endpoint is fetched through here; the bundle stores
 * api/<path>.json and the Python server answers both /api/<path> and
 * /api/<path>.json.  A query string is preserved after the suffix. */
async function fetchJSON(rel) {
  const queryAt = rel.indexOf("?");
  const path = queryAt === -1 ? rel : rel.slice(0, queryAt);
  const query = queryAt === -1 ? "" : rel.slice(queryAt);
  const url = `${DASH_CONFIG.api_base}${path}.json${query}`;
  const response = await fetch(url, { headers: { Accept: "application/json" }, cache: "no-store" });
  if (!response.ok) {
    let message = response.statusText || `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body && body.error) message = body.error;
    } catch (_) { /* non-JSON error body */ }
    throw new Error(message);
  }
  return response.json();
}

/* Episode images (rgb/seg png) always go through the configured template. */
function fileURL(split, episodeId, name) {
  return DASH_CONFIG.file_template
    .split("{split}").join(encode(split))
    .split("{id}").join(encode(episodeId))
    .split("{name}").join(encode(name));
}

/* Static bundles ship the full episode list and have no paging; the server
 * keeps the ?split= query contract. */
function episodesPath(split) {
  return DASH_CONFIG.paginate === false
    ? `episodes/${encode(split)}`
    : `episodes?split=${encode(split)}`;
}

function setStatus(message, isError = false) {
  els.status.textContent = message || "";
  els.status.classList.toggle("error", Boolean(isError));
}

function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function showEmpty(message) {
  els.emptyState.hidden = false;
  els.episodeView.hidden = true;
  if (message) els.emptyMsg.textContent = message;
}

function clearEpisode() {
  state.episode = null;
  state.episodeId = null;
  state.records = [];
  state.loadToken += 1;
  els.episodeView.hidden = true;
  els.emptyState.hidden = false;
  els.epFilter.value = "";
  applyFilter();
  markActive();
}

/* ------------------------------------------------------------------ */
/* geometry — exact mirrors of physwm/geometry.py                      */
/* ------------------------------------------------------------------ */

function quatToMat(quat) {
  let [w, x, y, z] = quat;
  const norm = Math.hypot(w, x, y, z) || 1;
  w /= norm; x /= norm; y /= norm; z /= norm;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}

/* world_to_cam: p_cam = (p_world - t) @ R  with R = T[:3,:3], t = T[:3,3]. */
function worldToCam(point, cam2world) {
  const T = cam2world;
  const tx = T[0][3], ty = T[1][3], tz = T[2][3];
  const dx = point[0] - tx, dy = point[1] - ty, dz = point[2] - tz;
  return [
    dx * T[0][0] + dy * T[1][0] + dz * T[2][0],
    dx * T[0][1] + dy * T[1][1] + dz * T[2][1],
    dx * T[0][2] + dy * T[1][2] + dz * T[2][2],
  ];
}

/* project: uvw = p_cam @ K.T; return (u/w, v/w). */
function projectCam(pointCam, K) {
  const u = pointCam[0] * K[0][0] + pointCam[1] * K[0][1] + pointCam[2] * K[0][2];
  const v = pointCam[0] * K[1][0] + pointCam[1] * K[1][1] + pointCam[2] * K[1][2];
  const w = pointCam[0] * K[2][0] + pointCam[1] * K[2][1] + pointCam[2] * K[2][2];
  return [u / w, v / w];
}

function projectWorld(pointWorld, camera) {
  const pc = worldToCam(pointWorld, camera.cam2world);
  if (!(pc[2] > 0.01)) return null;
  const uv = projectCam(pc, camera.K);
  if (!Number.isFinite(uv[0]) || !Number.isFinite(uv[1])) return null;
  return { x: uv[0], y: uv[1] };
}

function projectCamCentre(pointCam, camera) {
  if (!(pointCam[2] > 0.01)) return null;
  const uv = projectCam(pointCam, camera.K);
  if (!Number.isFinite(uv[0]) || !Number.isFinite(uv[1])) return null;
  return { x: uv[0], y: uv[1] };
}

/* box_corners: local = [sx*hx, sy*hy, sz*hz]; world = local @ R.T + pos. */
function boxCorners(object, objectState) {
  const [hx, hy, hz] = object.half_extents;
  const R = quatToMat(objectState.quat);
  const pos = objectState.pos;
  const corners = [];
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        const l = [sx * hx, sy * hy, sz * hz];
        corners.push([
          R[0][0] * l[0] + R[0][1] * l[1] + R[0][2] * l[2] + pos[0],
          R[1][0] * l[0] + R[1][1] * l[1] + R[1][2] * l[2] + pos[1],
          R[2][0] * l[0] + R[2][1] * l[1] + R[2][2] * l[2] + pos[2],
        ]);
      }
    }
  }
  return corners;
}

/* ------------------------------------------------------------------ */
/* data access helpers                                                 */
/* ------------------------------------------------------------------ */

function currentFrame() {
  return (state.episode && state.episode.frames && state.episode.frames[state.frameIdx]) || null;
}

function viewEntry(frame, wanted) {
  if (!frame || !Array.isArray(frame.views) || !frame.views.length) return null;
  return frame.views.find((entry) => entry.view === wanted) || frame.views[0];
}

function viewCamera(viewIndex) {
  const data = state.episode;
  if (!data) return null;
  const views = Array.isArray(data.views) ? data.views : [];
  return views[viewIndex] || views[0] || data.camera || null;
}

function nearestStateRow(data, t) {
  const states = Array.isArray(data.states) ? data.states : [];
  if (!states.length) return null;
  let times = null;
  if (Array.isArray(data.times) && data.times.length === states.length) times = data.times;
  else if (Array.isArray(data.frame_times) && data.frame_times.length === states.length) times = data.frame_times;
  if (!times) return null;
  let best = -1;
  let bestDelta = Infinity;
  for (let i = 0; i < times.length; i += 1) {
    const delta = Math.abs(Number(times[i]) - t);
    if (delta < bestDelta) { bestDelta = delta; best = i; }
  }
  return best >= 0 ? { row: states[best], delta: bestDelta } : null;
}

function stateRowForTime(data, t) {
  const found = nearestStateRow(data, t);
  return found && found.delta <= 0.3 ? found.row : null;
}

function sameTimeGrid(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
  return a.every((value, index) => Math.abs(Number(value) - Number(b[index])) < 1e-6);
}

function stateRowForFrame(data, frame) {
  const states = Array.isArray(data.states) ? data.states : [];
  if (!states.length || !frame) return null;
  const key = Number.isInteger(frame.k) ? frame.k : state.frameIdx;
  if (Array.isArray(data.frame_times) && data.frame_times.length === states.length
      && sameTimeGrid(data.frame_times, data.times)) {
    return states[key] || states[states.length - 1] || null;
  }
  return stateRowForTime(data, frame.t) || states[key] || states[0] || null;
}

function objectColour(object, index) {
  const rgba = object && object.rgba;
  if (!Array.isArray(rgba) || rgba.length < 3) return PALETTE[index % PALETTE.length];
  const values = rgba.slice(0, 3).map(Number);
  const scale = values.some((value) => value > 1.0001) ? 1 : 255;
  const [r, g, b] = values.map((value) => Math.max(0, Math.min(255, Math.round(value * scale))));
  return `rgb(${r}, ${g}, ${b})`;
}

function predValue(record, object, index) {
  const objects = record && record.objects;
  if (!objects) return null;
  if (Array.isArray(objects)) return objects[index] ?? objects[object.id] ?? null;
  if (typeof objects === "object") {
    if (Object.prototype.hasOwnProperty.call(objects, String(object.id))) return objects[String(object.id)];
    if (Object.prototype.hasOwnProperty.call(objects, object.id)) return objects[object.id];
  }
  return null;
}

function validVec(value) {
  return Array.isArray(value) && value.length >= 3 && value.slice(0, 3).every((x) => Number.isFinite(Number(x)));
}

function recordFor(task) {
  return state.records.find((record) => record.task === task) || null;
}

function pngUrl(name) {
  return fileURL(state.split, state.episodeId, name);
}

/* ------------------------------------------------------------------ */
/* sidebar                                                             */
/* ------------------------------------------------------------------ */

function renderSplits() {
  els.splitList.replaceChildren();
  if (!state.splits.length) {
    els.splitList.append(make("li", "split-item", "No splits found"));
    return;
  }
  for (const split of state.splits) {
    const item = make("li", "split-item");
    item.dataset.split = split.split;
    item.append(make("span", null, split.split));
    item.append(make("span", "n", split.n_episodes));
    item.addEventListener("click", () => {
      window.location.hash = `#${encode(split.split)}`;
    });
    els.splitList.append(item);
  }
  markActive();
}

function renderEpisodes() {
  els.epCount.textContent = `(${state.episodes.length})`;
  els.episodeList.replaceChildren();
  if (!state.episodes.length) {
    els.episodeList.append(make("li", "episode-item", "No episodes in this split."));
    return;
  }
  for (const episode of state.episodes) {
    const item = make("li", "episode-item");
    item.dataset.id = episode.episode_id;
    if (episode.thumb) {
      const img = document.createElement("img");
      img.loading = "lazy";
      img.decoding = "async";
      img.alt = "";
      img.src = fileURL(state.split, episode.episode_id, episode.thumb);
      img.onerror = () => {
        const fallback = make("span", "episode-thumb-fallback", "no rgb");
        if (img.parentNode) img.replaceWith(fallback);
      };
      item.append(img);
    } else {
      item.append(make("span", "episode-thumb-fallback", "no rgb"));
    }
    const meta = make("div", "episode-meta");
    meta.append(make("span", "episode-id", episode.episode_id));
    const elevation = Number.isFinite(Number(episode.elevation_deg))
      ? `${Number(episode.elevation_deg).toFixed(0)}°`
      : "—";
    meta.append(make(
      "span",
      "episode-sub",
      `${episode.n_objects} obj · elev ${elevation} · ${episode.n_frames ?? "?"}f/${episode.n_views ?? "?"}v`,
    ));
    item.append(meta);
    item.addEventListener("click", () => {
      window.location.hash = `#${encode(state.split)}/${encode(episode.episode_id)}`;
    });
    els.episodeList.append(item);
  }
  applyFilter();
  markActive();
}

function applyFilter() {
  const query = els.epFilter.value.trim().toLowerCase();
  for (const item of els.episodeList.querySelectorAll(".episode-item[data-id]")) {
    item.hidden = Boolean(query) && !item.dataset.id.toLowerCase().includes(query);
  }
}

function markActive() {
  for (const item of els.splitList.querySelectorAll(".split-item[data-split]")) {
    item.classList.toggle("active", item.dataset.split === state.split);
  }
  for (const item of els.episodeList.querySelectorAll(".episode-item[data-id]")) {
    item.classList.toggle("active", item.dataset.id === state.episodeId);
  }
}

/* ------------------------------------------------------------------ */
/* prediction source + records                                         */
/* ------------------------------------------------------------------ */

function renderPredSources() {
  els.predSource.replaceChildren();
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "(none)";
  els.predSource.append(none);
  for (const name of state.preds) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    els.predSource.append(option);
  }
  els.predSource.value = state.preds.includes(state.predName) ? state.predName : "";
  els.predSource.disabled = !state.preds.length;
}

async function loadRecords(loadToken = state.loadToken) {
  if (!state.predName || !state.episode) {
    state.records = [];
    renderPredTable();
    renderRaw();
    updatePredNote();
    drawOverlay();
    return;
  }
  const split = state.split;
  const episodeId = state.episodeId;
  try {
    const records = await fetchJSON(`pred/${encode(state.predName)}/${encode(split)}/${encode(episodeId)}`);
    if (loadToken !== state.loadToken) return;
    state.records = Array.isArray(records) ? records : [];
  } catch (error) {
    if (loadToken !== state.loadToken) return;
    state.records = [];
    setStatus(`Predictions: ${error.message}`, true);
  }
  renderPredTable();
  renderRaw();
  updatePredNote();
  drawOverlay();
}

function updatePredNote() {
  const notes = [];
  if (!state.predName) {
    notes.push("No source selected. Predictions are configured at export time.");
  } else if (!state.records.length) {
    notes.push(`No records for “${state.predName}” on this episode.`);
  } else {
    const tasks = [...new Set(state.records.map((record) => record.task).filter(Boolean))];
    notes.push(`${state.records.length} record(s) · tasks ${tasks.join(" / ") || "?"}.`);
    notes.push("Crosses appear on the matching t = 0 s / t = 1 s frame.");
  }
  const entry = currentFrame() ? viewEntry(currentFrame(), state.viewIdx) : null;
  if (state.predName && state.records.length && entry && entry.view !== 0) {
    notes.push("Predicted centres are camera-frame of view 0.");
  }
  els.predNote.textContent = notes.join(" ");
}

/* ------------------------------------------------------------------ */
/* episode rendering                                                   */
/* ------------------------------------------------------------------ */

function renderEpisode() {
  const data = state.episode;
  if (!data) return;
  const frames = Array.isArray(data.frames) ? data.frames : [];
  const views = Array.isArray(data.views) ? data.views : [];
  const cameras = views.length ? views : (data.camera ? [data.camera] : []);
  const camera = cameras[0] || {};

  els.episodeView.hidden = false;
  els.emptyState.hidden = true;
  els.epTitle.textContent = data.episode_id || state.episodeId;
  const elevation = Number.isFinite(Number(camera.elevation_deg))
    ? `${Number(camera.elevation_deg).toFixed(1)}°`
    : "—";
  els.epSubtitle.textContent = [
    data.split,
    data.config ? `config ${data.config}` : null,
    data.seed !== undefined ? `seed ${data.seed}` : null,
    `camera elev ${elevation}`,
    `${frames.length} frame(s) × ${cameras.length} view(s)`,
  ].filter(Boolean).join(" · ");

  els.epBadges.replaceChildren();
  const objects = Array.isArray(data.objects) ? data.objects : [];
  const targetId = data.action ? data.action.target : null;
  const target = objects.find((object) => object.id === targetId);
  for (const [label, value] of [
    ["objects", objects.length],
    ["frames", frames.length],
    ["views", cameras.length],
  ]) {
    const badge = make("span", "badge");
    badge.append(`${label} `, make("b", null, value));
    els.epBadges.append(badge);
  }
  if (target) {
    const badge = make("span", "badge");
    badge.append("target ", make("b", null, target.name));
    els.epBadges.append(badge);
  }

  els.epAction.textContent = (data.action && data.action.text) || "No action text in meta.json.";

  /* frame slider */
  state.frameIdx = Math.min(state.frameIdx, Math.max(0, frames.length - 1));
  els.frameSlider.max = String(Math.max(0, frames.length - 1));
  els.frameSlider.value = String(state.frameIdx);
  els.frameSlider.disabled = frames.length <= 1;

  /* views: union of camera list and frames actually present */
  const viewIndices = new Set();
  for (const frame of frames) for (const entry of frame.views || []) viewIndices.add(entry.view);
  for (let i = 0; i < cameras.length; i += 1) viewIndices.add(i);
  const viewList = [...viewIndices].sort((a, b) => a - b);
  els.viewSelect.replaceChildren();
  for (const index of viewList) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `view ${index}`;
    els.viewSelect.append(option);
  }
  const firstEntry = frames.length ? viewEntry(frames[0], state.viewIdx) : null;
  state.viewIdx = firstEntry ? firstEntry.view : (viewList[0] ?? 0);
  els.viewSelect.value = String(state.viewIdx);
  els.viewWrap.hidden = viewList.length <= 1;
  els.viewSelect.disabled = viewList.length <= 1;

  renderLegend(objects, targetId);
  updateFrame();
  renderPredTable();
  renderRaw();
  updatePredNote();
}

function renderLegend(objects, targetId) {
  els.legend.replaceChildren();
  objects.forEach((object, index) => {
    const item = make("li");
    const swatch = make("span", "swatch");
    swatch.style.background = objectColour(object, index);
    item.append(swatch);
    item.append(make("span", null, object.name));
    const isTarget = object.id === targetId;
    if (isTarget) item.append(make("span", "target", "* target"));
    item.append(make("span", "tag", `#${object.id}`));
    els.legend.append(item);
  });
  const keys = [
    ["▭ +", "ground-truth box / centre", "#9aa8bd"],
    ["✕", "predicted centre (task A/B)", "#e7ecf3"],
    ["┄", "GT → pred centre error", ERROR_COLOUR],
  ];
  for (const [symbol, text, colour] of keys) {
    const item = make("li", "legend-key");
    const mark = make("span", "legend-symbol", symbol);
    mark.style.color = colour;
    item.append(mark, make("span", "muted", text));
    els.legend.append(item);
  }
}

function updateFrame() {
  const data = state.episode;
  const frame = currentFrame();
  if (!data || !frame) {
    els.rgb.removeAttribute("src");
    els.seg.classList.remove("on");
    els.frameBadge.textContent = "";
    clearCanvas();
    return;
  }
  const entry = viewEntry(frame, state.viewIdx);
  if (!entry) {
    els.frameBadge.textContent = "No image for this frame/view";
    els.rgb.removeAttribute("src");
    clearCanvas();
    return;
  }
  if (entry.view !== state.viewIdx) state.viewIdx = entry.view;
  els.viewSelect.value = String(state.viewIdx);

  const src = pngUrl(entry.rgb);
  if (els.rgb.dataset.src !== src) {
    els.rgb.dataset.src = src;
    els.rgb.src = src;
  } else if (els.rgb.complete && els.rgb.naturalWidth) {
    window.requestAnimationFrame(() => {
      fitStage();
      drawOverlay();
    });
  }
  applySeg(entry);

  const total = data.frames.length;
  els.frameBadge.textContent = `t = ${Number(frame.t).toFixed(2)} s · frame ${state.frameIdx + 1}/${total} · view ${entry.view}`;
  els.frameReadout.textContent = `t=${Number(frame.t).toFixed(2)} s`;
  els.frameSlider.value = String(state.frameIdx);
  updatePredNote();
  window.requestAnimationFrame(drawOverlay);
}

function applySeg(entry) {
  if (state.show.seg && entry && entry.seg) {
    const src = pngUrl(entry.seg);
    if (els.seg.dataset.src !== src) {
      els.seg.dataset.src = src;
      els.seg.src = src;
    }
    els.seg.classList.add("on");
  } else {
    els.seg.classList.remove("on");
  }
}

function fitStage() {
  const data = state.episode;
  const frame = currentFrame();
  const entry = frame ? viewEntry(frame, state.viewIdx) : null;
  const camera = entry ? viewCamera(entry.view) : null;
  const width = els.rgb.naturalWidth || (camera && camera.width) || 1;
  const height = els.rgb.naturalHeight || (camera && camera.height) || 1;
  if (width > 0 && height > 0) els.stage.style.aspectRatio = `${width} / ${height}`;
}

/* ------------------------------------------------------------------ */
/* canvas overlay                                                      */
/* ------------------------------------------------------------------ */

function clearCanvas() {
  const ctx = els.overlay.getContext("2d");
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
}

function drawOverlay() {
  const data = state.episode;
  const frame = currentFrame();
  const ctx = els.overlay.getContext("2d");
  if (!data || !frame) { clearCanvas(); return; }
  const entry = viewEntry(frame, state.viewIdx);
  if (!entry) { clearCanvas(); return; }

  const camera = viewCamera(entry.view);
  const width = els.rgb.naturalWidth || (camera && camera.width) || 448;
  const height = els.rgb.naturalHeight || (camera && camera.height) || 448;
  if (els.overlay.width !== width || els.overlay.height !== height) {
    els.overlay.width = width;
    els.overlay.height = height;
  }
  ctx.clearRect(0, 0, width, height);
  if (!camera || !camera.K || !camera.cam2world) return;

  const scale = Math.max(0.85, Math.min(3, width / 448));
  const targetId = data.action ? data.action.target : null;
  const objects = Array.isArray(data.objects) ? data.objects : [];

  /* ground-truth boxes for the currently displayed frame */
  if (state.show.gt) {
    const row = stateRowForFrame(data, frame);
    if (row) {
      objects.forEach((object, index) => {
        const objectState = row[index];
        if (!objectState || !Array.isArray(object.half_extents)) return;
        const colour = objectColour(object, index);
        const isTarget = object.id === targetId;
        const projected = boxCorners(object, objectState).map((corner) => projectWorld(corner, camera));
        if (!projected.some((point) => !point)) {
          drawBox(ctx, projected, colour, isTarget, scale);
        }
        const centre = projectWorld(objectState.pos, camera);
        if (centre) {
          drawPlus(ctx, centre, isTarget ? "#ffffff" : colour, isTarget, scale);
          if (state.show.labels) {
            drawLabel(ctx, object.name + (isTarget ? " *" : ""), centre.x, centre.y, colour, scale, width, height);
          }
        }
      });
    }
  }

  /* predicted centres, only on their task-time frame and only for view 0 */
  const canOverlayPreds = state.show.preds && state.records.length && entry.view === 0;
  if (canOverlayPreds) {
    for (const task of ["A", "B"]) {
      if (!state.tasks[task]) continue;
      if (Math.abs(Number(frame.t) - TASK_TIME[task]) > TASK_TOL) continue;
      const record = recordFor(task);
      if (!record) continue;
      const gtRow = stateRowForTime(data, TASK_TIME[task]);
      objects.forEach((object, index) => {
        const prediction = predValue(record, object, index);
        if (!validVec(prediction)) return;
        const centreCam = prediction.slice(0, 3).map(Number);
        const predUv = projectCamCentre(centreCam, camera);
        if (!predUv) return;
        const colour = objectColour(object, index);

        if (state.show.errors && gtRow && gtRow[index] && Array.isArray(gtRow[index].pos)) {
          const gtUv = projectWorld(gtRow[index].pos, camera);
          if (gtUv) {
            const gtCam = worldToCam(gtRow[index].pos, camera.cam2world);
            const error = Math.hypot(
              centreCam[0] - gtCam[0],
              centreCam[1] - gtCam[1],
              centreCam[2] - gtCam[2],
            );
            drawErrorLine(ctx, gtUv, predUv, scale);
            drawLabel(
              ctx,
              `${error.toFixed(3)} m`,
              (gtUv.x + predUv.x) / 2,
              (gtUv.y + predUv.y) / 2 - 4 * scale,
              ERROR_COLOUR,
              scale,
              width,
              height,
            );
          }
        }
        drawCross(ctx, predUv, colour, scale);
      });
    }
  }

  if (state.show.preds && state.records.length && entry.view !== 0) {
    els.viewHint.hidden = false;
    els.viewHint.textContent = "pred overlays: view 0 only";
  } else {
    els.viewHint.hidden = true;
  }
}

function drawBox(ctx, points, colour, isTarget, scale) {
  ctx.save();
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(0, 0, 0, 0.55)";
  ctx.lineWidth = (isTarget ? 4.6 : 3.6) * scale;
  for (const [a, b] of BOX_EDGES) {
    ctx.beginPath();
    ctx.moveTo(points[a].x, points[a].y);
    ctx.lineTo(points[b].x, points[b].y);
    ctx.stroke();
  }
  ctx.strokeStyle = colour;
  ctx.lineWidth = (isTarget ? 2.4 : 1.7) * scale;
  for (const [a, b] of BOX_EDGES) {
    ctx.beginPath();
    ctx.moveTo(points[a].x, points[a].y);
    ctx.lineTo(points[b].x, points[b].y);
    ctx.stroke();
  }
  if (isTarget) {
    ctx.globalAlpha = 0.9;
    ctx.lineWidth = 0.9 * scale;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
    for (const [a, b] of BOX_EDGES) {
      ctx.beginPath();
      ctx.moveTo(points[a].x, points[a].y);
      ctx.lineTo(points[b].x, points[b].y);
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawPlus(ctx, point, colour, isTarget, scale) {
  const r = (isTarget ? 6 : 4.5) * scale;
  ctx.save();
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(0, 0, 0, 0.7)";
  ctx.lineWidth = (isTarget ? 4 : 3) * scale;
  ctx.beginPath();
  ctx.moveTo(point.x - r, point.y); ctx.lineTo(point.x + r, point.y);
  ctx.moveTo(point.x, point.y - r); ctx.lineTo(point.x, point.y + r);
  ctx.stroke();
  ctx.strokeStyle = colour;
  ctx.lineWidth = (isTarget ? 2 : 1.5) * scale;
  ctx.beginPath();
  ctx.moveTo(point.x - r, point.y); ctx.lineTo(point.x + r, point.y);
  ctx.moveTo(point.x, point.y - r); ctx.lineTo(point.x, point.y + r);
  ctx.stroke();
  ctx.restore();
}

function drawCross(ctx, point, colour, scale) {
  const r = 5 * scale;
  ctx.save();
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(0, 0, 0, 0.75)";
  ctx.lineWidth = 4 * scale;
  ctx.beginPath();
  ctx.moveTo(point.x - r, point.y - r); ctx.lineTo(point.x + r, point.y + r);
  ctx.moveTo(point.x + r, point.y - r); ctx.lineTo(point.x - r, point.y + r);
  ctx.stroke();
  ctx.strokeStyle = colour;
  ctx.lineWidth = 2.2 * scale;
  ctx.beginPath();
  ctx.moveTo(point.x - r, point.y - r); ctx.lineTo(point.x + r, point.y + r);
  ctx.moveTo(point.x + r, point.y - r); ctx.lineTo(point.x - r, point.y + r);
  ctx.stroke();
  ctx.restore();
}

function drawErrorLine(ctx, from, to, scale) {
  ctx.save();
  ctx.setLineDash([5 * scale, 4 * scale]);
  ctx.strokeStyle = "rgba(0, 0, 0, 0.55)";
  ctx.lineWidth = 2.8 * scale;
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
  ctx.strokeStyle = ERROR_COLOUR;
  ctx.lineWidth = 1.5 * scale;
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
  ctx.restore();
}

function drawLabel(ctx, text, anchorX, anchorY, colour, scale, width, height) {
  ctx.save();
  const fontSize = Math.round(11 * scale);
  ctx.font = `600 ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
  ctx.textBaseline = "alphabetic";
  const padding = 3 * scale;
  const boxWidth = ctx.measureText(text).width + padding * 2;
  const boxHeight = fontSize + padding * 2;
  let x = anchorX + 7 * scale;
  let y = anchorY - boxHeight - 3 * scale;
  x = Math.max(2, Math.min(x, width - boxWidth - 2));
  y = Math.max(2, Math.min(y, height - boxHeight - 2));
  ctx.fillStyle = "rgba(5, 8, 12, 0.74)";
  ctx.fillRect(x, y, boxWidth, boxHeight);
  ctx.fillStyle = colour;
  ctx.fillText(text, x + padding, y + padding + fontSize - 1);
  ctx.restore();
}

/* ------------------------------------------------------------------ */
/* per-object table + raw answers                                      */
/* ------------------------------------------------------------------ */

function fmtVec(value) {
  if (!value) return "—";
  return value.slice(0, 3).map((x) => Number(x).toFixed(3)).join(", ");
}

function fmtNum(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "—";
}

function renderPredTable() {
  const box = els.predTable;
  box.replaceChildren();
  const data = state.episode;
  if (!data) return;
  if (!state.predName) {
    box.append(make("div", "empty", "No prediction source selected."));
    return;
  }
  if (!state.records.length) {
    box.append(make("div", "empty", `No records for “${state.predName}” on this episode.`));
    return;
  }
  const cameras = Array.isArray(data.views) && data.views.length ? data.views : (data.camera ? [data.camera] : []);
  const camera = cameras[0];
  const objects = Array.isArray(data.objects) ? data.objects : [];
  let rendered = false;

  for (const task of ["A", "B"]) {
    if (!state.tasks[task]) continue;
    rendered = true;
    const record = recordFor(task);
    const heading = make("h4");
    heading.append(`Task ${task} · t = ${TASK_TIME[task].toFixed(1)} s`);
    const stats = make("span", null, "");
    heading.append(stats);
    box.append(heading);

    if (!record) {
      box.append(make("div", "empty", `No record for task ${task} in “${state.predName}”.`));
      continue;
    }
    const gtRow = stateRowForTime(data, TASK_TIME[task]);
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    for (const label of ["Object", "GT x,y,z", "Pred x,y,z", "L2 (m)", "Miss"]) {
      headerRow.append(make("th", null, label));
    }
    thead.append(headerRow);
    table.append(thead);
    const tbody = document.createElement("tbody");
    const errors = [];
    let missing = 0;
    let targetError = null;

    objects.forEach((object, index) => {
      const row = document.createElement("tr");
      const gtState = gtRow ? gtRow[index] : null;
      const gtCam = gtState && camera ? worldToCam(gtState.pos, camera.cam2world) : null;
      const prediction = predValue(record, object, index);
      const hasPred = validVec(prediction);

      const nameCell = document.createElement("td");
      const swatch = make("span", "swatch");
      swatch.style.background = objectColour(object, index);
      swatch.style.display = "inline-block";
      swatch.style.marginRight = "5px";
      nameCell.append(swatch, document.createTextNode(object.name));
      if (object.id === (data.action && data.action.target)) nameCell.append(" *");
      row.append(nameCell);
      row.append(make("td", "num", gtCam ? fmtVec(gtCam) : "—"));

      if (!hasPred) {
        missing += 1;
        row.classList.add("missing");
        row.append(make("td", null, "—"), make("td", null, "—"), make("td", "missing-tag", "missing"));
      } else {
        const predCam = prediction.slice(0, 3).map(Number);
        row.append(make("td", "num", fmtVec(predCam)));
        if (gtCam) {
          const error = Math.hypot(predCam[0] - gtCam[0], predCam[1] - gtCam[1], predCam[2] - gtCam[2]);
          errors.push(error);
          if (object.id === (data.action && data.action.target)) targetError = error;
          row.append(make("td", "err", fmtNum(error)), make("td", null, "no"));
        } else {
          row.append(make("td", null, "—"), make("td", null, "no"));
        }
      }
      tbody.append(row);
    });

    table.append(tbody);
    const mean = errors.length ? errors.reduce((a, b) => a + b, 0) / errors.length : null;
    stats.textContent = `mean ${fmtNum(mean)} m · target ${fmtNum(targetError)} m · ${missing} missing`;
    box.append(table);
  }
  if (!rendered) box.append(make("div", "empty", "Enable task A or task B."));
}

function renderRaw() {
  const box = els.rawContainer;
  box.replaceChildren();
  if (!state.predName) {
    box.append(make("p", "muted small", "Select a prediction source to see raw answers."));
    return;
  }
  if (!state.records.length) {
    box.append(make("p", "muted small", "No records for this episode."));
    return;
  }
  for (const task of ["A", "B"]) {
    const records = state.records.filter((record) => record.task === task);
    records.forEach((record, index) => {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = `Task ${task}${records.length > 1 ? ` #${index + 1}` : ""} — raw answer`;
      details.append(summary);
      const info = [];
      if (record.style) info.push(`style=${record.style}`);
      if (record.image_mode) info.push(`image_mode=${record.image_mode}`);
      if (record.lora) info.push(`lora=${record.lora}`);
      if (record.parsed !== undefined) info.push(`parsed=${record.parsed}`);
      if (info.length) details.append(make("div", "raw-meta", info.join(" · ")));
      const pre = document.createElement("pre");
      pre.textContent = record.raw === undefined || record.raw === null
        ? "(no `raw` field in this record)"
        : String(record.raw);
      details.append(pre);
      box.append(details);
    });
  }
  if (!box.children.length) box.append(make("p", "muted small", "Records contain no task A/B answers."));
}

/* ------------------------------------------------------------------ */
/* metrics tab                                                         */
/* ------------------------------------------------------------------ */

async function loadMetrics() {
  if (state.metrics) { renderMetrics(); return; }
  els.metricsBody.replaceChildren(make("p", "muted", "Loading metrics…"));
  try {
    state.metrics = await fetchJSON("metrics");
    renderMetrics();
  } catch (error) {
    els.metricsBody.replaceChildren(make("div", "metrics-empty", `Could not load metrics: ${error.message}`));
  }
}

function renderMetrics() {
  const body = els.metricsBody;
  body.replaceChildren();
  const metrics = state.metrics;
  if (!metrics) return;
  let any = false;

  if (metrics.table_md) {
    any = true;
    body.append(make("h2", null, "table.md"));
    const pre = make("pre", "table-md", metrics.table_md);
    body.append(pre);
  }

  if (Array.isArray(metrics.metrics_csv) && metrics.metrics_csv.length) {
    any = true;
    body.append(make("h2", null, "metrics.csv"));
    const rows = metrics.metrics_csv;
    const columns = Math.max(...rows.map((row) => row.length));
    const wrap = make("div", "metrics-table-wrap");
    const table = make("table", "metrics-table");
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    for (let i = 0; i < columns; i += 1) headerRow.append(make("th", null, rows[0][i] ?? ""));
    thead.append(headerRow);
    table.append(thead);
    const tbody = document.createElement("tbody");
    for (const row of rows.slice(1)) {
      if (!row.some((cell) => String(cell).trim())) continue;
      const tr = document.createElement("tr");
      for (let i = 0; i < columns; i += 1) tr.append(make("td", null, row[i] ?? ""));
      tbody.append(tr);
    }
    table.append(tbody);
    wrap.append(table);
    body.append(wrap);
  }

  if (!any) {
    body.append(make(
      "div",
      "metrics-empty",
      "No results found. Launch with --runs DIR containing results/table.md and/or results/metrics.csv.",
    ));
  }
}

function setTab(tab) {
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tab);
  });
  els.viewerPanel.hidden = tab !== "viewer";
  els.metricsPanel.hidden = tab !== "metrics";
  if (tab === "metrics") loadMetrics();
}

/* ------------------------------------------------------------------ */
/* routing                                                             */
/* ------------------------------------------------------------------ */

function parseHash() {
  let raw = window.location.hash.replace(/^#/, "");
  try { raw = decodeURIComponent(raw); } catch (_) { /* keep raw */ }
  if (!raw) return {};
  const slash = raw.indexOf("/");
  if (slash === -1) return { split: raw };
  return { split: raw.slice(0, slash), id: raw.slice(slash + 1) };
}

async function selectSplit(split, episodeId) {
  const seq = ++state.navSeq;
  const changed = state.split !== split;
  if (changed) {
    state.split = split;
    state.episodes = [];
    clearEpisode();
    showEmpty(`Loading ${split}…`);
    markActive();
    setStatus(`Loading ${split}…`);
    try {
      const episodes = await fetchJSON(episodesPath(split));
      if (seq !== state.navSeq) return;
      state.episodes = Array.isArray(episodes) ? episodes : [];
    } catch (error) {
      if (seq !== state.navSeq) return;
      state.episodes = [];
      setStatus(`Episodes: ${error.message}`, true);
      showEmpty(`Could not load split “${split}”: ${error.message}`);
      renderEpisodes();
      return;
    }
    renderEpisodes();
  } else {
    markActive();
  }
  if (episodeId) {
    await selectEpisode(episodeId);
  } else {
    clearEpisode();
    showEmpty("Select an episode from the left to inspect its frames, boxes and predictions.");
    setStatus("");
  }
}

async function selectEpisode(episodeId) {
  const seq = state.navSeq;
  const loadToken = ++state.loadToken;
  state.episodeId = episodeId;
  markActive();
  setStatus(`Loading ${episodeId}…`);
  try {
    const data = await fetchJSON(`episode/${encode(state.split)}/${encode(episodeId)}`);
    if (loadToken !== state.loadToken || seq !== state.navSeq) return;
    state.episode = data;
    state.episodeId = data.episode_id || episodeId;
    state.frameIdx = 0;
    state.records = [];
    renderEpisode();
    markActive();
    await loadRecords(loadToken);
    setStatus("");
  } catch (error) {
    if (loadToken !== state.loadToken || seq !== state.navSeq) return;
    setStatus(`Episode: ${error.message}`, true);
    showEmpty(`Could not load “${episodeId}”: ${error.message}`);
  }
}

async function route() {
  const { split, id } = parseHash();
  if (!state.splits.length) return;
  setTab("viewer");
  if (!split) {
    const first = state.splits[0].split;
    window.history.replaceState(null, "", `#${encode(first)}`);
    await selectSplit(first, id);
    return;
  }
  if (!state.splits.some((entry) => entry.split === split)) {
    setStatus(`Unknown split in URL: ${split}`, true);
    return;
  }
  await selectSplit(split, id);
}

/* ------------------------------------------------------------------ */
/* wiring + boot                                                       */
/* ------------------------------------------------------------------ */

function bindUI() {
  els.frameSlider.addEventListener("input", (event) => {
    state.frameIdx = Number(event.target.value);
    updateFrame();
  });

  els.viewSelect.addEventListener("change", (event) => {
    state.viewIdx = Number(event.target.value);
    updateFrame();
  });

  els.toggleGt.addEventListener("change", (event) => {
    state.show.gt = event.target.checked;
    drawOverlay();
  });
  els.toggleLabels.addEventListener("change", (event) => {
    state.show.labels = event.target.checked;
    drawOverlay();
  });
  els.togglePreds.addEventListener("change", (event) => {
    state.show.preds = event.target.checked;
    drawOverlay();
  });
  els.toggleErrors.addEventListener("change", (event) => {
    state.show.errors = event.target.checked;
    drawOverlay();
  });
  els.toggleSeg.addEventListener("change", (event) => {
    state.show.seg = event.target.checked;
    const frame = currentFrame();
    applySeg(frame ? viewEntry(frame, state.viewIdx) : null);
  });

  els.predSource.addEventListener("change", (event) => {
    state.predName = event.target.value;
    loadRecords();
  });

  els.taskA.addEventListener("change", (event) => {
    state.tasks.A = event.target.checked;
    renderPredTable();
    drawOverlay();
  });
  els.taskB.addEventListener("change", (event) => {
    state.tasks.B = event.target.checked;
    renderPredTable();
    drawOverlay();
  });

  els.epFilter.addEventListener("input", applyFilter);

  els.rgb.addEventListener("load", () => {
    fitStage();
    drawOverlay();
  });
  els.rgb.addEventListener("error", () => {
    setStatus("Could not load frame image", true);
    clearCanvas();
  });

  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => setTab(button.dataset.tab));
  });

  window.addEventListener("hashchange", () => {
    route();
  });
}

async function boot() {
  bindUI();
  setStatus("Loading splits…");
  try {
    const splits = await fetchJSON("splits");
    state.splits = Array.isArray(splits) ? splits : [];
  } catch (error) {
    setStatus(`Splits: ${error.message}`, true);
    showEmpty(`Could not load splits: ${error.message}`);
    return;
  }
  renderSplits();

  try {
    const preds = await fetchJSON("preds");
    state.preds = Array.isArray(preds) ? preds : [];
  } catch (_) {
    state.preds = [];
  }
  if (state.preds.length && !state.predName) state.predName = state.preds[0];
  renderPredSources();

  if (!state.splits.length) {
    setStatus("No splits found", true);
    showEmpty(DASH_CONFIG.paginate === false
      ? "No data found in this static bundle. Re-run dashboard_export.py with more episodes."
      : "No data roots found. Check the --data path passed to dashboard.py.");
    return;
  }
  await route();
}

boot();