const SHOT_LABELS = {
  defensive_shot: "수비형 샷", drive: "드라이브", drop: "드롭", hairpin: "헤어핀",
  high_clear: "하이클리어", long_service: "롱서비스", push: "푸시",
  short_service: "숏서비스", smash: "스매시", under_clear: "언더클리어",
};
const SERVICE_SHOTS = ["short_service", "long_service"];
const RALLY_SHOTS = Object.keys(SHOT_LABELS).filter((shot) => !SERVICE_SHOTS.includes(shot));
const X_EDGES = [-2.59, -2.59 + 5.18 / 6, 0, 2.59 - 5.18 / 6, 2.59];
const LOCAL_Y_EDGES = [0, 6.7 / 6, 3.35, 6.7 - 6.7 / 6, 6.7];
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
}[char]));

function freshDraft(hitter = "A", first = false, positions = {}) {
  return {
    hitter, shot_type: "", target_zone: null,
    player_a_position: positions.player_a_position || null,
    player_b_position: positions.player_b_position || null,
    delta_time_s: first ? 0 : 0.8,
  };
}

const state = {
  strokes: [], analysis: null, activeTarget: "a", selectedResult: null,
  mode: "new", draft: freshDraft("A", true), examples: [],
};

const midpoint = (edges, index) => (edges[index] + edges[index + 1]) / 2;
function cellCenter(side, row, column) {
  const localY = midpoint(LOCAL_Y_EDGES, row);
  return { x_m: midpoint(X_EDGES, column), y_m: side === "A" ? localY : 13.4 - localY };
}
function targetZone(hitter, row, column) {
  return row * 4 + (hitter === "A" ? column : 3 - column) + 1;
}
function zoneCell(hitter, zone) {
  const row = Math.floor((zone - 1) / 4);
  const actorColumn = (zone - 1) % 4;
  return { side: hitter === "A" ? "B" : "A", row, column: hitter === "A" ? actorColumn : 3 - actorColumn };
}
function nearestCell(position, side) {
  if (!position) return null;
  const localY = side === "A" ? position.y_m : 13.4 - position.y_m;
  const row = Math.max(0, Math.min(3, LOCAL_Y_EDGES.findIndex((_, index) => index < 4 && localY <= LOCAL_Y_EDGES[index + 1])));
  const column = Math.max(0, Math.min(3, X_EDGES.findIndex((_, index) => index < 4 && position.x_m <= X_EDGES[index + 1])));
  return { side, row, column };
}
function sameCell(cell, target) {
  return target && cell.side === target.side && Number(cell.row) === target.row && Number(cell.column) === target.column;
}

function buildCourt() {
  const court = $("#court");
  court.innerHTML = "";
  for (let visualRow = 0; visualRow < 8; visualRow += 1) {
    for (let column = 0; column < 4; column += 1) {
      const side = visualRow < 4 ? "B" : "A";
      const row = side === "B" ? visualRow : 7 - visualRow;
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "court-cell";
      Object.assign(cell.dataset, { side, row: String(row), column: String(column) });
      cell.style.setProperty("--court-row", String(column + 1));
      cell.style.setProperty("--court-column", String(side === "A" ? row + 1 : 8 - row));
      cell.setAttribute("aria-label", `선수 ${side} 쪽 코트, 기준선에서 ${row + 1}번째 줄, ${column + 1}번째 칸`);
      cell.addEventListener("click", () => chooseCourtCell(side, row, column));
      court.appendChild(cell);
    }
  }
}

function chooseCourtCell(side, row, column) {
  hideError();
  if (state.mode === "example") return showError("기록 예시는 먼저 '마지막 타구 취소'를 눌러 빌더로 전환해 주세요.");
  if (state.activeTarget === "a") {
    if (side !== "A") return showError("선수 A는 아래쪽 코트에서 선택해 주세요.");
    state.draft.player_a_position = cellCenter(side, row, column);
    state.activeTarget = "b";
  } else if (state.activeTarget === "b") {
    if (side !== "B") return showError("선수 B는 위쪽 코트에서 선택해 주세요.");
    state.draft.player_b_position = cellCenter(side, row, column);
    state.activeTarget = "target";
  } else {
    const requiredSide = state.draft.hitter === "A" ? "B" : "A";
    if (side !== requiredSide) return showError(`선수 ${state.draft.hitter}의 타구는 상대 코트에 착지해야 합니다.`);
    state.draft.target_zone = targetZone(state.draft.hitter, row, column);
  }
  render();
}

function renderCourt() {
  const aCell = nearestCell(state.draft.player_a_position, "A");
  const bCell = nearestCell(state.draft.player_b_position, "B");
  const targetCell = state.draft.target_zone > 0 ? zoneCell(state.draft.hitter, state.draft.target_zone) : null;
  const selected = state.analysis && state.selectedResult !== null ? state.analysis.timeline[state.selectedResult] : null;
  const recommendations = new Map();
  if (selected) selected.cql.top_actions.forEach((action) => {
    const cell = zoneCell(selected.hitter, action.target_zone);
    recommendations.set(`${cell.side}-${cell.row}-${cell.column}`, action.rank);
  });
  $$(".court-cell").forEach((element) => {
    const cell = { side: element.dataset.side, row: Number(element.dataset.row), column: Number(element.dataset.column) };
    element.className = "court-cell";
    element.innerHTML = "";
    const validSide = state.activeTarget === "a" ? "A" : state.activeTarget === "b" ? "B" : state.draft.hitter === "A" ? "B" : "A";
    element.disabled = state.mode === "example" || cell.side !== validSide || state.draft.target_zone === 0;
    if (sameCell(cell, aCell)) {
      element.classList.add("selected-a");
      element.insertAdjacentHTML("beforeend", '<span class="marker player-marker a">A</span>');
    }
    if (sameCell(cell, bCell)) {
      element.classList.add("selected-b");
      element.insertAdjacentHTML("beforeend", '<span class="marker player-marker b">B</span>');
    }
    if (sameCell(cell, targetCell)) {
      element.classList.add("selected-target");
      element.insertAdjacentHTML("beforeend", '<span class="marker shuttle">●</span>');
    }
    const hitterForSide = cell.side === "B" ? "A" : "B";
    element.insertAdjacentHTML("beforeend", `<span class="zone-label">Z${targetZone(hitterForSide, cell.row, cell.column)}</span>`);
    const rank = recommendations.get(`${cell.side}-${cell.row}-${cell.column}`);
    if (rank) element.insertAdjacentHTML("beforeend", `<span class="recommendation-rank">${rank}</span>`);
  });
}

function renderShotOptions() {
  const select = $("#shot-type");
  const choices = state.strokes.length === 0 ? SERVICE_SHOTS : RALLY_SHOTS;
  const value = state.draft.shot_type;
  select.innerHTML = '<option value="">타구를 선택하세요</option>' + choices.map((shot) =>
    `<option value="${shot}">${SHOT_LABELS[shot]} · ${shot}</option>`).join("");
  if (choices.includes(value)) select.value = value;
  else state.draft.shot_type = "";
  select.disabled = state.mode === "example" || rallyEnded();
}
const formatPosition = (position) => position ? `x ${position.x_m.toFixed(2)} · y ${position.y_m.toFixed(2)}m` : "선택 필요";

function renderDraft() {
  $("#hitter-value").textContent = state.draft.hitter;
  $("#delta-time").value = String(state.draft.delta_time_s);
  $("#delta-time").disabled = state.mode === "example" || state.strokes.length === 0 || rallyEnded();
  $("#a-position").textContent = formatPosition(state.draft.player_a_position);
  $("#b-position").textContent = formatPosition(state.draft.player_b_position);
  $("#target-zone").textContent = state.draft.target_zone === 0 ? "아웃/네트 · Z0" : state.draft.target_zone ? `Z${state.draft.target_zone}` : "선택 필요";
  $("#stroke-count").textContent = `${state.strokes.length}타구`;
  $("#mode-badge").textContent = state.mode === "example" ? "기록 예시" : state.strokes.length ? "랠리 빌더" : "새 랠리";
  const add = $("#add-stroke");
  add.disabled = state.mode === "example" || rallyEnded();
  add.textContent = state.mode === "example" ? "예시 탐색 중" : rallyEnded() ? "랠리가 종료되었습니다" : "타구 추가하고 분석";
  $("#out-net").disabled = state.mode === "example" || rallyEnded();
  $("#undo-stroke").disabled = state.strokes.length === 0;
  $$(".selection-tab").forEach((button) => button.classList.toggle("active", button.dataset.target === state.activeTarget));
  renderShotOptions();
}

const contextLabel = (source) => ({ exact_context: "동일 직전 타구·선수 위치", previous_shot: "직전 타구 기준 보완", global: "전체 행동 기준 보완" }[source] || source);
function renderResults() {
  const hasResult = state.analysis && state.analysis.timeline.length;
  $("#empty-result").hidden = Boolean(hasResult);
  $("#result-content").hidden = !hasResult;
  if (!hasResult) {
    $("#result-stroke").textContent = "입력 대기";
    $("#result-stroke").className = "badge muted";
    return;
  }
  const index = state.selectedResult ?? state.analysis.timeline.length - 1;
  const result = state.analysis.timeline[index];
  $("#result-stroke").textContent = `${result.stroke_no}번째 타구`;
  $("#result-stroke").className = "badge";
  $("#probability-value").textContent = `${(result.gru.player_a_win_probability * 100).toFixed(1)}%`;
  const delta = result.gru.probability_delta;
  $("#probability-delta").textContent = delta === null ? "첫 예측" : `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)}%p`;
  $("#probability-delta").style.background = delta !== null && delta < 0 ? "#f5b5af" : "var(--lime)";
  renderChart(index);
  $("#recommendations").innerHTML = result.cql.top_actions.map((action) => `
    <div class="recommendation"><span class="rank">${action.rank}</span>
      <div><strong>${escapeHtml(SHOT_LABELS[action.shot_type])} · Z${action.target_zone}</strong><small>${escapeHtml(action.shot_type)} · 학습 ${action.train_count.toLocaleString()}회${action.context_train_count === null ? "" : ` · 동일 맥락 ${action.context_train_count}회`}</small></div>
      <span class="q">Q ${action.q_value.toFixed(3)}</span></div>`).join("");
  const actual = `${SHOT_LABELS[result.shot_type]} · ${result.target_zone === 0 ? "Z0" : `Z${result.target_zone}`}`;
  $("#actual-comparison").innerHTML = result.cql.actual_action_supported
    ? `<strong>실제 선택:</strong> ${escapeHtml(actual)} · CQL ${result.cql.actual_rank}위 · Q ${result.cql.actual_q.toFixed(3)}<br><strong>support:</strong> ${contextLabel(result.cql.context_source)}`
    : `<strong>실제 선택:</strong> ${escapeHtml(actual)}<br>이 상황에서 학습 근거가 부족해 Q 순위를 표시하지 않습니다.`;
  $("#result-warnings").innerHTML = result.warnings.map((warning) => `<p>• ${escapeHtml(warning)}</p>`).join("");
}

function renderChart(selectedIndex) {
  const values = state.analysis.timeline.map((item) => item.gru.player_a_win_probability);
  const width = 500, height = 140, left = 24, right = 10, top = 10, bottom = 25;
  const usableWidth = width - left - right, usableHeight = height - top - bottom;
  const x = (index) => values.length === 1 ? left + usableWidth / 2 : left + index * usableWidth / (values.length - 1);
  const y = (value) => top + (1 - value) * usableHeight;
  const points = values.map((value, index) => `${x(index)},${y(value)}`).join(" ");
  const circles = values.map((value, index) => `<circle cx="${x(index)}" cy="${y(value)}" r="${index === selectedIndex ? 5 : 3}" fill="${index === selectedIndex ? "#c9f05a" : "#0c6b50"}" stroke="#fffdf7" stroke-width="2" />`).join("");
  $("#probability-chart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${values.length}개 타구의 승리확률 추이">
    <line x1="${left}" y1="${y(.5)}" x2="${width - right}" y2="${y(.5)}" stroke="#bfc9c2" stroke-dasharray="5 5" /><text x="0" y="${y(.5) + 4}" font-size="10" fill="#64746f">50</text>
    <polyline points="${points}" fill="none" stroke="#0c6b50" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />${circles}
    <text x="${left}" y="${height - 5}" font-size="10" fill="#64746f">1타구</text><text x="${width - right}" y="${height - 5}" text-anchor="end" font-size="10" fill="#64746f">${values.length}타구</text></svg>`;
}

function renderTimeline() {
  if (!state.analysis || !state.analysis.timeline.length) {
    $("#timeline").innerHTML = '<p class="timeline-empty">아직 추가된 타구가 없습니다.</p>';
    return;
  }
  $("#timeline").innerHTML = state.analysis.timeline.map((item, index) => {
    const recommendation = item.cql.top_actions[0];
    return `<button type="button" class="timeline-item ${index === state.selectedResult ? "active" : ""}" data-index="${index}">
      <span class="stroke-index">${item.stroke_no}</span><span><strong>${item.hitter} · ${escapeHtml(SHOT_LABELS[item.shot_type])} → ${item.target_zone === 0 ? "Z0" : `Z${item.target_zone}`}</strong><small>CQL ${escapeHtml(SHOT_LABELS[recommendation.shot_type])} → Z${recommendation.target_zone}</small></span>
      <span class="timeline-probability">${(item.gru.player_a_win_probability * 100).toFixed(1)}%</span></button>`;
  }).join("");
  $$(".timeline-item").forEach((button) => button.addEventListener("click", () => {
    state.selectedResult = Number(button.dataset.index);
    render();
  }));
}

function render() { renderDraft(); renderCourt(); renderResults(); renderTimeline(); }
function rallyEnded() { return state.strokes.length > 0 && state.strokes.at(-1).target_zone === 0; }
function requestPayload(strokes = state.strokes) {
  return { player_a_name: $("#player-a-name").value.trim() || "Player A", player_b_name: $("#player-b-name").value.trim() || "Player B", strokes };
}
async function analyze(strokes) {
  const response = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(requestPayload(strokes)) });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error([body.detail, ...(body.errors || [])].filter(Boolean).join("\n") || "분석 요청에 실패했습니다.");
  }
  return response.json();
}

async function addStroke() {
  hideError();
  if (!state.draft.player_a_position) return showError("코트에서 선수 A 위치를 선택해 주세요.");
  if (!state.draft.player_b_position) return showError("코트에서 선수 B 위치를 선택해 주세요.");
  if (!state.draft.shot_type) return showError("타구 종류를 선택해 주세요.");
  if (state.draft.target_zone === null) return showError("상대 코트의 셔틀 착지 구역을 선택해 주세요.");
  const delta = Number($("#delta-time").value);
  if (!Number.isFinite(delta) || delta < 0 || (state.strokes.length > 0 && delta <= 0)) return showError("경과 시간을 올바르게 입력해 주세요.");
  const stroke = { ...state.draft, delta_time_s: delta };
  const nextStrokes = [...state.strokes, stroke];
  $("#add-stroke").disabled = true;
  $("#add-stroke").textContent = "두 모델 분석 중…";
  try {
    const result = await analyze(nextStrokes);
    state.strokes = nextStrokes;
    state.analysis = result;
    state.selectedResult = result.timeline.length - 1;
    state.mode = "builder";
    if (stroke.target_zone !== 0) {
      state.draft = freshDraft(stroke.hitter === "A" ? "B" : "A", false, { player_a_position: { ...stroke.player_a_position }, player_b_position: { ...stroke.player_b_position } });
      state.activeTarget = "a";
    }
    render();
  } catch (error) { showError(error.message); render(); }
}

async function loadExamples(query = "") {
  const response = await fetch(`/api/examples?limit=200&query=${encodeURIComponent(query)}`);
  if (!response.ok) throw new Error("기록 랠리 목록을 불러오지 못했습니다.");
  state.examples = await response.json();
  $("#example-select").innerHTML = '<option value="">선수 또는 경기를 선택하세요</option>' + state.examples.map((item) =>
    `<option value="${escapeHtml(item.rally_id)}">${escapeHtml(item.label)} · ${escapeHtml(item.tournament)}</option>`).join("");
}
async function loadExample() {
  hideError();
  const rallyId = $("#example-select").value;
  if (!rallyId) return showError("불러올 기록 랠리를 선택해 주세요.");
  try {
    const response = await fetch(`/api/examples/${encodeURIComponent(rallyId)}`);
    if (!response.ok) throw new Error("예시 랠리를 불러오지 못했습니다.");
    const detail = await response.json();
    $("#player-a-name").value = detail.request.player_a_name;
    $("#player-b-name").value = detail.request.player_b_name;
    state.strokes = detail.request.strokes;
    state.analysis = await analyze(state.strokes);
    state.selectedResult = state.analysis.timeline.length - 1;
    state.mode = "example";
    const last = state.strokes.at(-1);
    state.draft = freshDraft(last.hitter === "A" ? "B" : "A", false, { player_a_position: { ...last.player_a_position }, player_b_position: { ...last.player_b_position } });
    state.activeTarget = "a";
    render();
  } catch (error) { showError(error.message); }
}

function undoStroke() {
  hideError();
  if (!state.strokes.length) return;
  const removed = state.strokes.at(-1);
  state.strokes = state.strokes.slice(0, -1);
  state.mode = "builder";
  if (state.strokes.length) {
    analyze(state.strokes).then((result) => {
      state.analysis = result;
      state.selectedResult = result.timeline.length - 1;
      const last = state.strokes.at(-1);
      state.draft = freshDraft(last.hitter === "A" ? "B" : "A", false, { player_a_position: { ...removed.player_a_position }, player_b_position: { ...removed.player_b_position } });
      render();
    }).catch((error) => showError(error.message));
  } else {
    state.analysis = null;
    state.selectedResult = null;
    state.draft = freshDraft(removed.hitter, true, { player_a_position: { ...removed.player_a_position }, player_b_position: { ...removed.player_b_position } });
    render();
  }
}

function resetRally() {
  state.strokes = []; state.analysis = null; state.selectedResult = null; state.mode = "new"; state.activeTarget = "a"; state.draft = freshDraft("A", true);
  $("#player-a-name").value = "Player A"; $("#player-b-name").value = "Player B"; $("#example-select").value = "";
  hideError(); render();
}
function showError(message) { const element = $("#error-message"); element.textContent = message; element.hidden = false; }
function hideError() { $("#error-message").hidden = true; }

async function initialize() {
  buildCourt(); render();
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    if (!response.ok) throw new Error(health.error || "모델 로딩 실패");
    $("#model-status").classList.add("ready");
    $("#model-status").innerHTML = `<span></span> ${escapeHtml(health.cql_version)} · ${health.cql_actions} actions · ${escapeHtml(health.device.toUpperCase())}`;
    await loadExamples();
  } catch (error) {
    $("#model-status").classList.add("error");
    $("#model-status").innerHTML = "<span></span> 모델 연결 실패";
    showError(error.message);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $$(".selection-tab").forEach((button) => button.addEventListener("click", () => { state.activeTarget = button.dataset.target; hideError(); render(); }));
  $("#shot-type").addEventListener("change", (event) => { state.draft.shot_type = event.target.value; });
  $("#delta-time").addEventListener("change", (event) => { state.draft.delta_time_s = Number(event.target.value); });
  $("#out-net").addEventListener("click", () => { state.draft.target_zone = state.draft.target_zone === 0 ? null : 0; render(); });
  $("#add-stroke").addEventListener("click", addStroke);
  $("#undo-stroke").addEventListener("click", undoStroke);
  $("#new-rally").addEventListener("click", resetRally);
  $("#load-example").addEventListener("click", loadExample);
  let searchTimer;
  $("#example-search").addEventListener("input", (event) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => loadExamples(event.target.value).catch((error) => showError(error.message)), 250);
  });
  initialize();
});
