const ENERGY_LABELS = {
  distance: '기존 거리 비용', M0_distance_reference: 'BFMD 경로 비용',
  M3_hypothesis_speed_only_1: 'BFMD 속력 가설', M3_hypothesis_curve_only_1: 'BFMD 방향전환 가설',
  M3_hypothesis_balanced_025: 'BFMD 균형 가설',
};
let energyMetadata = null;
let energyResult = null;
let energyRequestVersion = 0;

async function energyFetch(url, payload) {
  const response = await fetch(url, payload ? { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) } : {});
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : (body.errors || []).join(' · ') || '입력을 확인해 주세요.');
  return body;
}
function energyPayload() {
  return { history: [1,2,3].map(i => Number($(`#energy-z${i}`).value)),
    current_hitter: $('#energy-hitter').value, strategy_player: $('#energy-player').value,
    cost_model: $('#energy-model').value, initial_a: Number($('#energy-initial-a').value),
    initial_b: Number($('#energy-initial-b').value), remaining_a: Number($('#energy-remaining-a').value),
    remaining_b: Number($('#energy-remaining-b').value) };
}
function energyInvalidate() {
  energyRequestVersion += 1;
  energyResult = null;
  $('#energy-summary').textContent = '';
  $('#energy-top').textContent = '';
  $('#energy-comparison').textContent = '';
  $('#energy-heatmap').textContent = '';
  $('#energy-cell-detail').textContent = '입력이 변경되었습니다. 전략을 다시 계산하세요.';
  $('#energy-status').textContent = energyMetadata ? '입력이 변경되었습니다. 전략을 다시 계산하세요.' : '전략 자료 확인 중';
}
function energyRender() {
  const r = energyResult;
  if (!r) return;
  $('#energy-status').textContent = r.reason || (r.numerical_validation_passed ? '대표 질의 수치 검증 통과 · 목표 구획 도착 조건부 전략' : '수치 검증 미완료 · 모형 실험 결과');
  if (!r.available) return;
  $('#energy-cell-detail').textContent = '구획을 선택하면 예상 이동비용과 지원 표본을 확인할 수 있습니다.';
  $('#energy-summary').innerHTML = `<div class="probability-card"><div><span>${escapeHtml(r.strategy_player)} 전략 모형 가치</span><strong>${(r.model_value*100).toFixed(1)}%</strong></div><span>기존 정책 ${(r.behavior_value*100).toFixed(1)}%<br>차이 ${(r.delta_vs_behavior*100).toFixed(1)}%p</span></div><p>동시 소진 가정 범위 ${(r.lower*100).toFixed(1)}–${(r.upper*100).toFixed(1)}% · 통계적 신뢰구간이 아닙니다.</p>`;
  const selected = $('#energy-shot').value;
  const cells = selected ? r.actions.filter(a => a.shot_type === selected) : r.cells;
  const map = $('#energy-heatmap'); map.textContent = '';
  const ordered = [...cells].sort((a,b) => {
    // Same screen convention as the existing full court: B baseline above;
    // A baseline below. B's horizontal mirror comes from the backend geometry.
    const ar = r.current_hitter === 'A' ? a.screen_row : 3-a.screen_row;
    const br = r.current_hitter === 'A' ? b.screen_row : 3-b.screen_row;
    return ar-br || a.screen_column-b.screen_column;
  });
  ordered.forEach(c => {
    const cell = document.createElement('button'); cell.type = 'button'; cell.className = 'energy-cell';
    const chosen = r.top_actions.some(a => a.target_zone === c.target_zone && a.shot_type === c.shot_type);
    if (chosen) cell.classList.add('recommended');
    if (c.supported) {
      cell.style.backgroundColor = `hsl(${200-45*c.value} 64% ${94-55*c.value}%)`;
      cell.style.color = c.value > .65 ? '#fff' : '#102b3a';
      cell.innerHTML = `<small>Z${c.target_zone}</small><strong>${escapeHtml(SHOT_LABELS[c.shot_type])}</strong><span>${(c.value*100).toFixed(1)}%</span>`;
      cell.setAttribute('aria-label', `Z${c.target_zone}, ${SHOT_LABELS[c.shot_type]}, 모형 가치 ${(c.value*100).toFixed(1)}퍼센트`);
      cell.addEventListener('click', () => {
        const own = r.strategy_player === 'A' ? c.cost_a : c.cost_b;
        const opp = r.strategy_player === 'A' ? c.cost_b : c.cost_a;
        const bfmd = c.bfmd ? ` · BFMD 직접 상태 0건 · A 비용군 ${c.bfmd.A.n}건(level ${c.bfmd.A.level}) / B ${c.bfmd.B.n}건(level ${c.bfmd.B.level})${c.bfmd.A.extrapolated || c.bfmd.B.extrapolated ? ' · 거리 외삽' : ''}` : '';
        $('#energy-cell-detail').textContent = `Z${c.target_zone} · ${SHOT_LABELS[c.shot_type]} · 나 ${own.toFixed(2)} / 상대 ${opp.toFixed(2)} J/kg · 기존 정책 대비 ${(c.delta_vs_behavior*100).toFixed(2)}%p · ${c.support_order}타구 문맥 ${c.train_count}건 / 유효 비용 ${c.valid_cost_count}건${bfmd}`;
      });
    } else {
      cell.classList.add('unsupported'); cell.textContent = `Z${c.target_zone} · 지원 부족`;
      cell.addEventListener('click', () => { $('#energy-cell-detail').textContent = `Z${c.target_zone}: 이 타구·구획 조합의 표본 지원이 부족해 추천을 보류합니다.`; });
    }
    map.appendChild(cell);
  });
  const incoming = energyPayload().history.at(-1);
  $('#energy-top').innerHTML = r.top_actions.length ? '<h3>추천 방향 · 상위 3개</h3>' + r.top_actions.map(a => `<p><strong>${a.rank}. incoming Z${incoming} → ${escapeHtml(SHOT_LABELS[a.shot_type])} → Z${a.target_zone}</strong> · ${(a.value*100).toFixed(1)}%</p>`).join('') : `<p>${escapeHtml(r.reason || '추천할 후보가 없습니다.')}</p>`;
}
async function energyCalculate() {
  energyInvalidate(); const version = energyRequestVersion;
  $('#energy-status').textContent = '전략 계산 중';
  try {
    const result = await energyFetch('/api/energy-strategy', energyPayload());
    if (version !== energyRequestVersion) return;
    energyResult = result; energyRender();
  } catch(e) { if (version === energyRequestVersion) $('#energy-status').textContent = e.message; }
}
async function energyFromRally() {
  energyInvalidate(); const version = energyRequestVersion;
  if (!state.strokes.length) { $('#energy-status').textContent = '랠리를 먼저 입력하거나 불러오세요.'; return; }
  try {
    const p = energyPayload();
    const body = await energyFetch('/api/energy-strategy/from-rally', { rally: requestPayload(), stroke_index: state.selectedResult,
      cost_model: p.cost_model, initial_a: p.initial_a, initial_b: p.initial_b });
    if (version !== energyRequestVersion) return;
    [1,2,3].forEach((i) => { $(`#energy-z${i}`).value = body.history[i-1] ?? 0; });
    $('#energy-hitter').value = body.current_hitter;
    $('#energy-remaining-a').value = Math.min(p.initial_a,Number(body.remaining_a.toFixed(3)));
    $('#energy-remaining-b').value = Math.min(p.initial_b,Number(body.remaining_b.toFixed(3)));
    $('#energy-status').textContent = `선택 타구 직전 소비량 A ${body.consumed_a.toFixed(2)} / B ${body.consumed_b.toFixed(2)} J/kg · 초기 구간은 위치 거리 근사`;
  } catch(e) { $('#energy-status').textContent = e.message; }
}
async function energyCompare() {
  const version = energyRequestVersion, payload = energyPayload();
  if (!energyMetadata) return;
  $('#energy-comparison').textContent = '비용 모형 비교 중';
  const rows = [], results = new Map();
  for (const model of energyMetadata.models) {
    try {
      const r = await energyFetch('/api/energy-strategy', {...payload, cost_model:model});
      results.set(model,r);
      const top = r.top_actions[0];
      rows.push(`<tr><td>${escapeHtml(ENERGY_LABELS[model])}</td><td>${r.available ? (r.model_value*100).toFixed(1)+'%' : '—'}</td><td>${top ? escapeHtml(SHOT_LABELS[top.shot_type])+' → Z'+top.target_zone : escapeHtml(r.reason)}</td><td>${r.numerical_validation_passed ? '통과' : '미완료'}</td></tr>`);
    } catch(e) { rows.push(`<tr><td>${escapeHtml(ENERGY_LABELS[model])}</td><td colspan="3">${escapeHtml(e.message)}</td></tr>`); }
  }
  if (version !== energyRequestVersion) return;
  $('#energy-comparison').innerHTML = `<table class="energy-table"><thead><tr><th>비용 기준</th><th>모형 가치</th><th>추천</th><th>수치 검증</th></tr></thead><tbody>${rows.join('')}</tbody></table>`;
  const baseline = results.get(payload.cost_model);
  if (!baseline?.available) return;
  const title = document.createElement('p'); title.textContent = '구획별 타구 차이 · 주황색은 비용 모형에 따라 최선의 타구가 바뀌는 구획';
  const differences = document.createElement('div'); differences.className = 'energy-cost-differences';
  const comparisonCells = [...baseline.cells].sort((a,b)=> {
    const ar=payload.current_hitter==='A'?a.screen_row:3-a.screen_row;
    const br=payload.current_hitter==='A'?b.screen_row:3-b.screen_row;
    return ar-br || a.screen_column-b.screen_column;
  });
  comparisonCells.forEach(cell => {
    const variants = [...results].filter(([,r])=>r.available).map(([model,r])=>({model,cell:r.cells.find(c=>c.target_zone===cell.target_zone)}));
    const changed = variants.some(v=>v.cell?.shot_type!==cell.shot_type);
    const button = document.createElement('button'); button.type = 'button';
    button.className = 'energy-diff-item'+(changed?' changed':'');
    button.textContent = `Z${cell.target_zone} · ${changed?'타구 변경':'동일'}`;
    const detail = variants.map(v=>`${ENERGY_LABELS[v.model]}: ${v.cell?.supported?SHOT_LABELS[v.cell.shot_type]+' '+(v.cell.value*100).toFixed(1)+'%':'지원 부족'}`).join(' / ');
    button.title = detail; button.addEventListener('click',()=>{ $('#energy-cell-detail').textContent = `Z${cell.target_zone} · ${detail}`; });
    differences.appendChild(button);
  });
  $('#energy-comparison').append(title,differences);
}
document.addEventListener('DOMContentLoaded', async () => {
  for (const tab of ['prediction','energy']) $(`#${tab}-tab`).addEventListener('click', () => {
    for (const id of ['prediction','energy']) {
      const active = id === tab; $(`#${id}-view`).hidden = !active;
      $(`#${id}-tab`).classList.toggle('active',active); $(`#${id}-tab`).setAttribute('aria-selected',String(active));
    }
  });
  RALLY_SHOTS.forEach(shot => { const option = document.createElement('option'); option.value = shot; option.textContent = SHOT_LABELS[shot]; $('#energy-shot').appendChild(option); });
  $('#energy-shot').addEventListener('change',energyRender);
  $('#energy-calculate').addEventListener('click',energyCalculate);
  $('#energy-from-rally').addEventListener('click',energyFromRally);
  $('#energy-compare').addEventListener('click',energyCompare);
  document.addEventListener('rallychange',energyInvalidate);
  $$('#energy-view input, #energy-view select').filter(e => e.id !== 'energy-shot').forEach(e => e.addEventListener('change',energyInvalidate));
  $('#energy-scenario').addEventListener('change', () => {
    if (!energyMetadata) return;
    const budget = energyMetadata.budgets[{50:0,75:1,95:2}[$('#energy-scenario').value]];
    for (const id of ['initial-a','initial-b','remaining-a','remaining-b']) $(`#energy-${id}`).value = Number(budget.toFixed(3));
  });
  try {
    energyMetadata = await energyFetch('/api/energy-strategy/metadata');
    $('#energy-model').innerHTML = energyMetadata.models.map(m => `<option value="${escapeHtml(m)}">${escapeHtml(ENERGY_LABELS[m])}</option>`).join('');
    $('#energy-model').value = 'distance';
    $('#energy-assumptions').innerHTML = energyMetadata.assumptions.map(x => `<p>${escapeHtml(x)}</p>`).join('');
    const budget = energyMetadata.budgets[1];
    for (const id of ['initial-a','initial-b','remaining-a','remaining-b']) $(`#energy-${id}`).value = Number(budget.toFixed(3));
    $('#energy-status').textContent = '이력과 예산을 입력하고 전략 계산을 누르세요.';
  } catch(e) { $('#energy-status').textContent = e.message; }
});
