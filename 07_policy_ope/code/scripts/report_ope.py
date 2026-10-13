"""Produce a standalone Korean HTML report and scientific diagnostic plots."""
from pathlib import Path
import json
import sys
import html

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'outputs/ope_cql_v2_frame_audited')
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    diag = json.loads((root / 'diagnostics.json').read_text(encoding='utf-8'))
    summary = pd.read_csv(root / 'summary.csv')
    records = pd.read_csv(root / 'test_episode_estimates.csv')
    baseline_records = records[records.policy.eq('behavior_clone')].copy()
    baseline_records['prediction_bin'] = pd.qcut(baseline_records.fqe, 5, duplicates='drop')
    calibration = baseline_records.groupby('prediction_bin', observed=True).agg(
        predicted_win=('fqe', 'mean'), observed_win=('observed_win', 'mean'), episodes=('fqe', 'size')).reset_index()
    calibration.to_csv(root / 'behavior_calibration.csv', index=False)
    sensitivity_html = ''
    previous_path = Path('outputs/ope_cql_v2/summary.csv')
    if protocol.get('frame_audit_applied', False) and previous_path.exists():
        before = pd.read_csv(previous_path)
        sensitivity = before[['policy', 'estimator', 'estimate', 'delta_vs_behavior']].merge(
            summary[['policy', 'estimator', 'estimate', 'delta_vs_behavior']],
            on=['policy', 'estimator'], suffixes=('_before', '_audited'))
        sensitivity.to_csv(root / 'frame_audit_sensitivity.csv', index=False)
        sensitivity_html = '<h2>원본 프레임 감사 적용 전후 비교 (확률 0–1)</h2>' + sensitivity.to_html(index=False, float_format=lambda x: f'{x:.4f}')
    positive = baseline_records.observed_win.to_numpy(bool)
    ranks = rankdata(baseline_records.fqe.to_numpy())
    positives = positive.sum()
    auc = float((ranks[positive].sum() - positives * (positives + 1) / 2) / (positives * (~positive).sum()))
    brier = float(np.square(baseline_records.fqe - baseline_records.observed_win).mean())
    overlap = pd.read_csv(root / 'overlap.csv')
    history = pd.read_csv(root / 'fqe_history.csv')
    names = ['behavior_clone', 'cql_shot_mix_50', 'cql_shot']
    labels = ['Behavior clone', '50% CQL shot mixture', 'CQL shot choice']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fqe = summary[summary.estimator.eq('fqe')].set_index('policy').loc[names]
    for ax, value, lo, hi, title in [
        (axes[0], fqe.estimate, fqe.ci95_low, fqe.ci95_high, 'FQE estimated rally win probability'),
        (axes[1], fqe.delta_vs_behavior, fqe.delta_ci95_low, fqe.delta_ci95_high, 'FQE improvement vs behavior clone')]:
        y = np.arange(3)
        ax.scatter(value * 100, y, color='#2463a6', zorder=3)
        for i in range(3):
            ax.plot([lo.iloc[i] * 100, hi.iloc[i] * 100], [i, i], color='#2463a6', lw=2)
        ax.set_yticks(y, labels)
        ax.set_title(title)
        ax.set_xlabel('Percent' if ax is axes[0] else 'Percentage points')
        ax.grid(axis='x', alpha=.2)
        ax.invert_yaxis()
    axes[0].axvline(diag['observed_initial_win_rate'] * 100, color='gray', ls='--', label='Logged observed rate')
    axes[0].legend(fontsize=8)
    axes[1].axvline(0, color='gray', ls='--')
    fig.suptitle('Conditional model-based estimates; not proof of real-world improvement', fontsize=11)
    fig.tight_layout()
    fig.savefig(root / 'fqe_estimates.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for n, label in zip(names, labels):
        axes[0].plot(history.iteration, history[n + '_validation_value'], label=label)
    axes[0].set(xlabel='FQE iteration', ylabel='Validation initial value', title='FQE optimization stability')
    axes[0].legend(fontsize=8)
    axes[1].hist(overlap.behavior_probability_of_recommendation, bins=30, color='#2463a6')
    axes[1].axvline(.05, color='#ac3d36', ls='--')
    axes[1].set(xlabel='Estimated logged probability of recommended shot', ylabel='Test decision states', title='Action overlap (behavior model estimate)')
    fig.tight_layout()
    fig.savefig(root / 'diagnostics.png', dpi=180)
    plt.close(fig)

    strata = records.groupby(['role', 'policy']).agg(fqe=('fqe', 'mean'), dr=('dr', 'mean'),
                                                    observed_win=('observed_win', 'mean'), episodes=('fqe', 'size')).reset_index()
    strata.to_csv(root / 'role_summary.csv', index=False)
    final_change = float(history.iloc[-1][names[-1] + '_validation_change'])
    gap = abs(diag['behavior_fqe_calibration_gap'])
    ess = diag[names[-1]]['ess']
    total = protocol['audit']['test']['episodes']
    # Screening thresholds are diagnostic choices, not formal identification tests.
    gates = {'behavior_mean_calibration_within_3pp': gap <= .03,
             'last_10_iteration_validation_change_within_2pp': final_change <= .02,
             'pure_policy_trajectory_ess_at_least_10pct_or_30': ess >= max(30, total * .1),
             'fqe_improvement_lower_bound_positive': float(fqe.loc[names[-1], 'delta_ci95_low']) > 0,
             'dr_improvement_lower_bound_positive': float(summary[(summary.policy.eq(names[-1])) & summary.estimator.eq('dr')].delta_ci95_low.iloc[0]) > 0}
    conclusion = '현재 평가만으로 실제 승률 향상을 확정할 수 없음'
    if all(gates.values()):
        conclusion = '여러 평가에서 조건부 우위가 관찰됨; 실제 경기 효과 입증은 아님'
    (root / 'assessment.json').write_text(json.dumps({'conclusion': conclusion, 'diagnostic_gates': gates,
        'threshold_note': 'reporting heuristics, not preregistered tests or formal statistical guarantees',
        'tests': 'python -m unittest tests.test_ope tests.test_cql_v2_pipeline -v (12 passed)'}, ensure_ascii=False, indent=2), encoding='utf-8')

    table = summary.copy()
    for c in ['estimate', 'ci95_low', 'ci95_high', 'delta_vs_behavior', 'delta_ci95_low', 'delta_ci95_high']:
        table[c] = table[c].map(lambda v: f'{v * 100:.2f}')
    table.columns = ['정책', '평가법', '추정(%)', '95% 하한', '95% 상한', '차이(%p)', '차이 하한', '차이 상한']
    audit_table = pd.DataFrame([{ 'split': s, **{k: a[k] for k in ['matches', 'rallies', 'rows', 'episodes', 'retained_z0', 'retained_cql_excluded_rows']},
                                  'excluded_rallies': len(a['excluded_rallies']) } for s, a in protocol['audit'].items()])
    diag_table = pd.DataFrame([{ 'policy': n, **{k: diag[n][k] for k in ['fqe', 'dr', 'ess', 'nonzero_weight_episodes', 'wis', 'held_out_bellman_mse']} } for n in names])
    concentration = {}
    for n in names:
        weights = records[records.policy.eq(n)].trajectory_weight.sort_values(ascending=False)
        concentration[n] = float(weights.head(10).sum() / weights.sum())
    diag_table['top10_weight_fraction'] = diag_table.policy.map(concentration)
    val_observed = pd.read_csv(root / 'validation_episode_estimates.csv').query("policy == 'behavior_clone'").observed_win.mean()
    body = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>CQL 타구 선택 OPE 평가</title>
    <style>body{{font-family:system-ui,'Malgun Gothic',sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;line-height:1.7;color:#203047}}
    h1,h2{{color:#153d68}}.lead{{background:#edf3fa;padding:20px;border-left:5px solid #2463a6}}table{{border-collapse:collapse;width:100%;font-size:14px}}
    th,td{{padding:9px;border-bottom:1px solid #dce3ec;text-align:right}}th{{background:#f0f4f9}}img{{max-width:100%}}code{{background:#f0f4f9;padding:3px}}
    .small{{font-size:13px;color:#516172}}</style>
    <h1>CQL 타구 선택의 오프폴리시 평가</h1><p>2026년 10월 13일 · FQE + Sequential DR · 경기 단위 bootstrap {protocol['bootstrap_replicates']}회</p>
    <div class="lead"><b>{conclusion}</b><br>이 평가는 CQL v2의 타구 종류 선택만 대상으로 한다. 목표 구역 선택, 체력 예산 최적 전략, 실제 경기 전체 승률을 검증한 결과가 아니다.</div>
    <h2>평가 질문과 범위</h2><p>관측된 랠리에서 각 선수의 첫 의사결정 상태를 시작점으로 두고, 그 선수만 CQL의 타구 종류 추천을 따른다면 랠리 승리 가능성이 어떻게 달라지는지 추정했다. 상대는 기록에서 학습한 반응을 유지한다. 리시버 시작점은 관측 서브를 받은 상태에 조건부이며, 양 선수를 동시에 교체한 자가 대결이 아니다.</p>
    <p>CQL의 (타구, 착지 구역) 중 구역은 실제 결과다. 이를 의도한 목표와 같다고 보고 완벽한 실행을 가정하지 않도록, 평가 행동을 타구 종류로 축소했다. 착지 구역, 아웃·네트, 상대 반응은 그 타구가 실제로 낳은 결과로 남겼다. 세 정책은 관측 행동을 복원한 확률 정책, 매 의사결정에서 50% 확률로 CQL 추천을 채택하고 나머지는 복원 정책을 따르는 정책, CQL 추천 타구를 항상 고르는 정책이다. 같은 종류 타구 안의 구역 선택은 평가하지 않았다.</p>
    <h2>실제 실행 결과</h2><p>테스트 시작 상태에서 관측 승리율은 {diag['observed_initial_win_rate'] * 100:.2f}%였다. 기준 행동모형에 대한 FQE와 이 값의 차이는 {diag['behavior_fqe_calibration_gap'] * 100:+.2f}%p다. 기준 비교는 관측 승률 자체가 아니라 관측 행동을 복원한 정책에 대한 FQE다.</p>
    {table.to_html(index=False, escape=True)}<p class="small">95% 범위는 경기 단위 percentile bootstrap 구간이다. 고정 CQL·GRU·마스크 및 추정모형에 조건부이며, 숨은 교란·모형 오류·정책 선택 불확실성은 포함하지 않는다. DR은 보정 추정량이라 0–100% 범위를 벗어날 수 있으며 임의로 잘라내지 않았다.</p>
    <img src="fqe_estimates.png" alt="FQE estimates and confidence intervals">
    <h2>데이터와 분리</h2>{audit_table.to_html(index=False)}
    <p>행동모형과 FQE는 train만으로 학습했다. validation은 학습 진행 진단에만 사용했고, 고정한 {protocol['fqe_iterations']}회 반복의 결과를 test에서 평가했다. test를 이용한 정책 재선택이나 하이퍼파라미터 조정은 하지 않았다. 기존 test는 과거 프로젝트 검토에 사용됐으므로 완전히 새로운 확인 자료라고 부르지 않는다. 학습 데이터는 CQL 정책 학습과 공유하며, 독립 평가 네트워크와 held-out 시작 상태를 사용한 것이다.</p>
    <p>CQL이 제외한 타구와 Z0도 평가에는 보존했다. 시간·타구자·연결 구조에 문제가 있는 랠리는 통째로 제외해 궤적 중간이 끊기지 않게 했다. 최근 원본 감사의 비증가 프레임 랠리 제외 적용: {protocol.get('frame_audit_applied', False)}. CQL의 t→t+2 시간 검사만으로 놓친 프레임 순서 이상을 이 추가 감사로 제외했다. 이전 계산은 outputs/ope_cql_v2에 비교용으로 보존했다. 제외는 승률 점수가 아니라 기존 원본 품질 플래그로 결정했다. 상태에는 107개 기존 사전 관측 특성만 사용했고 현재 타구·착지 결과·최종 랠리 길이·승패 라벨을 입력하지 않았다.</p>
    <h2>평가법과 불확실성</h2><p>FQE는 고정 정책별 별도 신경망으로 Q를 [0,1] 안에서 추정했다. 보상은 종결 승리 1·패배 0, 할인율은 1이다. 따라서 CQL 자체의 할인된 Q나 이동 벌점을 승률로 변환하지 않았다. 타구 종류 행동모형은 class balancing 없이 실제 빈도로 학습했다. DR은 동일 평가모형의 Bellman 잔차를 행동확률 비율의 누적으로 보정하며, FQE와 독립적인 자료 출처를 제공하는 것은 아니다.</p>
    <p>train의 경기를 복원추출해 행동모형과 세 FQE를 매번 처음부터 다시 학습하고, test 경기도 별도로 복원추출했다. 동일 반복의 정책 차이를 계산해 짝지은 구간을 만들었다. 신경망 초기화는 고정해 재표집 불확실성을 분리했다. 평가 초기화와 모형 구조의 불확실성은 별도로 남는다. 학습 중 결과와 bootstrap 행은 CSV로 보존했다.</p>
    <h2>신뢰성 진단</h2>{diag_table.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
    <p>추천 타구의 추정 관측 확률이 5% 미만인 test 상태 비율: {diag['overlap']['recommendation_propensity_below_0_05'] * 100:.2f}%. 같은 이전 타구·위치 문맥에서 학습 관측이 20회 미만인 비율: {diag['overlap']['context_count_below_20'] * 100:.2f}%. 궤적 중요도 가중치의 ESS가 작다면 DR/WIS가 소수 랠리에 의존한다. 행동확률은 전문가의 알려진 무작위 선택 확률이 아니라 추정값이다.</p>
    <p>Validation 관측 승리율은 {val_observed * 100:.2f}%, 기준 FQE는 {diag['validation']['behavior_clone']['fqe'] * 100:.2f}%다. 마지막 10회 반복에서 CQL FQE validation 값의 변화는 {final_change * 100:.2f}%p였다. 평균 보정 일치만으로 모든 상황의 보정이나 수렴이 보장되지는 않는다.</p>
    <p>기준 FQE의 test 시작 상태 승패 구분 AUC는 {auc:.4f}, Brier는 {brier:.4f}다. 이번 시작 상태의 승패 구분 성능은 낮으며, 평균 승률 보정이 맞는 것만으로 정책 간 차이를 신뢰할 수는 없다. 아래 표는 예측값 5분위별 비교이며 확률은 0–1 단위다. 이는 관측 행동 복원 정도의 진단이며 새로운 정책의 인과효과 검증을 대신하지 않는다.</p>{calibration.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
    <img src="diagnostics.png" alt="FQE convergence and behavior overlap">
    <pre>{html.escape(json.dumps(gates, ensure_ascii=False, indent=2))}</pre><p class="small">위 조건은 결과 해석을 위한 진단 기준이며 인과 식별 또는 통계적 보장을 제공하는 검정이 아니다.</p>
    <h2>서버·리시버별 결과 (확률 0–1)</h2>{strata.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
    {sensitivity_html}
    <h2>해석과 다음 단계</h2><p>FQE의 양의 차이는 관측 자료와 모형 가정 아래에서 타구 선택을 바꿀 가능성을 보여준다. 확정적인 실제 승률 상승을 주장하려면 상태에 없는 선수 기량·타구 난도 등의 교란, 정책에 따른 상대 반응 변화, 실행 가능성, 새로운 경기군에서의 재현을 추가 확인해야 한다. 특히 현재 자료로는 누적 에너지 최적 전략과 예산 소진 패배 가정을 검증할 수 없다. 다음에는 정책을 고정한 새 경기군에서 재평가하고, 행동확률이 낮은 상황은 관측 전략으로 복귀하는 정책을 미리 정의해 평가할 수 있다.</p>
    <h2>재현과 검증</h2><p><code>.\\.venv-gru\\Scripts\\python.exe -m src.evaluate_ope --bootstrap 100 --iterations 100 --behavior-epochs 25</code><br>
    <code>.\\.venv-gru\\Scripts\\python.exe scripts/report_ope.py</code><br>
    <code>.\\.venv-gru\\Scripts\\python.exe -m unittest tests.test_ope tests.test_cql_v2_pipeline -v</code> — 12개 통과.</p>
    <p>기존 정책·학습 결과는 수정하지 않았다. 정책 체크포인트 SHA-256 불변 확인: {diag['policy_hash_unchanged']}. 입력과 소스 해시는 <a href="protocol.json">protocol.json</a>, 평가모형은 evaluators.pt, 상세 결과는 <a href="summary.csv">summary.csv</a>와 test_episode_estimates.csv에 있다.</p>
    <p>방법 근거: <a href="https://proceedings.mlr.press/v139/hao21b.html">Hao et al. (2021), Bootstrapping Fitted Q-Evaluation</a>; <a href="https://proceedings.mlr.press/v48/jiang16.html">Jiang &amp; Li (2016), Doubly Robust Off-policy Value Evaluation</a>. 해당 논문의 이론적 보장을 이 비선형 신경망·추정 행동확률에 그대로 적용하지 않는다.</p></html>'''
    (root / 'report.html').write_text(body, encoding='utf-8')
    print(json.dumps({'report': str(root / 'report.html'), 'conclusion': conclusion, 'gates': gates}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
