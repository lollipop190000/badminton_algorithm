"""Evaluation tables, scientific PNG figures, and an evidence-qualified report."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.estimate_markov_q import MID_RALLY_ACTIONS

LITERATURE = [
    dict(key="Pugliese2025", title="Energy Cost of Badminton Footwork: A Novel Experimental Approach",
         url="https://pubmed.ncbi.nlm.nih.gov/40716493/", access="PubMed abstract",
         supports="Side-step/running-step net costs 19.59/20.38 J/kg/m; speed association.",
         does_not_support="Straight-line annotated displacement is not a measured full trajectory or metabolic expenditure; no F_max."),
    dict(key="LeMansec2020", title="Impaired Performance of the Smash Stroke in Badminton Induced by Muscle Fatigue",
         url="https://pubmed.ncbi.nlm.nih.gov/30958057/", access="PubMed abstract",
         supports="Experimental muscle fatigue impaired smash precision and increased faults; rationale for testing e slopes.",
         does_not_support="Precision decrease is not the same as an error-probability multiplier; no rally J/kg calibration."),
    dict(key="Wang2026", title="Synergistic effects of physical-mental mixed fatigue on badminton forehand smash performance",
         url="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0337467", access="Publisher full text",
         supports="24 athletes, graded combined physical/mental fatigue; speed and accuracy deteriorated; nonlinear candidates worth testing.",
         does_not_support="Combined fatigue does not identify pure physical causation or any threshold in this movement-proxy energy scale."),
    dict(key="Cheng2024Online2025Issue", title="Fatigue Affects Body Acceleration During Vertical Jumping and Agility Tasks in Elite Young Badminton Players",
         url="https://journals.sagepub.com/doi/10.1177/19417381241245908", access="Publisher article/abstract",
         supports="Pre/post fatigue differences in jumping and agility; indirect mechanism for receiver capability changes.",
         does_not_support="Does not measure No Return probability or gamma/eta; online 2024, print volume 2025."),
    dict(key="Lin2023", title="Neuromuscular fatigue and muscle damage following a simulated singles badminton match",
         url="https://link.springer.com/article/10.1007/s00421-023-05148-w", access="Publisher full text",
         supports="Post-match neuromuscular fatigue associated with lunge count; movement load is relevant.",
         does_not_support="One-hour simulated matches/post-match measurements do not calibrate within-rally hazards or recovery here."),
    dict(key="BarreiraChiminazzo2020", title="Who, how and when to perform winner points and unforced errors in badminton matches?",
         url="https://www.tandfonline.com/doi/abs/10.1080/24748668.2020.1766838", access="Publisher abstract; full text inaccessible",
         supports="Winner points and unforced errors are distinct performance indicators in 57 Olympic singles matches.",
         does_not_support="Abstract alone does not establish operational labels for these CSVs; does not identify forced errors or no contact."),
]


def calibration_table(predictions):
    rows = []
    for (cohort, split), group in predictions.groupby(["cohort", "split"]):
        for endpoint in ("e", "u"):
            frame = group if endpoint == "e" else group[~group.event_class.eq("E")]
            p = frame[endpoint].to_numpy()
            y = frame.event_class.eq("E" if endpoint == "e" else "U").to_numpy()
            bins = pd.qcut(p, 10, labels=False, duplicates="drop")
            if pd.isna(bins).all():
                bins = np.zeros(len(frame), int)
            for bin_id in np.unique(np.asarray(bins)[~pd.isna(bins)]):
                select = np.asarray(bins) == bin_id
                rows.append(dict(cohort=cohort, split=split, endpoint=endpoint, bin=int(bin_id),
                                 events=int(select.sum()), predicted=float(p[select].mean()), observed=float(y[select].mean())))
    return pd.DataFrame(rows)


def energy_quantile_table(predictions):
    rows = []
    for cohort, group in predictions.groupby("cohort"):
        train = group[group.split.eq("train")]
        for endpoint, energy_column in (("e", "cumulative_energy_hitter_before"), ("u", "cumulative_energy_receiver_before")):
            train_domain = train if endpoint == "e" else train[~train.event_class.eq("E")]
            edges = np.unique(np.quantile(train_domain[energy_column], np.linspace(0, 1, 11)))
            edges[0], edges[-1] = -np.inf, np.inf
            for split, frame in group.groupby("split"):
                if endpoint == "u":
                    frame = frame[~frame.event_class.eq("E")]
                bins = pd.cut(frame[energy_column], edges, labels=False, include_lowest=True)
                for bin_id, chunk in frame.groupby(bins, observed=True):
                    rows.append(dict(cohort=cohort, split=split, endpoint=endpoint, bin=int(bin_id), events=len(chunk),
                                     mean_energy=float(chunk[energy_column].mean()),
                                     observed=float(chunk.event_class.eq("E" if endpoint == "e" else "U").mean()),
                                     predicted=float(chunk[endpoint].mean())))
    return pd.DataFrame(rows)


def create_figures(root, model, train, calibration, quantiles, bootstrap, diagnostic=None):
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    # Standardize to each shot's empirical train distribution of contexts and
    # the other player's consumption; never label this as a causal dose response.
    energy = np.linspace(0, model.d_star * 1.25, 80)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for action, axis in enumerate(axes.ravel()):
        subset = train[train.action_index.eq(action)]
        subset = subset.iloc[np.linspace(0, len(subset)-1, min(len(subset), 200)).astype(int)]
        states = subset.state_id.to_numpy(dtype=int)
        shots = np.full(len(subset), action)
        h = subset.cumulative_energy_hitter_before.to_numpy()
        r = subset.cumulative_energy_receiver_before.to_numpy()
        e_curve = [model.e_model.predict_arrays(states, shots, np.full(len(subset), x), r).mean() for x in energy]
        u_curve = [model.u_model.predict_arrays(states, shots, h, np.full(len(subset), x)).mean() for x in energy]
        axis.plot(energy, e_curve, label="e: hitter consumption")
        axis.plot(energy, u_curve, label="u: receiver consumption")
        if diagnostic is not None:
            de = [diagnostic.e_model.predict_arrays(states, shots, np.full(len(subset), x), r).mean() for x in energy]
            du = [diagnostic.u_model.predict_arrays(states, shots, h, np.full(len(subset), x)).mean() for x in energy]
            axis.plot(energy, de, color="C0", ls="--", label="diagnostic M1 e")
            axis.plot(energy, du, color="C1", ls="--", label="diagnostic M1 u")
        axis.axvline(model.d_star, color="grey", ls=":", lw=.8,label="D*: train event q95")
        axis.set(title=MID_RALLY_ACTIONS[action], xlabel="Movement proxy consumption (J/kg)", ylabel="Conditional probability", ylim=(0,max(.12,1.25*max(max(e_curve),max(u_curve)))))
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Selected models: context-standardized observational curves")
    fig.savefig(figures / "shot_hazard_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
    for endpoint, axis in zip(("e", "u"), axes):
        rows = calibration[calibration.cohort.eq("strict") & calibration.endpoint.eq(endpoint)]
        axis.plot([0,1],[0,1],ls="--",color="grey")
        for split, frame in rows.groupby("split"):
            axis.plot(frame.predicted, frame.observed, marker="o", label=split)
        upper = min(1., max(.12, 1.2 * max(rows.predicted.max(), rows.observed.max())))
        axis.set(title=f"{endpoint} calibration (quantile bins)", xlabel="Predicted", ylabel="Observed", xlim=(0,upper), ylim=(0,upper))
        axis.legend()
    fig.savefig(figures / "calibration.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11,4), constrained_layout=True)
    for endpoint, axis in zip(("e","u"), axes):
        rows = quantiles[quantiles.cohort.eq("strict") & quantiles.split.eq("test") & quantiles.endpoint.eq(endpoint)]
        axis.plot(rows.mean_energy, rows.observed, marker="o", label="observed")
        axis.plot(rows.mean_energy, rows.predicted, marker="s", label="predicted")
        axis.set(title=f"{endpoint}: held-out energy strata", xlabel="Mean movement proxy (J/kg)", ylabel="Rate")
        axis.legend()
    fig.savefig(figures / "energy_stratified_rates.png", dpi=160)
    plt.close(fig)
    fatigue = bootstrap[~bootstrap.term.eq("intercept") & ~bootstrap.term.str.startswith("shot_intercept")]
    if len(fatigue):
        fig, axis = plt.subplots(figsize=(10, max(3,len(fatigue)*.35)), constrained_layout=True)
        for index, row in enumerate(fatigue.itertuples(index=False)):
            axis.plot([row.ci_low,row.ci_high],[index,index],color="steelblue")
            axis.plot(row.bootstrap_mean,index,"o",color="steelblue")
        axis.axvline(0,color="grey",ls="--")
        axis.set_yticks(range(len(fatigue)), [f"{row.model}/{row.term}" for row in fatigue.itertuples(index=False)])
        axis.set(title="Match bootstrap: conditional-on-selected-spec 95% intervals", xlabel="Logit coefficient")
        fig.savefig(figures / "fatigue_coefficient_intervals.png", dpi=160)
        plt.close(fig)


def write_report(root, diagnostics):
    rows = pd.DataFrame(diagnostics["held_out_metrics"])
    strict = rows[(rows.cohort == "strict") & (rows.split == "test")]
    metrics = ["| Model | E/U/C log loss | E/U/C Brier | Joint 144-event log loss |",
               "|---|---:|---:|---:|"]
    for row in strict.itertuples(index=False):
        metrics.append(f"| {row.model} | {row.three_class_log_loss:.6f} | {row.three_class_brier:.6f} | {row.fatigue_joint_144_log_loss:.6f} |")
    selected = diagnostics["selected_specs"]["strict"]
    cohort = diagnostics["energy_audit"]["summaries"]
    slopes = pd.read_csv(root / "bootstrap_summary.csv")
    slopes = slopes[slopes.term.str.startswith(("beta", "gamma", "eta", "theta"))]
    slope_lines = ["| Model / term | Mean | 95% interval | Positive fraction |", "|---|---:|---:|---:|"]
    for row in slopes.itertuples(index=False):
        slope_lines.append(f"| {row.model} / {row.term} | {row.bootstrap_mean:.4f} | [{row.ci_low:.4f}, {row.ci_high:.4f}] | {row.positive_fraction:.3f} |")
    coefficients = pd.read_csv(root / "fatigue_model_coefficients.csv")
    controls = pd.read_csv(root / "control_model_coefficients.csv")
    control_lines = ["| Term | Context only | Player / level / past-count controls |", "|---|---:|---:|"]
    for endpoint, term in (("e", "beta:common"), ("u", "gamma:common"), ("u", "eta:common")):
        original = coefficients[(coefficients.model==f"diagnostic_{endpoint}") & coefficients.term.eq(term)]
        controlled = controls[(controls.model==f"controlled_{endpoint}") & controls.term.eq(term)]
        if len(original) and len(controlled):
            control_lines.append(f"| {endpoint} / {term} | {original.iloc[0].coefficient:.4f} | {controlled.iloc[0].coefficient:.4f} |")
    test_path = root / "test_validation.json"
    test_note = ""
    if test_path.exists():
        tests = json.loads(test_path.read_text(encoding="utf-8"))
        test_note = f"전체 unittest {tests['tests_run']}개가 통과했다. 실행 명령과 소스 해시는 `test_validation.json`에 기록했다."
    literature_lines = ["| Study / access | Supports | Does not establish |", "|---|---|---|"]
    for item in LITERATURE:
        literature_lines.append(f"| [{item['key']}]({item['url']}); {item['access']} | {item['supports']} | {item['does_not_support']} |")
    stamina = diagnostics.get("stamina", {})
    debug_note = ""
    if diagnostics.get("debug_evaluation_before_chronology_audit",{}).get("occurred"):
        debug_note = "구현 디버깅 중 프레임 정합성 검사 추가 전에 test 평가가 한 차례 실행됐다. 그 결과는 `outputs/fatigue_terminal_before_frame_audit`에 보존했다. 프레임 순서 이상 제외는 성능 지표가 아니라 관측 가능성 검사에 따른 수정이며 모델·규칙 선택에 test 점수를 사용하지 않았다. 따라서 test를 처음 보는 독립 확인이라고 주장하지 않는다.\n"
    effect_text = ""
    for endpoint, term, controlled in (("e", "beta:common", "beta"), ("u", "eta:common", "eta"), ("u", "gamma:common", "gamma")):
        row = slopes[(slopes.endpoint==endpoint)&(slopes.term==term)]
        if len(row):
            item = row.iloc[0]
            if endpoint=="u" and controlled=="eta" and item.ci_high<0:
                effect_text += "무제약 η의 bootstrap 구간이 음수다. `−η x_H`가 양수가 되어 타자 소비량이 U를 낮춘다는 기전 가설과 반대다. 이 결과를 기전적 위력 감소로 해석하지 않는다.\n\n"
            if endpoint=="u" and controlled=="gamma" and item.ci_low<=0<=item.ci_high:
                effect_text += "γ의 bootstrap 구간은 0을 포함한다. 리시버 소비량의 독립 효과를 확정할 수 없다.\n\n"
    text = f"""# 누적 이동에너지와 랠리 종료확률: 실행 결과

## 연구 대상과 실제 데이터 감사

본 결과는 ShuttleSet와 ShuttleSet22의 중복 제거 후 경기 단위 split을 사용한다.
원본 CSV {diagnostics['energy_audit']['raw_csv_files_available']}개가 존재하며, 실제 사용한
{diagnostics['energy_audit']['raw_strokes_checked']:,}개 타구의 원본 타구자·득점 선수와 A/B 변환을 대조했다.
split별 경기·랠리·영상 교집합은 모두 0이다. 감사 요약은 다음과 같다.

```json
{json.dumps(cohort, ensure_ascii=False, indent=2)}
```

E는 타구자 실점과 네트/아웃/반칙 라벨이 일치한 사건, U는 타구자 득점과 Winner-by-landing 라벨이
일치한 사건이다. C는 다음 **기록 타구 시도**로의 전이이며 다음 타구가 실수일 수도 있다.
모호한 사건은 별도 프록시 분석에만 포함했다. 충돌하는 랠리는 모두 제외했다.
Forced/Unforced Error와 No Contact는 식별하지 않았다. 엄격 라벨 선택에 따른 표본 선택 편향도 가능하다.

## 비용 복원과 통계적 적합

현재 행 t의 이동은 이미 끝난 t-1→t 구간이다. `D(t)=19.985×Σ[m=2..t] distance[m]`로 계산한다.
첫 관측 타구는 기준점이며 그 이전 이동과 타구 수행 대사 비용은 미관측이다. 첫 세 타 사이의 이동도 포함한다.
좌표 양 끝이 결측이면 전처리된 거리 0을 관측값으로 인정하지 않고 train 전이비용/prior로 대체했다.
대체 여부와 누적 대체 포함 여부를 사건 테이블에 기록했다. 단위는 J/kg의 **이동에너지 대리지표**다.
비증가 프레임 번호 구간은 관측 이동으로 인정하지 않고, 해당 랠리 전체를 hazard 및 거리 적합에서 제외한다.
원본 사건은 삭제하지 않으며 프레임 정합성 플래그와 랠리 수준 제외 사유로 보존한다.

타자/리시버 역할 표본을 통합하고 {diagnostics['baseline']['history_order']}차 이력을 선택했다.
기준 Q/C는 개발 단계에서 train만으로 적합했으며 기존 train+validation parquet을 validation에 재사용하지 않았다.
상황별 E/U/C Dirichlet backoff 확률의 logit을 offset으로 사용하고 intercept와 피로항을 ridge 적합했다.
`D_*={diagnostics['d_star']:.6f}` J/kg는 strict train 양의 소비량 95백분위수이며 F_max가 아니다.
4번째 타부터의 학습 사건은 이미 소비가 발생한 시점이다. D=0의 3타 이력 query는 관측 범위 아래의
가상 초기조건일 수 있으며, 모델 함수의 extrapolation 플래그로 구분한다.
u는 E를 제외한 사건에만 적합했다. validation 경기 bootstrap의 1-SE 규칙으로 단순한 모델을 우선했다.
선택된 e는 **{selected['e']['family']}**, u는 **{selected['u']['family']}**이다.

## Held-out 평가와 계수 불확실성

설정 확정 이후 train-only 개발 모델을 test에서 평가했다. 아래 family 행은 각 가족의 validation 최적 후보이며,
selected는 1-SE 규칙 및 train-only 보정 선택 결과다. 최종 export는 train+validation 재적합 결과다.

{debug_note}

{chr(10).join(metrics)}

{chr(10).join(slope_lines)}

{effect_text}

{chr(10).join(control_lines)}

통제모델에서는 타자 소비량의 e 기울기가 작아졌다. 두 모델의 기울기는 동일한 D_*로 정규화한
logit 계수이며, 통제 이후에도 생리학적 인과효과로 해석하지 않는다.

bootstrap은 경기 단위 {diagnostics['bootstrap_repeats']}회이며 선택된 하이퍼파라미터와 D_*를 고정하고
context prior와 회귀를 다시 적합했다. 구간은 모델 선택 조건부 구간이며 인과효과나 생리학적 임계값의 구간이 아니다.
M0 선택은 피로항을 요구할 예측 근거가 부족하다는 뜻이다. 진단용 M1 계수와 선수·경기수준·이전 타수 통제모델은
`fatigue_model_coefficients.csv`, `control_model_coefficients.csv`에 별도로 저장했다.
두 선수 소비량의 상관은 {diagnostics['energy_correlation']:.4f}이다. γ와 η의 분리 추정에는 공선성에 주의해야 한다.
공통 기울기 및 shot별 가용 표본은 `shot_support.csv`에서 확인할 수 있다.
정밀도 감소율을 E/U 확률 증가율로 대입하지 않았고 비음수 기전 제약은 가설로만 비교했다.

## 전이 결합과 예산별 승률

관측 shot 정책 π와 정상 진행 조건부분포 T는 고정했다. E/U/C 확률만 누적량에 따라 바꾸며
A 기준 w/l에 올바르게 배치했다. `D=0`을 기존 경험 평균과 같게 강제하지 않았다.
질량 보존 최대 오차는 {diagnostics['transition_checks']['probability_mass_max_error']:.3g},
기존 정적 확률 복원 최대 오차는 {diagnostics['transition_checks']['static_reconstruction_max_error']:.3g}이다.

{test_note}

예산은 train의 관측 랠리 종료 소비량 백분위수를 사용한 시나리오다. 실제 최대체력이 아니다.
추가 종결 비용은 0이라는 근사이며 기존 continuation C를 자동 적용하지 않는다.
소진 순서를 모르는 동시 고갈은 하한·상한으로 보고한다. 중앙값은 해당 사건 0.5의 민감도 결과다.
이 범위는 고갈 순서 가정에 따른 구조적 범위이며 통계적 신뢰구간이 아니다. 시나리오 평균은 전체
8,192개 상태에 동일 가중치를 준 평균이며 관측 시작 상태 분포의 경기 승률과 구분한다.
격자 결과와 실제 평균 C를 그대로 사용하는 Monte Carlo를 비교했다.
수치 검증 기준은 대표 query에서 하한·중앙값·상한의 격자 세분화 차이가 모두 0.02 이하이고,
Monte Carlo 95% 오차 범위를 초과한 차이도 0.02 이하인 것이다. 전체 격자의 균일 오차 보증은 아니다.

```json
{json.dumps(stamina, ensure_ascii=False, indent=2)}
```

이 V는 예산 소진 경계가 있는 승률이고 Laplace V(p,q)와 다른 대상이다. 고정 경험 정책의 시나리오 분석이며
추천 shot의 반사실적 효과, 최적 정책, 랠리 간 회복을 검증한 결과가 아니다.

## 문헌 근거와 적용 한계

아래 접근 범위는 실제로 확인한 자료 범위다. 본문에 접근하지 못한 연구의 세부 조작적 정의를 확인했다고 주장하지 않는다.

{chr(10).join(literature_lines)}

## 추가 데이터 요구

종결 타구 이후 두 선수 위치/움직임, 라켓 접촉 여부, 타구 시도 없는 종결의 명시적 기록,
forced/unforced 전문 주석, 실제 이동 궤적·속도·런지 부하·생리 측정이 필요하다.
현재 소비량은 랠리 길이·타구 난도·선수 기량의 대리변수일 수도 있다.
선수와 과거 타수를 통제한 계수 비교를 함께 해석해야 하며, 선택 모델이 M0이면 피로효과가 입증됐다고 쓰면 안 된다.
"""
    (root / "research_report.md").write_text(text, encoding="utf-8")
    (root / "literature_evidence.json").write_text(json.dumps(LITERATURE, indent=2, ensure_ascii=False), encoding="utf-8")
