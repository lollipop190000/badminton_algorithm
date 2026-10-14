"""Add human-readable strategy examples and verify an executed run's inputs.

This does not refit, select models, change policy grids or repeat a scientific
audit. It derives tables from recorded results and records delivery checks.
"""
from pathlib import Path
import hashlib
import json
import sys
import shutil
import importlib.metadata as metadata
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.energy_strategy import StrategyGrid
from src.estimate_markov_q import MID_RALLY_ACTIONS,decode_state_id

LABELS={'distance':'기존 거리','M0_distance_reference':'BFMD 경로',
        'M3_hypothesis_speed_only_1':'BFMD 속력 가설','M3_hypothesis_curve_only_1':'BFMD 방향전환 가설',
        'M3_hypothesis_balanced_025':'BFMD 균형 가설'}


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def table(frame):
    rows=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    rows+=['| '+' | '.join(f'{x:.3f}' if isinstance(x,float) else str(x) for x in r)+' |' for r in frame.itertuples(index=False,name=None)]
    return '\n'.join(rows)


def main():
    run_id=json.loads((ROOT/'outputs/energy_strategy/latest.json').read_text(encoding='utf-8'))['run_id']
    out=ROOT/'outputs/energy_strategy/runs'/run_id
    record=json.loads((out/'RUN.json').read_text(encoding='utf-8'))
    if record['status']!='completed':raise ValueError('Only a completed execution can be finalized')
    for name,expected in record['signature']['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Executed input changed: '+name)
    for name,expected in record['signature']['code_sha256'].items():
        if digest(out/'source_snapshot'/Path(name).name)!=expected:raise ValueError('Executed source snapshot changed: '+name)
    reg=json.loads((out/'registry.json').read_text(encoding='utf-8'))
    choices=pd.read_parquet(out/'recommendations.parquet')
    state=json.loads((out/'protocol.json').read_text(encoding='utf-8'))['representative_states'][0]
    z1,z2,z3=map(int,decode_state_id(state))
    examples=[]
    for entry in reg['entries']:
        if entry['focal']!='A':continue
        grid=StrategyGrid.load(out/entry['grid']);behavior=StrategyGrid.load(out/entry['behavior_grid'])
        for percentile,budget in zip((50,75,95),reg['budgets']):
            choice=int(choices[(choices.model_id==entry['model_id']) & (choices.focal=='A') &
                               (choices.state_id==state) & (choices.percentile==percentile)].energy_action.iloc[0])
            frame=pq.read_table(out/'action_scores'/f"{entry['model_id']}_A_p{percentile}.parquet",filters=[('state_id','=',state)]).to_pandas()
            action=frame[frame.shot_type.eq(MID_RALLY_ACTIONS[choice//16]) & frame.target_zone.eq(choice%16+1)].iloc[0]
            v=grid.query(4096+state,budget,budget);base=behavior.query(4096+state,budget,budget)
            examples.append(dict(cost_model=entry['model_id'],history=f'{z1}→{z2}→{z3}',incoming_zone=z3,
                percentile=percentile,budget=budget,shot_type=action.shot_type,target_zone=int(action.target_zone),
                model_value=float(v[0]),behavior_value=float(base[0]),cost_self=float(action.cost_A),cost_opponent=float(action.cost_B),
                support_order=int(action.support_order),support_count=int(action.transition_count)))
        grid.values._mmap.close();behavior.values._mmap.close()
    ex=pd.DataFrame(examples);ex.to_csv(out/'strategy_examples.csv',index=False)
    comparison=pd.read_csv(out/'policy_comparison.csv')
    for stem in ('expected_energy','exhaustion'):
        comparison[stem+'_self']=np.where(comparison.focal.eq('A'),comparison[stem+'_A'],comparison[stem+'_B'])
        comparison[stem+'_opponent']=np.where(comparison.focal.eq('A'),comparison[stem+'_B'],comparison[stem+'_A'])
    tradeoffs=comparison.groupby(['model_id','policy'])[['midpoint','expected_energy_self','expected_energy_opponent','exhaustion_self','exhaustion_opponent']].mean().reset_index()
    tradeoffs.to_csv(out/'strategy_tradeoffs.csv',index=False)
    validation=pd.read_csv(out/'numerical_validation.csv')
    full_rows=sum(pq.ParquetFile(p).metadata.num_rows for p in (out/'action_scores').glob('*.parquet'))
    if len(reg['entries'])!=10 or full_rows!=15728640 or not all(e['numerical_validation_passed'] for e in reg['entries']):
        raise ValueError('Expected all five cost models, both players, complete action tables and passed checks')
    display=ex.copy();display['cost_model']=display.cost_model.map(LABELS)
    display['model_value_percent']=display.model_value*100;display['behavior_value_percent']=display.behavior_value*100
    display=display[['cost_model','history','budget','shot_type','target_zone','model_value_percent','behavior_value_percent','cost_self','cost_opponent']]
    add='\n\n<!-- delivery-interpretation -->\n### 들어온 구획에서 보낼 구획으로\n\n'
    add+='아래는 A 선수, 최근 구획 '+f'`{z1}→{z2}→{z3}`'+', 양쪽 같은 잔여 예산의 실제 계산 예다. 모형 V는 목표 구획 도착과 예산 소진 규칙을 가정한 값이다.\n\n'+table(display)
    add+='\n\n원본 수치는 [strategy_examples.csv](strategy_examples.csv)에 있다. 대표 질의 평균의 내/상대 기대 소비량·소진확률은 [strategy_tradeoffs.csv](strategy_tradeoffs.csv)에 별도로 정리했다. 승리 가치를 최대화하는 전략이므로 모든 경우에 소비량을 최소화한다고 해석하지 않는다.\n\n'
    add+='### 검증 범위와 완전한 결과\n\n'
    add+=f'5개 비용 모형 × 양 선수, 고정 질의 {len(validation)}개에 각각 MC 20,000회를 적용했다. 최대 Bellman 잔차 {validation.bellman_residual.max():.6g}, 최대 격자 차이 {validation.refinement_difference.max():.6f}, 최대 MC 95% 구간 밖 오차 {validation.mc_excess_95ci.max():.6f}로 모두 기준을 통과했다. 실제 경기 승률·대사 비용·피로 인과효과 검증은 아니다.\n\n'
    add+=f'전체 행동 점수 {full_rows:,}행은 각 전략 선수의 차례 4,096개 패턴 × 128행동 × 양 선수 × 3개 공통 예산 × 5개 비용 모형을 모두 포함한다. 큰 격자는 각 전략에 대해 교대 포함 8,192개 상태를 보존한다. 추천 변화율은 이 상태들을 균등 가중한 비교이며 관측 경기에서의 행동 빈도가 아니다. 정책 비교 표는 선수별 4개 질의를 동등 가중한 모형 rollout 평균이다.\n\n'
    add+='BFMD 비용은 기존 parity 거리 표에 조건부 추정을 연결한 동결 lookup이다. 기존 역할 통합 거리 비용과의 차이는 경로 비용뿐 아니라 비용 추정 방식·조건부 거리 prior의 차이도 포함한다. 가감속·방향전환 계수는 피팅되지 않은 가설이다. 비용별 replay 백분위수는 프레임 유효 train 전체와 감사된 대체 소비량을 포함하며, 공통 예산의 기존 완전 관측 모집단과 구분한다.\n\n'
    add+='CQL 비교는 strict 적격 test 기록 상태에서 수행한 추천 일치 비교다. CQL Q를 승률로 바꾸지 않고, 목표 구획 전략의 실제 효과를 OPE로 검증했다고 표현하지 않는다. [비교 요약](cql_comparison_summary.csv), [전체 비교](cql_comparison.parquet), [추천 변화율](recommendation_change_rates.csv), [비용별 백분위수](cost_specific_budgets.csv), [예산 척도별 비교](normalized_budget_comparison.csv).\n\n'
    add+='실행 코드는 기준 commit과 미커밋 변경의 조합이었다. 실제 실행 파일 SHA256와 [source_snapshot](source_snapshot)을 보존했다. 입력·checkpoint를 다시 감사했다고 주장하지 않는다. 이번 완료 과정에서는 입력 해시 유지와 새 산출물·실행 경로만 확인했다.\n'
    report=out/'REPORT.md';previous=report.read_text(encoding='utf-8')
    delivery_checks='\n\n<!-- delivery-checks -->'+previous.split('<!-- delivery-checks -->',1)[1] if '<!-- delivery-checks -->' in previous else ''
    text=previous.split('<!-- delivery-interpretation -->')[0].rstrip()
    report.write_text(text+add+delivery_checks,encoding='utf-8')
    large=json.loads((out/'large_artifacts.json').read_text(encoding='utf-8'))
    for item in large:
        item['path']=item['path'].replace('\\','/')
        item['portable_recreation_command']='python run.py energy-strategy --run-id <NEW_UNIQUE_RUN_ID> --grid-intervals 40 --max-grid-intervals 160 --mc-samples 20000'
        item['working_folder_recreation_command']='python scripts/run_energy_strategy.py --run-id <NEW_UNIQUE_RUN_ID> --grid-intervals 40 --max-grid-intervals 160 --mc-samples 20000'
    (out/'large_artifacts.json').write_text(json.dumps(large,ensure_ascii=False,indent=2),encoding='utf-8')
    record['environment']['packages']={p:metadata.version(p) for p in ('numpy','pandas','pyarrow','scipy','numba','llvmlite','matplotlib','torch','fastapi')}
    record['delivery_input_hashes_verified']=True
    record['supplemental_input_sha256']={p:digest(ROOT/p) for p in ('outputs/fatigue_terminal/event_energy_table.parquet',
        'outputs/fatigue_terminal/stamina/protocol.json','outputs/cql_v2/predictions/cql_test_recommendations.csv')}
    record['delivery_analysis_code_sha256']=digest(Path(__file__))
    shutil.copyfile(Path(__file__),out/'source_snapshot'/Path(__file__).name)
    record['metrics'].update(full_action_rows=full_rows,max_bellman_residual=float(validation.bellman_residual.max()),
        max_refinement_difference=float(validation.refinement_difference.max()),max_mc_excess_95ci=float(validation.mc_excess_95ci.max()))
    record['metric_definitions']=dict(midpoint='conditional model win value 0..1; equal-weight representative query mean',
        energy='J/kg movement proxy; complete continuation interval cost including exhaustion overshoot',
        exhaustion='simulated probability 0..1 under assumed budget-loss rule',
        structural_bounds='same policy; simultaneous exhaustion assigned all loss/all win; not statistical CI')
    record['outputs']=[dict(path=p.relative_to(out).as_posix(),sha256=digest(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='RUN.json' and p.suffix!='.npy']
    (out/'RUN.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f'Verified {full_rows:,} full action rows; all {len(validation)} numerical queries passed; input/source hashes preserved')


if __name__=='__main__':main()
