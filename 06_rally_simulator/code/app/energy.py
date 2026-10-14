"""Read-only serving of executed energy-strategy runs and their provenance."""
from pathlib import Path
import json
import hashlib
import numpy as np
from src.energy_strategy import (ASSUMPTIONS, StrategyGrid, StrategySystem, action_scores, load_system, _choose)
from src.estimate_markov_q import state_id, MID_RALLY_ACTIONS
from app.geometry import zone_cell


class EnergyStrategyService:
    def __init__(self,root=None):
        self.root=Path(root or Path(__file__).resolve().parents[1])
        self.cache={};self.grids={};self.run=None

    def registry(self):
        latest=json.loads((self.root/'outputs/energy_strategy/latest.json').read_text(encoding='utf-8'))
        run=self.root/'outputs/energy_strategy/runs'/latest['run_id']
        if run!=self.run:self.cache.clear();self.grids.clear();self.run=run
        return json.loads((run/'registry.json').read_text(encoding='utf-8'))

    def metadata(self):
        r=self.registry()
        return dict(run_id=r['run_id'],budgets=r['budgets'],models=sorted(set(e['model_id'] for e in r['entries'])),
            scenarios=r['entries'],assumptions=ASSUMPTIONS)

    def system(self,model_id):
        if model_id not in self.cache:
            self.verify_model_inputs()
            with np.load(self.run/'support.npz') as f:
                support=tuple(f[k] for k in ('levels','counts','cost_counts','global_counts'))
            # Load audited baseline/hazards once; BFMD cost bundle is the exact
            # keyed result of the executed lookup adapter, not a refit.
            system=load_system(self.root,'distance',support=support)
            path=self.run/f'costs_{model_id}.npz'
            with np.load(path) as f:
                system.transition.cost_a=f['cost_a'];system.transition.cost_b=f['cost_b']
                system.provenance={p:{k:f[f'{p}_{k}'] for k in ('n','level','extrapolated')}
                                   for p in ('A','B')} if model_id!='distance' else {}
            system.model_id=model_id;self.cache[model_id]=system
        return self.cache[model_id]

    def verify_model_inputs(self):
        """Never combine an old policy grid with newly refitted hazards/Q."""
        record=json.loads((self.run/'RUN.json').read_text(encoding='utf-8'))
        expected=record['signature']['input_sha256']
        for relative in ('outputs/fatigue_terminal/baseline_final.npz',
                         'outputs/fatigue_terminal/models/strict_final/parameters.npz',
                         'outputs/fatigue_terminal/models/strict_final/model.json'):
            value=hashlib.sha256((self.root/relative).read_bytes()).hexdigest()
            if value!=expected.get(relative):
                raise ValueError('현재 모델과 전략 실행의 입력 해시가 다릅니다. 해당 입력을 복원하거나 새 전략 실행을 만드세요.')

    def analyze(self,payload):
        r=self.registry()
        entries=[e for e in r['entries'] if e['model_id']==payload.cost_model and e['focal']==payload.strategy_player]
        if not entries:raise ValueError('해당 비용·선수 전략이 없습니다. energy-strategy 실행 결과를 확인하세요.')
        entry=entries[0]
        history=payload.history
        response=dict(run_id=r['run_id'],cost_model=payload.cost_model,strategy_player=payload.strategy_player,
            current_hitter=payload.current_hitter,numerical_validation_passed=entry['numerical_validation_passed'],
            assumptions=ASSUMPTIONS,available=False,cells=[],top_actions=[],reason=None)
        if len(history)!=3 or 0 in history:
            response['reason']='유효한 최근 3타구 구획이 필요합니다. Z0 종료 상태는 추천하지 않습니다.';return response
        a,b=payload.remaining_a,payload.remaining_b
        if a>payload.initial_a or b>payload.initial_b:raise ValueError('잔여 예산은 초기 예산을 넘을 수 없습니다.')
        if max(a,b)>entry['domain']:raise ValueError(f"사전 계산 범위는 {entry['domain']:.3f} J/kg까지입니다. 더 큰 예산은 별도 실행이 필요합니다.")
        system=self.system(payload.cost_model)
        for name in ('grid','behavior_grid'):
            key=entry[name]
            if key not in self.grids:self.grids[key]=StrategyGrid.load(self.run/key)
        grid=self.grids[entry['grid']];behavior=self.grids[entry['behavior_grid']]
        si=int(state_id(*history));parity=1 if payload.current_hitter=='A' else 0
        score,ambiguity=action_scores(system,grid,si,parity,a,b)
        baseline=float(behavior.query(parity*4096+si,a,b)[0])
        v=grid.query(parity*4096+si,a,b)
        response.update(state_id=si,available=True,model_value=float(v[0]),lower=float(v[0]-.5*v[1]),upper=float(v[0]+.5*v[1]),
            behavior_value=baseline,delta_vs_behavior=float(v[0]-baseline),remaining_a=a,remaining_b=b)
        if a<=0 or b<=0:
            response.update(available=False,reason='예산 소진 상태입니다.')
            return response
        if payload.current_hitter!=payload.strategy_player:
            response.update(available=False,reason='현재는 상대 차례입니다. 다음 내 차례에서 전략을 적용합니다.')
            return response
        cost_a,cost_b=system.transition.cost_a[parity,si],system.transition.cost_b[parity,si]
        supported=(system.levels[si]>0).copy()
        own_cost=cost_a if payload.strategy_player=='A' else cost_b
        def describe(action):
            shot,z=divmod(action,16)
            side,row,col=zone_cell(payload.current_hitter,z+1)
            result=dict(shot_type=MID_RALLY_ACTIONS[shot],target_zone=z+1,screen_side=side,screen_row=row,screen_column=col,
                value=float(score[shot,z]),lower=float(score[shot,z]-.5*ambiguity[shot,z]),upper=float(score[shot,z]+.5*ambiguity[shot,z]),
                delta_vs_behavior=float(score[shot,z]-baseline),cost_a=float(cost_a[shot,z]),cost_b=float(cost_b[shot,z]),
                support_order=int(system.levels[si,shot,z]),train_count=int(system.counts[si,shot,z]),valid_cost_count=int(system.cost_counts[si,shot,z]),
                global_count=int(system.global_counts[shot,z]),supported=bool(supported[shot,z]))
            if system.provenance:
                result['bfmd']={p:{k:float(values[parity,si,shot,z]) for k,values in meta.items()} for p,meta in system.provenance.items()}
                result['bfmd']['n_exact']=0
            return result
        response['actions']=[describe(i) for i in range(128)]
        for zone in range(16):
            mask=np.zeros_like(supported);mask[:,zone]=supported[:,zone]
            best=_choose(score.ravel(),mask.ravel(),own_cost.ravel(),system.counts[si].ravel())
            response['cells'].append(describe(best) if best>=0 else dict(target_zone=zone+1,supported=False,
                screen_side=zone_cell(payload.current_hitter,zone+1)[0],screen_row=zone_cell(payload.current_hitter,zone+1)[1],screen_column=zone_cell(payload.current_hitter,zone+1)[2]))
        mask=supported.ravel().copy()
        for rank in range(3):
            best=_choose(score.ravel(),mask,own_cost.ravel(),system.counts[si].ravel())
            if best<0:break
            response['top_actions'].append(dict(rank=rank+1,**describe(best)));mask[best]=False
        if not response['top_actions']:response['reason']='지원 표본이 충분한 후보가 없어 추천을 보류합니다.'
        return response

    def from_rally(self,payload):
        """Compute before-current-attempt incoming energy with explicit sources.

        The last input stroke is the current attempt to advise: its actual shot
        and actual destination never enter its recommendation or energy.
        """
        strokes=payload.rally.strokes
        index=payload.stroke_index if payload.stroke_index is not None else len(strokes)-1
        if index<0 or index>=len(strokes):raise ValueError('잘못된 타구 번호입니다.')
        current=strokes[index]
        self.registry();system=self.system(payload.cost_model)
        spent_a=spent_b=0.;sources=[]
        for i in range(1,index+1):
            prev,now=strokes[i-1],strokes[i]
            if i>=4 and prev.shot_type in MID_RALLY_ACTIONS and all(s.target_zone>0 for s in strokes[i-4:i]):
                history=[s.target_zone for s in strokes[i-4:i-1]]
                state=int(state_id(*history));parity=1 if prev.hitter=='A' else 0
                sh=MID_RALLY_ACTIONS.index(prev.shot_type);zone=prev.target_zone-1
                spent_a+=float(system.transition.cost_a[parity,state,sh,zone]);spent_b+=float(system.transition.cost_b[parity,state,sh,zone])
                sources.append('model continuation cost')
            else:
                # Early/service intervals have no 3-hit key; positions are
                # measured inputs or explicit zone-centroid approximations.
                for p in ('a','b'):
                    x=getattr(prev,f'player_{p}_position');y=getattr(now,f'player_{p}_position')
                    cost=19.985*np.hypot(y.x_m-x.x_m,y.y_m-x.y_m)
                    if p=='a':spent_a+=float(cost)
                    else:spent_b+=float(cost)
                sources.append('position distance anchor (early/service)')
        return dict(history=[s.target_zone for s in strokes[max(0,index-3):index]],current_hitter=current.hitter,
            consumed_a=spent_a,consumed_b=spent_b,remaining_a=max(0.,payload.initial_a-spent_a),
            remaining_b=max(0.,payload.initial_b-spent_b),sources=sources,
            anchor='first annotated hit; terminal extra cost zero')
