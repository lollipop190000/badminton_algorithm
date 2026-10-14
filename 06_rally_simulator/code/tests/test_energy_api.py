from pathlib import Path
import tempfile
import json
import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.energy import EnergyStrategyService
from app.schemas import EnergyRallyRequest, RallyRequest
from src.energy_strategy import load_system,StrategyGrid

ROOT=Path(__file__).resolve().parents[1]


class EnergyApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system=load_system(ROOT)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.service=EnergyStrategyService(ROOT);self.service.run=Path(self.temp.name)
        self.service.cache={'distance':self.system}
        self.service.registry=lambda:dict(run_id='unit-test',entries=[dict(model_id='distance',focal=f,grid=f,
             behavior_grid=f+'base',domain=1000.,numerical_validation_passed=False) for f in ('A','B')])
        values=np.zeros((3,3,8192,2));values[...,0]=.5
        self.service.grids={key:StrategyGrid(values,1000,1000,f,'energy') for f in ('A','B') for key in (f,f+'base')}
        self.patcher=patch('app.main.energy_service',self.service);self.patcher.start()
        self.client=TestClient(app)
        self.payload=dict(history=[1,2,3],current_hitter='A',strategy_player='A',cost_model='distance',initial_a=1000,initial_b=1000,remaining_a=800,remaining_b=600)

    def tearDown(self):
        self.patcher.stop();self.temp.cleanup()

    def test_api_all_cells_and_b_side_mirror(self):
        r=self.client.post('/api/energy-strategy',json=self.payload)
        self.assertEqual(r.status_code,200);body=r.json()
        self.assertEqual(len(body['actions']),128);self.assertEqual(len(body['cells']),16)
        self.assertFalse(body['numerical_validation_passed'])
        self.assertTrue(all(a['supported'] for a in body['top_actions']))
        self.payload.update(current_hitter='B',strategy_player='B')
        body=self.client.post('/api/energy-strategy',json=self.payload).json()
        z1=next(a for a in body['actions'] if a['target_zone']==1)
        self.assertEqual(z1['screen_column'],3);self.assertEqual(z1['screen_side'],'A')

    def test_bad_budgets_missing_history_and_exhaustion(self):
        for update in [dict(remaining_a=1001),dict(remaining_a=-1),dict(history=[17,1,1]),dict(cost_model='bad')]:
            self.assertEqual(self.client.post('/api/energy-strategy',json={**self.payload,**update}).status_code,422)
        for h in ([1,2],[1,0,2],[]):
            self.assertFalse(self.client.post('/api/energy-strategy',json={**self.payload,'history':h}).json()['available'])
        body=self.client.post('/api/energy-strategy',json={**self.payload,'remaining_a':0}).json()
        self.assertEqual(body['top_actions'],[])
        self.assertFalse(body['available'])
        body=self.client.post('/api/energy-strategy',json={**self.payload,'current_hitter':'B'}).json()
        self.assertFalse(body['available']);self.assertEqual(body['cells'],[])

    def test_changed_model_cannot_be_mixed_with_saved_grid(self):
        (self.service.run/'RUN.json').write_text(json.dumps({'signature':{'input_sha256':{}}}),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'해시'):
            self.service.verify_model_inputs()

    def test_rally_energy_is_pre_action_and_includes_early_motion(self):
        def stroke(i):
            return dict(hitter='A' if i%2==0 else 'B',shot_type='short_service' if i==0 else 'drive',
                target_zone=i%16+1,player_a_position=dict(x_m=0,y_m=2+i*.1),player_b_position=dict(x_m=0,y_m=10-i*.1),
                delta_time_s=0 if i==0 else .5)
        rally=RallyRequest(strokes=[stroke(i) for i in range(6)])
        p=EnergyRallyRequest(rally=rally,stroke_index=4,cost_model='distance',initial_a=1000,initial_b=1000)
        first=self.service.from_rally(p)
        self.assertEqual(first['history'],[2,3,4]);self.assertGreater(first['consumed_a'],0)
        altered=rally.model_copy(deep=True)
        altered.strokes[4].shot_type='smash';altered.strokes[4].target_zone=16
        second=self.service.from_rally(p.model_copy(update={'rally':altered}))
        self.assertEqual(first,second)


if __name__=='__main__':unittest.main()
