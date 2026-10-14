from pathlib import Path
import unittest
import numpy as np
import pyarrow.parquet as pq
from src.estimate_markov_q import MID_RALLY_ACTIONS


class BFMDStrategyAdapterTests(unittest.TestCase):
    def test_actual_full_lookup_beta_zero_identity_and_structural_keys(self):
        root=Path(__file__).resolve().parents[1]
        shape=(2,2,4096,8,16)
        values={m:np.full(shape,np.nan) for m in ('M0_distance_reference','M3_beta0_BFMD_normalized')}
        shots={s:i for i,s in enumerate(MID_RALLY_ACTIONS)}
        columns=['i','j','r','p','s','model_id','cost_J_per_kg','n_exact','imputed_flag']
        for batch in pq.ParquetFile(root/'outputs/bfmd_zago/bfmd_zago_lookup.parquet').iter_batches(batch_size=65536,columns=columns):
            f=batch.to_pandas()
            self.assertTrue(((f.j//16)==(f.i%256)).all())
            self.assertTrue(f.n_exact.eq(0).all());self.assertTrue(f.imputed_flag.all())
            for model in values:
                q=f[f.model_id.eq(model)]
                key=(q.p.map({'A':0,'B':1}).to_numpy(int),q.r.to_numpy(int),q.i.to_numpy(int),q.s.map(shots).to_numpy(int),q.j.to_numpy(int)%16)
                self.assertFalse(np.isfinite(values[model][key]).any())
                values[model][key]=q.cost_J_per_kg.to_numpy(float)
        self.assertTrue(all(np.isfinite(v).all() for v in values.values()))
        np.testing.assert_array_equal(values['M0_distance_reference'],values['M3_beta0_BFMD_normalized'])


if __name__=='__main__':unittest.main()
