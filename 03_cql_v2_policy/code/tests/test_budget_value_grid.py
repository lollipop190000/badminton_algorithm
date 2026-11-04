import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests.test_energy_strategy import tiny
from src.energy_strategy import solve_strategy
from src.budget_value_grid import solve_value_grid, ValueGrid, ensure_grid_space, required_grid_bytes


class MeanValueGridTests(unittest.TestCase):
    def test_preallocated_file_grid_is_promoted_only_after_complete_solve(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'grid';partial=p.with_suffix('.npy.partial')
            g=solve_value_grid(tiny(1,2),10.,12.,intervals=3,storage_path=partial)
            self.assertIsInstance(g.values,np.memmap)
            self.assertTrue(partial.exists());self.assertFalse(p.with_suffix('.npy').exists())
            expected=g.values.copy();g.save(p)
            self.assertFalse(partial.exists());self.assertTrue(p.with_suffix('.npy').exists())
            np.testing.assert_array_equal(g.values,expected)
            g.values._mmap.close()

    def test_comparison_checks_space_before_any_new_bellman_solve(self):
        from scripts.compare_integrated_cql import prepare_grids
        system=SimpleNamespace(transition=SimpleNamespace(pi=np.ones((2,1,1))))
        args=SimpleNamespace(mean_only_grid=True,max_grid_intervals=400,grid_intervals=400,coarse_comparison_id=None)
        with tempfile.TemporaryDirectory() as tmp, \
             patch('scripts.compare_integrated_cql.player_symmetric',return_value=True), \
             patch('scripts.compare_integrated_cql.ensure_grid_space',side_effect=OSError('capacity')), \
             patch('scripts.compare_integrated_cql.solve_value_grid') as solve:
            with self.assertRaises(OSError):
                prepare_grids(system,None,{'budget_scale':10.,'cost_fingerprint':'cost'},None,Path(tmp),args,{}, {})
            solve.assert_not_called()

    def test_matches_legacy_win_value_for_both_players_and_exhaustion_boundaries(self):
        for focal in ('A','B'):
            for ca,cb in ((0,0),(1,2),(4,3),(20,0),(0,20),(20,20)):
                with self.subTest(focal=focal,ca=ca,cb=cb):
                    system=tiny(ca,cb)
                    old=solve_strategy(system,10.,12.,focal,intervals=5)
                    new=solve_value_grid(system,10.,12.,focal,intervals=5)
                    self.assertEqual(new.values.dtype,np.float64)
                    self.assertEqual(new.values.shape[-1],1)
                    np.testing.assert_allclose(new.values[...,0],old.values[...,0],atol=3e-11,rtol=0)
                    np.testing.assert_allclose(new.query([0,1],3.3,8.7)[...,0],old.query([0,1],3.3,8.7)[...,0],atol=3e-11,rtol=0)
                    self.assertLessEqual(new.max_residual,1e-10)

    def test_zero_cost_cycle_matches_independent_analytic_value(self):
        g=solve_value_grid(tiny(),10.,10.,intervals=3)
        self.assertAlmostEqual(g.query(1,10.,10.)[0],(.2+.7*.1)/(1-.7**2),places=9)

    def test_read_reload_preserves_values_and_rejects_overwriting_completed_grid(self):
        g=solve_value_grid(tiny(1,2),10.,12.,intervals=3)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'grid';g.save(p);loaded=ValueGrid.load(p)
            np.testing.assert_array_equal(g.values,loaded.values)
            self.assertEqual(loaded.focal,'A')
            loaded.values._mmap.close()
            with self.assertRaises(FileExistsError):g.save(p)

    def test_space_guard_accounts_full_float64_array_and_reserve(self):
        needed=required_grid_bytes(400,8192)
        self.assertGreaterEqual(needed,401*401*8192*8)
        with patch('src.budget_value_grid.shutil.disk_usage',return_value=SimpleNamespace(free=needed+999)):
            with self.assertRaises(OSError):ensure_grid_space(Path('.'),400,8192,reserve_bytes=1000)
        with patch('src.budget_value_grid.shutil.disk_usage',return_value=SimpleNamespace(free=needed+1000)):
            ensure_grid_space(Path('.'),400,8192,reserve_bytes=1000)

    def test_multiple_refinement_levels_reserve_all_retained_files_before_solving(self):
        from scripts.compare_integrated_cql import prepare_grids
        system=SimpleNamespace(transition=SimpleNamespace(pi=np.ones((2,4096,1))))
        args=SimpleNamespace(mean_only_grid=True,max_grid_intervals=400,grid_intervals=200,coarse_comparison_id=None)
        final_only=required_grid_bytes(400,8192)+1024**3
        with tempfile.TemporaryDirectory() as tmp, \
             patch('scripts.compare_integrated_cql.player_symmetric',return_value=True), \
             patch('src.budget_value_grid.shutil.disk_usage',return_value=SimpleNamespace(free=final_only)), \
             patch('scripts.compare_integrated_cql.solve_value_grid',side_effect=AssertionError('solve started')) as solve:
            with self.assertRaisesRegex(OSError,'free bytes'):
                prepare_grids(system,None,{'budget_scale':10.,'cost_fingerprint':'cost'},None,Path(tmp),args,{}, {})
            solve.assert_not_called()


if __name__=='__main__':unittest.main()
