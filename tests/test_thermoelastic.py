from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brake_fem.thermoelastic import assemble_elasticity_operators, solve_thermoelastic_snapshots
from test_solver import base_config, synthetic_df
from brake_fem.solver import run_simulation


class ThermoelasticTests(unittest.TestCase):
    def thermo_config(self) -> dict:
        cfg = copy.deepcopy(base_config())
        cfg["thermoelastic"] = {
            "enabled": True,
            "model": "plane_stress",
            "reference_temperature": "initial_mean",
            "use_solid_fraction": True,
            "cg_tol": 1.0e-8,
            "cg_max_iter": 400,
            "material": {
                "young_modulus_pa": 30.0e9,
                "poisson_ratio": 0.22,
                "thermal_expansion": 2.0e-6,
            },
        }
        return cfg

    def test_elasticity_operator_has_positive_free_diagonal(self) -> None:
        cfg = self.thermo_config()
        result = run_simulation(synthetic_df(power=False, temp=120.0), cfg, store_snapshots=True)
        ops = assemble_elasticity_operators(result.mesh, cfg)
        self.assertTrue(np.all(ops.diagonal[ops.free_dofs] > 0.0))
        self.assertEqual(len(ops.constrained_dofs), 3)

    def test_uniform_temperature_at_reference_has_near_zero_stress(self) -> None:
        cfg = self.thermo_config()
        result = run_simulation(synthetic_df(power=False, temp=120.0), cfg, store_snapshots=True)
        stress = solve_thermoelastic_snapshots(result)
        self.assertIsNotNone(stress)
        assert stress is not None
        self.assertLess(float(np.max(stress.max_von_mises_pa)), 1.0e-3)
        self.assertLess(float(np.max(stress.max_displacement_m)), 1.0e-12)

    def test_nonuniform_temperature_produces_finite_stress(self) -> None:
        cfg = self.thermo_config()
        cfg["solver"]["initial_temperature"] = 100.0
        cfg["solver"]["snapshot_count"] = 2
        result = run_simulation(synthetic_df(power=True, temp=100.0), cfg, store_snapshots=True)
        stress = solve_thermoelastic_snapshots(result)
        self.assertIsNotNone(stress)
        assert stress is not None
        self.assertTrue(np.all(np.isfinite(stress.max_von_mises_pa)))
        self.assertGreater(float(np.max(stress.max_von_mises_pa)), 0.0)


if __name__ == "__main__":
    unittest.main()