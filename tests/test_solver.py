from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brake_fem.solver import run_simulation
from brake_fem.source import gaussian_pad_load
from brake_fem.telemetry import ensure_derived_columns


def base_config() -> dict:
    return {
        "geometry": {
            "r_inner": 0.06,
            "r_outer": 0.164,
            "thickness": 0.032,
            "solid_fraction": 0.35,
            "n_radial": 3,
            "n_theta": 24,
        },
        "material": {
            "rho": 1800.0,
            "cp": 1400.0,
            "k_inplane": 30.0,
            "emissivity": 0.0,
        },
        "cooling": {
            "ambient_c": 25.0,
            "h0": 0.0,
            "h1": 0.0,
            "alpha": 1.0,
            "h_scale": 0.0,
            "cooling_area_factor": 2.0,
            "include_edge_cooling": False,
        },
        "source": {
            "eta_heat": 1.0,
            "r_pad": 0.125,
            "sigma_r": 0.018,
            "sigma_theta": 0.22,
            "theta0": 0.0,
        },
        "solver": {
            "initial_temperature": "data_front",
            "sample_stride": 1,
            "cg_tol": 1.0e-10,
            "cg_max_iter": 120,
            "snapshot_count": 2,
            "animation_frames": 0,
        },
    }


def synthetic_df(power: bool = False, temp: float = 100.0) -> pd.DataFrame:
    torque = 10.0 if power else 0.0
    omega = 100.0 if power else 0.0
    df = pd.DataFrame(
        {
            "lap_s": [0.0, 1.0],
            "Ground Speed": [0.0, 0.0],
            "Brake Pos": [100.0 if power else 0.0, 100.0 if power else 0.0],
            "Brake Torque FL": [torque, torque],
            "Brake Torque FR": [torque, torque],
            "Brake Temp FL": [temp, temp],
            "Brake Temp FR": [temp, temp],
            "Wheel Angular Speed FL": [omega, omega],
            "Wheel Angular Speed FR": [omega, omega],
        }
    )
    return ensure_derived_columns(df)


class SolverTests(unittest.TestCase):
    def test_constant_temperature_is_preserved_without_load_or_cooling(self) -> None:
        cfg = base_config()
        df = synthetic_df(power=False, temp=123.0)
        result = run_simulation(df, cfg, store_snapshots=True)
        self.assertLess(float(np.max(np.abs(result.snapshots[-1] - 123.0))), 1.0e-7)
        self.assertAlmostEqual(float(result.t_mean[-1]), 123.0, places=7)

    def test_heat_input_matches_energy_increase_without_losses(self) -> None:
        cfg = base_config()
        cfg["material"]["k_inplane"] = 0.0
        cfg["solver"]["initial_temperature"] = 0.0
        df = synthetic_df(power=True, temp=0.0)
        result = run_simulation(df, cfg, store_snapshots=True)
        energy = float(np.sum(result.operators.mass * result.snapshots[-1]))
        self.assertAlmostEqual(energy, 1000.0, delta=1.0e-4)

    def test_gaussian_source_integrates_to_requested_power(self) -> None:
        cfg = base_config()
        df = synthetic_df(power=False, temp=0.0)
        result = run_simulation(df, cfg, store_snapshots=False)
        load = gaussian_pad_load(result.mesh, result.operators.node_area, 1234.5, 0.4, cfg["source"])
        self.assertAlmostEqual(float(np.sum(load)), 1234.5, places=7)


if __name__ == "__main__":
    unittest.main()
