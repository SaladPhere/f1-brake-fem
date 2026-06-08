from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brake_fem.fem import assemble_heat_operators
from brake_fem.mesh import generate_annulus_mesh, mesh_area, triangle_areas


def small_config() -> dict:
    return {
        "geometry": {
            "r_inner": 0.06,
            "r_outer": 0.164,
            "thickness": 0.032,
            "solid_fraction": 0.35,
            "n_radial": 4,
            "n_theta": 32,
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
            "include_edge_cooling": True,
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
            "cg_tol": 1.0e-9,
            "cg_max_iter": 100,
            "snapshot_count": 2,
            "animation_frames": 0,
        },
    }


class MeshFEMTests(unittest.TestCase):
    def test_annulus_mesh_area_and_orientation(self) -> None:
        mesh = generate_annulus_mesh(0.06, 0.164, 4, 64)
        areas = triangle_areas(mesh)
        expected = math.pi * (0.164**2 - 0.06**2)
        self.assertTrue(np.all(areas > 0.0))
        self.assertLess(abs(mesh_area(mesh) - expected) / expected, 0.01)

    def test_lumped_mass_is_positive(self) -> None:
        cfg = small_config()
        mesh = generate_annulus_mesh(0.06, 0.164, 4, 32)
        ops = assemble_heat_operators(mesh, cfg)
        self.assertTrue(np.all(ops.mass > 0.0))
        self.assertTrue(np.all(ops.node_area > 0.0))
        self.assertGreater(ops.thermal_capacity_total, 0.0)


if __name__ == "__main__":
    unittest.main()
