from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import deep_copy_config, with_overrides
from .solver import run_simulation


@dataclass
class CalibrationResult:
    best_config: dict[str, Any]
    trials: pd.DataFrame
    best_eta_heat: float
    best_h_scale: float
    best_rmse: float
    baseline_rmse: float

    def params_dict(self) -> dict[str, float]:
        return {
            "eta_heat": float(self.best_eta_heat),
            "h_scale": float(self.best_h_scale),
            "search_rmse": float(self.best_rmse),
            "baseline_search_rmse": float(self.baseline_rmse),
        }


def calibrate_parameters(df: pd.DataFrame, config: dict[str, Any]) -> CalibrationResult:
    cfg = deep_copy_config(config)
    cal = cfg.get("calibration", {})
    eta0 = float(cfg["source"].get("eta_heat", 1.0))
    h0 = float(cfg["cooling"].get("h_scale", 1.0))

    search_cfg = _search_config(cfg)
    rows: list[dict[str, float | str]] = []

    baseline_rmse = _evaluate(df, search_cfg, eta0, h0, eta0, h0, cal, "baseline", rows)

    eta_candidates = [eta0 * float(f) for f in cal.get("eta_factors", [0.5, 1.0, 1.5])]
    h_candidates = [h0 * float(f) for f in cal.get("h_factors", [0.7, 1.0, 1.3])]
    best = (float("inf"), eta0, h0)
    seen: set[tuple[float, float]] = set()

    for eta in eta_candidates:
        for h_scale in h_candidates:
            key = (round(eta, 10), round(h_scale, 10))
            if key in seen:
                continue
            seen.add(key)
            score = _evaluate(df, search_cfg, eta, h_scale, eta0, h0, cal, "coarse", rows)
            rmse = float(rows[-1]["rmse"])
            if score < best[0]:
                best = (score, eta, h_scale)

    for eta_factor in cal.get("local_eta_factors", [0.85, 1.0, 1.15]):
        for h_factor in cal.get("local_h_factors", [0.85, 1.0, 1.15]):
            eta = best[1] * float(eta_factor)
            h_scale = best[2] * float(h_factor)
            key = (round(eta, 10), round(h_scale, 10))
            if key in seen:
                continue
            seen.add(key)
            score = _evaluate(df, search_cfg, eta, h_scale, eta0, h0, cal, "local", rows)
            if score < best[0]:
                best = (score, eta, h_scale)

    trials = pd.DataFrame(rows).sort_values(["score", "rmse"]).reset_index(drop=True)
    best_eta = float(best[1])
    best_h = float(best[2])
    best_cfg = with_overrides(
        cfg,
        {
            "source": {"eta_heat": best_eta},
            "cooling": {"h_scale": best_h},
        },
    )
    best_rmse = float(trials.iloc[0]["rmse"]) if len(trials) else float("nan")
    return CalibrationResult(best_cfg, trials, best_eta, best_h, best_rmse, baseline_rmse)


def _search_config(config: dict[str, Any]) -> dict[str, Any]:
    cal = config.get("calibration", {})
    mesh_overrides = cal.get("mesh", {})
    return with_overrides(
        config,
        {
            "geometry": {
                "n_radial": int(mesh_overrides.get("n_radial", config["geometry"]["n_radial"])),
                "n_theta": int(mesh_overrides.get("n_theta", config["geometry"]["n_theta"])),
            },
            "solver": {
                "sample_stride": int(cal.get("sample_stride", 4)),
                "snapshot_count": 0,
                "animation_frames": 0,
                "cg_tol": max(float(config.get("solver", {}).get("cg_tol", 1.0e-6)), 3.0e-6),
            },
        },
    )


def _evaluate(
    df: pd.DataFrame,
    base_cfg: dict[str, Any],
    eta: float,
    h_scale: float,
    eta0: float,
    h0: float,
    cal: dict[str, Any],
    stage: str,
    rows: list[dict[str, float | str]],
) -> float:
    cfg = with_overrides(base_cfg, {"source": {"eta_heat": eta}, "cooling": {"h_scale": h_scale}})
    result = run_simulation(df, cfg, store_snapshots=False)
    rmse = result.rmse_front
    span = max(1.0, float(np.nanmax(result.t_gt_front) - np.nanmin(result.t_gt_front)))
    prior_weight = float(cal.get("prior_weight", 0.0))
    prior = prior_weight * span * (
        np.log(max(eta, 1.0e-12) / max(eta0, 1.0e-12)) ** 2
        + np.log(max(h_scale, 1.0e-12) / max(h0, 1.0e-12)) ** 2
    )
    score = float(rmse + prior)
    rows.append(
        {
            "stage": stage,
            "eta_heat": float(eta),
            "h_scale": float(h_scale),
            "rmse": float(rmse),
            "score": score,
        }
    )
    return score
