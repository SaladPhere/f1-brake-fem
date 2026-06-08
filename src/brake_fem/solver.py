from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .fem import FEMOperators, assemble_heat_operators, operator_matvec
from .mesh import AnnulusMesh, generate_annulus_mesh
from .source import disc_power_history, moving_gaussian_pad_load, pad_angles
from .telemetry import thin_telemetry


SIGMA_SB = 5.670374419e-8


@dataclass
class SimulationResult:
    mesh: AnnulusMesh
    operators: FEMOperators
    times: np.ndarray
    speed_ms: np.ndarray
    brake_pos: np.ndarray
    power_disc: np.ndarray
    h_conv: np.ndarray
    t_mean: np.ndarray
    t_max: np.ndarray
    t_gt_front: np.ndarray
    t_gt_rear: np.ndarray | None
    snapshots_t: np.ndarray
    snapshots: np.ndarray
    config: dict[str, Any]
    cg_iterations: np.ndarray
    cg_residuals: np.ndarray

    @property
    def rmse_front(self) -> float:
        if len(self.t_gt_front) == 0:
            return float("nan")
        return float(np.sqrt(np.mean((self.t_mean - self.t_gt_front) ** 2)))


def run_simulation(df: pd.DataFrame, config: dict[str, Any], store_snapshots: bool = True) -> SimulationResult:
    solver_cfg = config.get("solver", {})
    df = thin_telemetry(df, int(solver_cfg.get("sample_stride", 1)))

    geom = config["geometry"]
    mesh = generate_annulus_mesh(
        float(geom["r_inner"]),
        float(geom["r_outer"]),
        int(geom["n_radial"]),
        int(geom["n_theta"]),
    )
    ops = assemble_heat_operators(mesh, config)

    times = df["lap_s"].to_numpy(float)
    speed = df["speed_ms"].to_numpy(float)
    brake_pos = df["Brake Pos"].to_numpy(float)
    t_gt_front = df["front_brake_temp"].to_numpy(float)
    t_gt_rear = df["rear_brake_temp"].to_numpy(float) if "rear_brake_temp" in df.columns else None
    omega = df["front_wheel_omega"].to_numpy(float)

    theta_history = pad_angles(times, omega, float(config["source"].get("theta0", 0.0)))
    power = disc_power_history(df, float(config["source"].get("eta_heat", 1.0)))
    h_history = convection_history(speed, config)

    initial = solver_cfg.get("initial_temperature", "data_front")
    if isinstance(initial, str) and initial == "data_front":
        initial_c = float(t_gt_front[0])
    else:
        initial_c = float(initial)
    T = np.full(mesh.node_count, initial_c, dtype=float)

    n_steps = len(times)
    t_mean = np.zeros(n_steps, dtype=float)
    t_max = np.zeros(n_steps, dtype=float)
    cg_iterations = np.zeros(n_steps, dtype=int)
    cg_residuals = np.zeros(n_steps, dtype=float)

    snapshot_count = int(solver_cfg.get("snapshot_count", 8))
    animation_frames = int(solver_cfg.get("animation_frames", 0))
    if store_snapshots and animation_frames > 0:
        snapshot_count = max(snapshot_count, animation_frames)
    snapshot_ids = _snapshot_indices(n_steps, snapshot_count) if store_snapshots else np.asarray([], dtype=int)
    snapshot_lookup = {int(idx): pos for pos, idx in enumerate(snapshot_ids)}
    snapshots = (
        np.zeros((len(snapshot_ids), mesh.node_count), dtype=np.float32)
        if store_snapshots
        else np.zeros((0, mesh.node_count), dtype=np.float32)
    )

    record_weight = ops.node_area / max(1.0e-30, float(np.sum(ops.node_area)))
    t_mean[0] = float(np.sum(record_weight * T))
    t_max[0] = float(np.max(T))
    if 0 in snapshot_lookup:
        snapshots[snapshot_lookup[0]] = T.astype(np.float32)

    cg_tol = float(solver_cfg.get("cg_tol", 1.0e-6))
    cg_max_iter = int(solver_cfg.get("cg_max_iter", 100))
    ambient = float(config.get("cooling", {}).get("ambient_c", 25.0))
    emissivity = float(config.get("material", {}).get("emissivity", 0.0))

    for step in range(1, n_steps):
        dt = max(1.0e-9, float(times[step] - times[step - 1]))
        h = float(h_history[step])
        h_rad = radiation_linearization(T, ambient, emissivity)
        cooling_diag = ops.cooling_weights * (h + h_rad)
        diagonal = ops.mass / dt + ops.stiffness_diag + cooling_diag

        source = moving_gaussian_pad_load(
            mesh,
            ops.node_area,
            float(power[step]),
            float(theta_history[step - 1]),
            float(theta_history[step]),
            config["source"],
        )
        rhs = ops.mass * T / dt + source + cooling_diag * ambient
        T, iters, rel_res = pcg_solve(
            diagonal,
            ops.stiffness_i,
            ops.stiffness_j,
            ops.stiffness_v,
            rhs,
            x0=T,
            tol=cg_tol,
            max_iter=cg_max_iter,
        )

        cg_iterations[step] = iters
        cg_residuals[step] = rel_res
        t_mean[step] = float(np.sum(record_weight * T))
        t_max[step] = float(np.max(T))
        if step in snapshot_lookup:
            snapshots[snapshot_lookup[step]] = T.astype(np.float32)

    return SimulationResult(
        mesh=mesh,
        operators=ops,
        times=times,
        speed_ms=speed,
        brake_pos=brake_pos,
        power_disc=power,
        h_conv=h_history,
        t_mean=t_mean,
        t_max=t_max,
        t_gt_front=t_gt_front,
        t_gt_rear=t_gt_rear,
        snapshots_t=times[snapshot_ids] if len(snapshot_ids) else np.asarray([], dtype=float),
        snapshots=snapshots,
        config=config,
        cg_iterations=cg_iterations,
        cg_residuals=cg_residuals,
    )


def convection_history(speed_ms: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    cooling = config.get("cooling", {})
    h0 = float(cooling.get("h0", 0.0))
    h1 = float(cooling.get("h1", 0.0))
    alpha = float(cooling.get("alpha", 1.0))
    h_scale = float(cooling.get("h_scale", 1.0))
    return h_scale * (h0 + h1 * np.maximum(speed_ms, 0.0) ** alpha)


def radiation_linearization(temp_c: np.ndarray, ambient_c: float, emissivity: float) -> np.ndarray:
    if emissivity <= 0.0:
        return np.zeros_like(temp_c)
    temp_k = np.maximum(temp_c + 273.15, 1.0)
    ambient_k = max(float(ambient_c) + 273.15, 1.0)
    return emissivity * SIGMA_SB * (temp_k + ambient_k) * (temp_k**2 + ambient_k**2)


def pcg_solve(
    diagonal: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    values: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int, float]:
    x = x0.copy()
    r = b - operator_matvec(diagonal, i_idx, j_idx, values, x)
    norm_b = float(np.linalg.norm(b))
    if norm_b == 0.0:
        return np.zeros_like(b), 0, 0.0
    rel = float(np.linalg.norm(r) / norm_b)
    if rel <= tol:
        return x, 0, rel

    inv_diag = 1.0 / np.maximum(diagonal, 1.0e-30)
    z = inv_diag * r
    p = z.copy()
    rz_old = float(np.dot(r, z))
    if rz_old == 0.0:
        return x, 0, rel

    for it in range(1, max_iter + 1):
        Ap = operator_matvec(diagonal, i_idx, j_idx, values, p)
        denom = float(np.dot(p, Ap))
        if abs(denom) < 1.0e-30:
            break
        alpha = rz_old / denom
        x += alpha * p
        r -= alpha * Ap
        rel = float(np.linalg.norm(r) / norm_b)
        if rel <= tol:
            return x, it, rel
        z = inv_diag * r
        rz_new = float(np.dot(r, z))
        if abs(rz_old) < 1.0e-30:
            break
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    return x, max_iter, rel


def _snapshot_indices(n_steps: int, count: int) -> np.ndarray:
    if n_steps <= 0 or count <= 0:
        return np.asarray([], dtype=int)
    count = min(count, n_steps)
    return np.unique(np.linspace(0, n_steps - 1, count, dtype=int))
