from __future__ import annotations

import numpy as np
import pandas as pd

from .mesh import AnnulusMesh
from .telemetry import front_brake_power_per_disc


def pad_angles(times: np.ndarray, wheel_omega: np.ndarray, theta0: float = 0.0) -> np.ndarray:
    theta = np.zeros_like(times, dtype=float)
    theta[0] = float(theta0)
    if len(theta) < 2:
        return theta
    dt = np.diff(times)
    omega_mid = 0.5 * (wheel_omega[1:] + wheel_omega[:-1])
    theta[1:] = theta0 + np.cumsum(dt * omega_mid)
    return theta


def disc_power_history(df: pd.DataFrame, eta_heat: float) -> np.ndarray:
    return float(eta_heat) * front_brake_power_per_disc(df)


def gaussian_pad_load(
    mesh: AnnulusMesh,
    node_area: np.ndarray,
    power_w: float,
    theta_pad: float,
    source_config: dict,
) -> np.ndarray:
    power_w = max(0.0, float(power_w))
    if power_w == 0.0:
        return np.zeros(mesh.node_count, dtype=float)

    r_pad = float(source_config["r_pad"])
    sigma_r = max(1.0e-9, float(source_config["sigma_r"]))
    sigma_theta = max(1.0e-9, float(source_config["sigma_theta"]))

    dtheta = np.arctan2(np.sin(mesh.node_theta - theta_pad), np.cos(mesh.node_theta - theta_pad))
    radial = np.exp(-0.5 * ((mesh.node_r - r_pad) / sigma_r) ** 2)
    angular = np.exp(-0.5 * (dtheta / sigma_theta) ** 2)
    weights = radial * angular * node_area
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return np.zeros(mesh.node_count, dtype=float)
    return power_w * weights / total


def moving_gaussian_pad_load(
    mesh: AnnulusMesh,
    node_area: np.ndarray,
    power_w: float,
    theta_start: float,
    theta_end: float,
    source_config: dict,
) -> np.ndarray:
    power_w = max(0.0, float(power_w))
    if power_w == 0.0:
        return np.zeros(mesh.node_count, dtype=float)

    travel = abs(float(theta_end) - float(theta_start))
    max_samples = max(1, int(source_config.get("time_average_samples", 8)))
    sigma_theta = max(1.0e-9, float(source_config["sigma_theta"]))

    if travel >= 2.0 * np.pi:
        return ring_averaged_pad_load(mesh, node_area, power_w, source_config)

    samples = min(max_samples, max(1, int(np.ceil(travel / max(0.75 * sigma_theta, 1.0e-6)))))
    if samples <= 1:
        return gaussian_pad_load(mesh, node_area, power_w, theta_end, source_config)

    load = np.zeros(mesh.node_count, dtype=float)
    for theta in np.linspace(theta_start, theta_end, samples):
        load += gaussian_pad_load(mesh, node_area, power_w, float(theta), source_config)
    return load / float(samples)


def ring_averaged_pad_load(
    mesh: AnnulusMesh,
    node_area: np.ndarray,
    power_w: float,
    source_config: dict,
) -> np.ndarray:
    r_pad = float(source_config["r_pad"])
    sigma_r = max(1.0e-9, float(source_config["sigma_r"]))
    radial = np.exp(-0.5 * ((mesh.node_r - r_pad) / sigma_r) ** 2)
    weights = radial * node_area
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return np.zeros(mesh.node_count, dtype=float)
    return max(0.0, float(power_w)) * weights / total
