from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_REQUIRED_COLUMNS = [
    "lap_s",
    "Ground Speed",
    "Brake Pos",
    "Brake Torque FL",
    "Brake Torque FR",
    "Wheel Angular Speed FL",
    "Wheel Angular Speed FR",
    "Brake Temp FL",
    "Brake Temp FR",
]


@dataclass(frozen=True)
class TelemetryValidation:
    rows: int
    lap_start_s: float
    lap_end_s: float
    lap_duration_s: float
    distance_m: float | None
    missing_columns: list[str]
    accel_min_g: float | None
    accel_max_g: float | None
    accel_spike_count: int
    speed_min_ms: float
    speed_max_ms: float
    brake_pos_max: float
    front_temp_min_c: float
    front_temp_max_c: float

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "lap_start_s": self.lap_start_s,
            "lap_end_s": self.lap_end_s,
            "lap_duration_s": self.lap_duration_s,
            "distance_m": self.distance_m,
            "missing_columns": self.missing_columns,
            "accel_min_g": self.accel_min_g,
            "accel_max_g": self.accel_max_g,
            "accel_spike_count_abs_gt_5g": self.accel_spike_count,
            "speed_min_ms": self.speed_min_ms,
            "speed_max_ms": self.speed_max_ms,
            "brake_pos_max": self.brake_pos_max,
            "front_temp_min_c": self.front_temp_min_c,
            "front_temp_max_c": self.front_temp_max_c,
        }


def load_telemetry(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = ensure_derived_columns(df)
    df = clean_telemetry(df)
    return df


def ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "speed_ms" not in df.columns and "Ground Speed" in df.columns:
        df["speed_ms"] = df["Ground Speed"] / 3.6
    if "long_accel_ms2" not in df.columns and "CG Accel Longitudinal" in df.columns:
        df["long_accel_ms2"] = df["CG Accel Longitudinal"] * 9.80665
    if "front_brake_torque" not in df.columns and _has(df, "Brake Torque FL", "Brake Torque FR"):
        df["front_brake_torque"] = 0.5 * (df["Brake Torque FL"] + df["Brake Torque FR"])
    if "rear_brake_torque" not in df.columns and _has(df, "Brake Torque RL", "Brake Torque RR"):
        df["rear_brake_torque"] = 0.5 * (df["Brake Torque RL"] + df["Brake Torque RR"])
    if "front_wheel_omega" not in df.columns and _has(
        df, "Wheel Angular Speed FL", "Wheel Angular Speed FR"
    ):
        df["front_wheel_omega"] = 0.5 * (
            df["Wheel Angular Speed FL"] + df["Wheel Angular Speed FR"]
        )
    if "rear_wheel_omega" not in df.columns and _has(
        df, "Wheel Angular Speed RL", "Wheel Angular Speed RR"
    ):
        df["rear_wheel_omega"] = 0.5 * (
            df["Wheel Angular Speed RL"] + df["Wheel Angular Speed RR"]
        )
    if "front_brake_temp" not in df.columns and _has(df, "Brake Temp FL", "Brake Temp FR"):
        df["front_brake_temp"] = 0.5 * (df["Brake Temp FL"] + df["Brake Temp FR"])
    if "rear_brake_temp" not in df.columns and _has(df, "Brake Temp RL", "Brake Temp RR"):
        df["rear_brake_temp"] = 0.5 * (df["Brake Temp RL"] + df["Brake Temp RR"])
    return df


def clean_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "lap_s" not in df.columns:
        raise ValueError("Telemetry must contain a 'lap_s' column.")

    df = df.sort_values("lap_s").drop_duplicates("lap_s", keep="first")
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    for col in numeric_cols:
        df[col] = df[col].interpolate(limit_direction="both")
        df[col] = df[col].ffill().bfill()

    if "speed_ms" in df.columns:
        df["speed_ms"] = df["speed_ms"].clip(lower=0.0)
    if "Brake Pos" in df.columns:
        df["Brake Pos"] = df["Brake Pos"].clip(lower=0.0)
    for col in [
        "Brake Torque FL",
        "Brake Torque FR",
        "Brake Torque RL",
        "Brake Torque RR",
        "front_brake_torque",
        "rear_brake_torque",
        "front_wheel_omega",
        "rear_wheel_omega",
    ]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)
    return df.reset_index(drop=True)


def validate_telemetry(df: pd.DataFrame) -> TelemetryValidation:
    missing = [col for col in BASE_REQUIRED_COLUMNS + ["front_brake_temp"] if col not in df.columns]
    lap = df["lap_s"].to_numpy(float)
    distance = _distance_covered(df)
    accel_min = accel_max = None
    spike_count = 0
    if "CG Accel Longitudinal" in df.columns:
        accel = df["CG Accel Longitudinal"].to_numpy(float)
        accel_min = float(np.nanmin(accel))
        accel_max = float(np.nanmax(accel))
        spike_count = int(np.sum(np.abs(accel) > 5.0))
    return TelemetryValidation(
        rows=int(len(df)),
        lap_start_s=float(lap[0]),
        lap_end_s=float(lap[-1]),
        lap_duration_s=float(lap[-1] - lap[0]),
        distance_m=distance,
        missing_columns=missing,
        accel_min_g=accel_min,
        accel_max_g=accel_max,
        accel_spike_count=spike_count,
        speed_min_ms=float(np.nanmin(df["speed_ms"])),
        speed_max_ms=float(np.nanmax(df["speed_ms"])),
        brake_pos_max=float(np.nanmax(df["Brake Pos"])),
        front_temp_min_c=float(np.nanmin(df["front_brake_temp"])),
        front_temp_max_c=float(np.nanmax(df["front_brake_temp"])),
    )


def front_brake_power_per_disc(df: pd.DataFrame) -> np.ndarray:
    if _has(df, "Brake Torque FL", "Brake Torque FR", "Wheel Angular Speed FL", "Wheel Angular Speed FR"):
        left = df["Brake Torque FL"].to_numpy(float) * df["Wheel Angular Speed FL"].to_numpy(float)
        right = df["Brake Torque FR"].to_numpy(float) * df["Wheel Angular Speed FR"].to_numpy(float)
        return np.maximum(0.0, 0.5 * (left + right))
    return np.maximum(
        0.0,
        df["front_brake_torque"].to_numpy(float) * df["front_wheel_omega"].to_numpy(float),
    )


def thin_telemetry(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    stride = max(1, int(stride))
    if stride == 1 or len(df) <= 2:
        return df.reset_index(drop=True)
    idx = list(range(0, len(df), stride))
    if idx[-1] != len(df) - 1:
        idx.append(len(df) - 1)
    return df.iloc[idx].reset_index(drop=True)


def _has(df: pd.DataFrame, *cols: Iterable[str]) -> bool:
    return all(col in df.columns for col in cols)


def _distance_covered(df: pd.DataFrame) -> float | None:
    if "Lap Distance" in df.columns:
        values = df["Lap Distance"].to_numpy(float)
        return float(np.nanmax(values) - np.nanmin(values))
    if "speed_ms" in df.columns and "lap_s" in df.columns:
        t = df["lap_s"].to_numpy(float)
        v = df["speed_ms"].to_numpy(float)
        return float(np.trapz(v, t))
    return None
