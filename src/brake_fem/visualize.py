from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .config import save_json
from .mesh import AnnulusMesh
from .solver import SimulationResult
from .telemetry import TelemetryValidation, front_brake_power_per_disc


RGB = tuple[int, int, int]
WHITE: RGB = (255, 255, 255)
BLACK: RGB = (20, 22, 24)
GRID: RGB = (218, 222, 226)
AXIS: RGB = (68, 74, 80)
BLUE: RGB = (38, 111, 213)
RED: RGB = (215, 58, 73)
ORANGE: RGB = (225, 132, 32)
GREEN: RGB = (31, 143, 88)
PURPLE: RGB = (124, 83, 174)


def write_simulation_outputs(
    result: SimulationResult,
    telemetry_df: pd.DataFrame,
    validation: TelemetryValidation,
    out_dir: str | Path,
    make_gif: bool = True,
) -> dict[str, object]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = result_timeseries(result)
    ts.to_csv(out / "temperature_timeseries.csv", index=False)
    np.savez_compressed(
        out / "temperature_snapshots.npz",
        times=result.snapshots_t,
        temperatures=result.snapshots,
        nodes=result.mesh.nodes,
        elements=result.mesh.elements,
    )

    save_telemetry_plot(telemetry_df, out / "fig_telemetry.png")
    save_fit_plot(result, out / "fig_fit.png")
    save_temperature_fields(result, out / "fig_temperature_fields.png")
    if make_gif and len(result.snapshots):
        save_temperature_animation(result, out / "temperature_animation.gif")

    summary = simulation_summary(result, validation)
    save_json(summary, out / "summary.json")
    return summary


def result_timeseries(result: SimulationResult) -> pd.DataFrame:
    data = {
        "lap_s": result.times,
        "speed_ms": result.speed_ms,
        "brake_pos": result.brake_pos,
        "T_mean": result.t_mean,
        "T_max": result.t_max,
        "T_gt_front": result.t_gt_front,
        "P_disc": result.power_disc,
        "h_v": result.h_conv,
        "cg_iterations": result.cg_iterations,
        "cg_relative_residual": result.cg_residuals,
    }
    if result.t_gt_rear is not None:
        data["T_gt_rear"] = result.t_gt_rear
    return pd.DataFrame(data)


def simulation_summary(result: SimulationResult, validation: TelemetryValidation) -> dict[str, object]:
    cfg = result.config
    return {
        "validation": validation.as_dict(),
        "mesh": {
            "nodes": result.mesh.node_count,
            "elements": result.mesh.element_count,
            "area_m2": result.operators.total_area,
            "thermal_capacity_j_per_k": result.operators.thermal_capacity_total,
        },
        "parameters": {
            "eta_heat": cfg["source"].get("eta_heat"),
            "h_scale": cfg["cooling"].get("h_scale"),
            "k_inplane": cfg["material"].get("k_inplane"),
            "rho": cfg["material"].get("rho"),
            "cp": cfg["material"].get("cp"),
        },
        "fit": {
            "rmse_front_c": result.rmse_front,
            "T_mean_min_c": float(np.min(result.t_mean)),
            "T_mean_max_c": float(np.max(result.t_mean)),
            "T_max_max_c": float(np.max(result.t_max)),
            "T_gt_front_min_c": float(np.min(result.t_gt_front)),
            "T_gt_front_max_c": float(np.max(result.t_gt_front)),
        },
        "solver": {
            "max_cg_iterations": int(np.max(result.cg_iterations)) if len(result.cg_iterations) else 0,
            "mean_cg_iterations": float(np.mean(result.cg_iterations)) if len(result.cg_iterations) else 0.0,
            "max_cg_relative_residual": float(np.max(result.cg_residuals)) if len(result.cg_residuals) else 0.0,
        },
    }


def save_telemetry_plot(df: pd.DataFrame, path: str | Path) -> None:
    t = df["lap_s"].to_numpy(float)
    power_kw = front_brake_power_per_disc(df) / 1000.0
    panels = [
        {
            "title": "Ground speed (km/h)",
            "series": [("speed", df["speed_ms"].to_numpy(float) * 3.6, BLUE)],
        },
        {
            "title": "Brake position (%)",
            "series": [("brake", df["Brake Pos"].to_numpy(float), RED)],
        },
        {
            "title": "Estimated one-disc braking power (kW)",
            "series": [("power", power_kw, ORANGE)],
        },
        {
            "title": "Brake temperature GT (C)",
            "series": [
                ("front", df["front_brake_temp"].to_numpy(float), GREEN),
                (
                    "rear",
                    df["rear_brake_temp"].to_numpy(float)
                    if "rear_brake_temp" in df.columns
                    else df["front_brake_temp"].to_numpy(float),
                    PURPLE,
                ),
            ],
        },
    ]
    if "CG Accel Longitudinal" in df.columns:
        panels.append(
            {
                "title": "Longitudinal acceleration (G)",
                "series": [("a_x", df["CG Accel Longitudinal"].to_numpy(float), BLACK)],
            }
        )
    draw_line_panels(path, "Telemetry validation", t, panels)


def save_fit_plot(result: SimulationResult, path: str | Path) -> None:
    panels = [
        {
            "title": f"Mean disc temperature vs GT, RMSE={result.rmse_front:.2f} C",
            "series": [
                ("sim mean", result.t_mean, BLUE),
                ("GT front", result.t_gt_front, RED),
                ("sim max", result.t_max, ORANGE),
            ],
        },
        {
            "title": "Brake power and convection",
            "series": [
                ("P_disc kW", result.power_disc / 1000.0, ORANGE),
                ("h(v)", result.h_conv, GREEN),
            ],
        },
    ]
    draw_line_panels(path, "FEM fit summary", result.times, panels, height=620)


def save_temperature_fields(result: SimulationResult, path: str | Path, max_panels: int = 8) -> None:
    if len(result.snapshots) == 0:
        return
    ids = np.unique(np.linspace(0, len(result.snapshots) - 1, min(max_panels, len(result.snapshots)), dtype=int))
    temps = result.snapshots[ids]
    times = result.snapshots_t[ids]
    vmin = float(np.min(result.snapshots))
    vmax = float(np.max(result.snapshots))
    panel_size = 250
    title_h = 32
    cols = min(4, len(ids))
    rows = int(np.ceil(len(ids) / cols))
    fig = Image.new("RGB", (cols * panel_size, rows * (panel_size + title_h)), WHITE)
    draw = ImageDraw.Draw(fig)
    font = _font()
    for n, (temp, time_s) in enumerate(zip(temps, times)):
        field = field_image(result.mesh, temp, panel_size, vmin, vmax)
        x = (n % cols) * panel_size
        y = (n // cols) * (panel_size + title_h)
        draw.text((x + 8, y + 8), f"t={time_s:5.1f}s", fill=BLACK, font=font)
        fig.paste(field, (x, y + title_h))
    _draw_colorbar(fig, vmin, vmax)
    fig.save(path)


def save_temperature_animation(result: SimulationResult, path: str | Path) -> None:
    if len(result.snapshots) == 0:
        return
    vmin = float(np.min(result.snapshots))
    vmax = float(np.max(result.snapshots))
    frames = []
    for temp, time_s, mean_t, max_t in zip(
        result.snapshots,
        result.snapshots_t,
        _interp_at(result.times, result.t_mean, result.snapshots_t),
        _interp_at(result.times, result.t_max, result.snapshots_t),
    ):
        img = Image.new("RGB", (420, 470), WHITE)
        field = field_image(result.mesh, temp, 400, vmin, vmax)
        img.paste(field, (10, 48))
        draw = ImageDraw.Draw(img)
        font = _font()
        draw.text((14, 12), f"Brake-disc FEM temperature, t={time_s:5.1f}s", fill=BLACK, font=font)
        draw.text((14, 30), f"mean={mean_t:7.1f} C   max={max_t:7.1f} C", fill=BLACK, font=font)
        frames.append(img)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=80, loop=0)


def draw_line_panels(
    path: str | Path,
    title: str,
    x: np.ndarray,
    panels: list[dict[str, object]],
    width: int = 1200,
    height: int = 820,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(img)
    font = _font()
    draw.text((28, 18), title, fill=BLACK, font=font)

    margin_l, margin_r = 78, 28
    top = 50
    gap = 22
    panel_h = int((height - top - 36 - gap * (len(panels) - 1)) / max(1, len(panels)))
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if x_max <= x_min:
        x_max = x_min + 1.0

    for p_idx, panel in enumerate(panels):
        y0 = top + p_idx * (panel_h + gap)
        y1 = y0 + panel_h
        _draw_axes(draw, margin_l, y0, width - margin_r, y1)
        draw.text((margin_l, y0 - 16), str(panel["title"]), fill=BLACK, font=font)

        series = panel["series"]
        all_y = np.concatenate([np.asarray(s[1], dtype=float) for s in series])
        y_min, y_max = _range(all_y)
        _draw_y_ticks(draw, font, margin_l, y0, width - margin_r, y1, y_min, y_max)
        for label, y_values, color in series:
            _draw_series(draw, x, np.asarray(y_values, dtype=float), color, margin_l, y0, width - margin_r, y1, x_min, x_max, y_min, y_max)
        _draw_legend(draw, font, series, width - margin_r - 170, y0 + 8)

    _draw_x_ticks(draw, font, margin_l, height - 30, width - margin_r, x_min, x_max)
    img.save(path)


def field_image(mesh: AnnulusMesh, temp: np.ndarray, size: int, vmin: float, vmax: float) -> Image.Image:
    coords = np.linspace(-mesh.radii[-1] * 1.08, mesh.radii[-1] * 1.08, size)
    xx, yy = np.meshgrid(coords, coords[::-1])
    rr = np.sqrt(xx**2 + yy**2)
    theta = np.mod(np.arctan2(yy, xx), 2.0 * np.pi)
    mask = (rr >= mesh.radii[0]) & (rr <= mesh.radii[-1])

    radial_pos = (rr - mesh.radii[0]) / (mesh.radii[-1] - mesh.radii[0])
    radial_idx = np.clip(np.rint(radial_pos * mesh.n_radial).astype(int), 0, mesh.n_radial)
    theta_idx = np.mod(np.rint(theta / (2.0 * np.pi) * mesh.n_theta).astype(int), mesh.n_theta)
    node_idx = radial_idx * mesh.n_theta + theta_idx

    values = temp[node_idx]
    arr = np.full((size, size, 3), 248, dtype=np.uint8)
    colors = colormap_array(values, vmin, vmax)
    arr[mask] = colors[mask]
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    center = size / 2.0
    scale = size / (2.0 * mesh.radii[-1] * 1.08)
    for radius in [mesh.radii[0], mesh.radii[-1]]:
        pix = radius * scale
        box = [center - pix, center - pix, center + pix, center + pix]
        draw.ellipse(box, outline=(40, 44, 48), width=2)
    return img


def colormap_array(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    span = max(1.0e-9, vmax - vmin)
    z = np.clip((values - vmin) / span, 0.0, 1.0)
    stops = np.array(
        [
            [28, 41, 120],
            [36, 123, 204],
            [50, 185, 180],
            [248, 213, 72],
            [214, 55, 43],
        ],
        dtype=float,
    )
    scaled = z * (len(stops) - 1)
    lo = np.floor(scaled).astype(int)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    frac = scaled - lo
    rgb = (1.0 - frac[..., None]) * stops[lo] + frac[..., None] * stops[hi]
    return rgb.astype(np.uint8)


def _draw_axes(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int) -> None:
    draw.rectangle([x0, y0, x1, y1], outline=AXIS, width=1)
    for n in range(1, 4):
        y = y0 + n * (y1 - y0) / 4.0
        draw.line([x0, y, x1, y], fill=GRID)


def _draw_series(
    draw: ImageDraw.ImageDraw,
    x: np.ndarray,
    y: np.ndarray,
    color: RGB,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    if len(x) < 2:
        return
    px = x0 + (x - x_min) / max(1.0e-12, x_max - x_min) * (x1 - x0)
    py = y1 - (y - y_min) / max(1.0e-12, y_max - y_min) * (y1 - y0)
    points = [(float(a), float(b)) for a, b in zip(px, py) if np.isfinite(a) and np.isfinite(b)]
    if len(points) >= 2:
        draw.line(points, fill=color, width=2)


def _draw_y_ticks(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    y_min: float,
    y_max: float,
) -> None:
    for value in np.linspace(y_min, y_max, 5):
        y = y1 - (value - y_min) / max(1.0e-12, y_max - y_min) * (y1 - y0)
        draw.text((6, y - 6), _fmt(value), fill=AXIS, font=font)


def _draw_x_ticks(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    x0: int,
    y: int,
    x1: int,
    x_min: float,
    x_max: float,
) -> None:
    for value in np.linspace(x_min, x_max, 6):
        x = x0 + (value - x_min) / max(1.0e-12, x_max - x_min) * (x1 - x0)
        draw.text((x - 12, y), _fmt(value), fill=AXIS, font=font)
    draw.text(((x0 + x1) / 2 - 28, y + 14), "lap_s", fill=AXIS, font=font)


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    series: Iterable[tuple[str, np.ndarray, RGB]],
    x: int,
    y: int,
) -> None:
    for offset, (label, _values, color) in enumerate(series):
        yy = y + 16 * offset
        draw.line([x, yy + 6, x + 22, yy + 6], fill=color, width=3)
        draw.text((x + 28, yy), str(label), fill=BLACK, font=font)


def _draw_colorbar(img: Image.Image, vmin: float, vmax: float) -> None:
    draw = ImageDraw.Draw(img)
    font = _font()
    width, height = img.size
    bar_w = min(280, width - 32)
    x0 = width - bar_w - 16
    y0 = height - 22
    gradient = np.linspace(vmin, vmax, bar_w)
    colors = colormap_array(gradient.reshape(1, -1), vmin, vmax)[0]
    for i, color in enumerate(colors):
        draw.line([x0 + i, y0, x0 + i, y0 + 10], fill=tuple(int(c) for c in color))
    draw.text((x0, y0 - 14), f"{vmin:.1f} C", fill=BLACK, font=font)
    draw.text((x0 + bar_w - 62, y0 - 14), f"{vmax:.1f} C", fill=BLACK, font=font)


def _range(values: np.ndarray) -> tuple[float, float]:
    y_min = float(np.nanmin(values))
    y_max = float(np.nanmax(values))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return 0.0, 1.0
    if y_max <= y_min:
        return y_min - 0.5, y_max + 0.5
    pad = 0.06 * (y_max - y_min)
    return y_min - pad, y_max + pad


def _fmt(value: float) -> str:
    if abs(value) >= 1000.0:
        return f"{value:.0f}"
    if abs(value) >= 100.0:
        return f"{value:.1f}"
    if abs(value) >= 10.0:
        return f"{value:.2f}"
    return f"{value:.3g}"


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _interp_at(x: np.ndarray, y: np.ndarray, xp: np.ndarray) -> np.ndarray:
    return np.interp(xp, x, y)
