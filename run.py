from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brake_fem.calibration import calibrate_parameters
from brake_fem.config import load_config, save_json, with_overrides
from brake_fem.solver import run_simulation
from brake_fem.telemetry import load_telemetry, validate_telemetry
from brake_fem.visualize import save_telemetry_plot, write_simulation_outputs


DEFAULT_DATA = ROOT / "data" / "monza_flying_lap_selected_telemetry.csv"
DEFAULT_CONFIG = ROOT / "config" / "default.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Telemetry-driven brake-disc FEM simulator.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate-data", help="Validate telemetry schema and ranges.")
    p_validate.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p_validate.add_argument("--out", type=Path, default=None, help="Optional directory for validation plot.")

    p_sim = sub.add_parser("simulate", help="Run thermal FEM simulation.")
    p_sim.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p_sim.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_sim.add_argument("--out", type=Path, required=True)
    p_sim.add_argument("--stride", type=int, default=None, help="Override telemetry sample stride.")
    p_sim.add_argument("--no-gif", action="store_true", help="Skip GIF generation.")
    p_sim.add_argument("--no-stress", action="store_true", help="Skip thermoelastic stress post-processing.")

    p_cal = sub.add_parser("calibrate", help="Fit eta_heat and h_scale against front brake GT.")
    p_cal.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p_cal.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_cal.add_argument("--out", type=Path, required=True)
    p_cal.add_argument("--no-gif", action="store_true", help="Skip GIF generation for final calibrated run.")
    p_cal.add_argument("--no-stress", action="store_true", help="Skip thermoelastic stress post-processing for final run.")

    sub.add_parser("test", help="Run unittest suite.")

    args = parser.parse_args(argv)
    if args.command == "validate-data":
        return cmd_validate(args)
    if args.command == "simulate":
        return cmd_simulate(args)
    if args.command == "calibrate":
        return cmd_calibrate(args)
    if args.command == "test":
        return cmd_test()
    raise AssertionError(args.command)


def cmd_validate(args: argparse.Namespace) -> int:
    df = load_telemetry(args.data)
    validation = validate_telemetry(df)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        save_telemetry_plot(df, args.out / "fig_telemetry.png")
        save_json(validation.as_dict(), args.out / "validation.json")
    print(json.dumps(validation.as_dict(), indent=2, sort_keys=True))
    return 0 if not validation.missing_columns else 2


def cmd_simulate(args: argparse.Namespace) -> int:
    df = load_telemetry(args.data)
    validation = validate_telemetry(df)
    cfg = load_config(args.config)
    overrides: dict[str, object] = {"solver": {}}
    if args.stride is not None:
        overrides["solver"]["sample_stride"] = int(args.stride)
    if args.no_gif:
        overrides["solver"]["animation_frames"] = 0
    if args.no_stress:
        overrides["thermoelastic"] = {"enabled": False}
    cfg = with_overrides(cfg, overrides)
    result = run_simulation(df, cfg, store_snapshots=True)
    summary = write_simulation_outputs(result, df, validation, args.out, make_gif=not args.no_gif)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    df = load_telemetry(args.data)
    validation = validate_telemetry(df)
    cfg = load_config(args.config)
    if args.no_gif:
        cfg = with_overrides(cfg, {"solver": {"animation_frames": 0}})
    if args.no_stress:
        cfg = with_overrides(cfg, {"thermoelastic": {"enabled": False}})

    args.out.mkdir(parents=True, exist_ok=True)
    calibration = calibrate_parameters(df, cfg)
    calibration.trials.to_csv(args.out / "calibration_trials.csv", index=False)
    save_json(calibration.params_dict(), args.out / "params_fit.json")

    result = run_simulation(df, calibration.best_config, store_snapshots=True)
    summary = write_simulation_outputs(result, df, validation, args.out, make_gif=not args.no_gif)
    summary["calibration"] = calibration.params_dict()
    save_json(summary, args.out / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_test() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
