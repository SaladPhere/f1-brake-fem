from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brake_fem.telemetry import load_telemetry, validate_telemetry


class TelemetryTests(unittest.TestCase):
    def test_monza_selected_telemetry_has_expected_derived_columns(self) -> None:
        df = load_telemetry(ROOT / "data" / "monza_flying_lap_selected_telemetry.csv")
        validation = validate_telemetry(df)
        self.assertEqual(validation.missing_columns, [])
        self.assertGreater(validation.rows, 1000)
        self.assertGreater(validation.lap_duration_s, 80.0)
        self.assertLess(validation.lap_duration_s, 95.0)
        self.assertGreaterEqual(validation.accel_spike_count, 1)


if __name__ == "__main__":
    unittest.main()
