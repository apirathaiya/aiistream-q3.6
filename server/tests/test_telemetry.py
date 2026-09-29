import unittest
from pathlib import Path

from aiistream_server.telemetry import parse_ioreg_battery, parse_swapusage, parse_thermal_state, pressure_label


class TelemetryTests(unittest.TestCase):
    def test_ioreg_fixture(self):
        raw = (Path(__file__).parent / "fixtures" / "ioreg_battery.txt").read_text()
        got = parse_ioreg_battery(raw)
        self.assertEqual(got["battery_c"], 30.29)
        self.assertEqual(got["battery_virtual_c"], 29.69)
        self.assertEqual(got["charge_percent"], 80)
        self.assertTrue(got["external_connected"])

    def test_swap_parser(self):
        raw = "total = 7168.00M  used = 5971.19M  free = 1196.81M  (encrypted)"
        self.assertEqual(parse_swapusage(raw), 5_971_190_000)

    def test_thermal_state_parser(self):
        fixture = (Path(__file__).parent / "fixtures" / "thermal_state.txt").read_text()
        self.assertEqual(parse_thermal_state(fixture), 1)
        self.assertEqual(parse_thermal_state("0\n"), 0)
        self.assertEqual(parse_thermal_state("3"), 3)
        with self.assertRaises(ValueError):
            parse_thermal_state("4")

    def test_pressure_labels(self):
        self.assertEqual(pressure_label(1), "normal")
        self.assertEqual(pressure_label(2), "warning")
        self.assertEqual(pressure_label(4), "critical")


if __name__ == "__main__":
    unittest.main()
