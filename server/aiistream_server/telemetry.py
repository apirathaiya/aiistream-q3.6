from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

from .safety import SafetyState


def parse_ioreg_battery(raw: str) -> dict:
    def num(name):
        m = re.search(rf'"{re.escape(name)}"\s*=\s*([0-9]+)', raw)
        return None if not m else int(m.group(1))
    def yn(name):
        m = re.search(rf'"{re.escape(name)}"\s*=\s*(Yes|No)', raw)
        return None if not m else (m.group(1) == "Yes")
    t = num("Temperature")
    vt = num("VirtualTemperature")
    return {
        "battery_c": None if t is None else t / 100.0,
        "battery_virtual_c": None if vt is None else vt / 100.0,
        "charge_percent": num("CurrentCapacity"),
        "external_connected": yn("ExternalConnected"),
    }
def parse_swapusage(raw: str) -> int | None:
    m = re.search(r"used\s*=\s*([0-9.]+)([KMG])", raw)
    if not m:
        return None
    mult = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000}[m.group(2)]
    return int(float(m.group(1)) * mult)


def parse_thermal_state(raw: str) -> int:
    value = int(raw.strip())
    if value not in (0, 1, 2, 3):
        raise ValueError(f"invalid thermalState {value}")
    return value


def pressure_label(level: int | None) -> str:
    return {1: "normal", 2: "warning", 4: "critical"}.get(level, "unknown")


def _run(args: list[str], timeout: float = 3.0) -> str:
    return subprocess.check_output(args, text=True, timeout=timeout).strip()


class SystemTelemetrySource:
    def __init__(self, thermal_helper: str | Path):
        self.thermal_helper = Path(thermal_helper)
    def sample(self) -> dict:
        if not self.thermal_helper.exists():
            raise RuntimeError(f"thermal helper missing: {self.thermal_helper}")
        battery = parse_ioreg_battery(_run(["/usr/sbin/ioreg", "-r", "-c", "AppleSmartBattery"]))
        thermal_state = parse_thermal_state(_run([str(self.thermal_helper)]))
        pressure = int(_run(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"]))
        swap_raw = _run(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **battery,
            "thermal_state": thermal_state,
            "swap_used_bytes": parse_swapusage(swap_raw),
            "memory_pressure_level": pressure,
            "memory_pressure_label": pressure_label(pressure),
            "mlx_active_bytes": int(mx.get_active_memory()),
            "mlx_peak_bytes": int(mx.get_peak_memory()),
            "mlx_cache_bytes": int(mx.get_cache_memory()),
        }


class TelemetryMonitor:
    def __init__(self, source, safety: SafetyState, *, poll_seconds: float = 5.0, log_path: str | Path | None = None):
        self.source = source
        self.safety = safety
        self.poll_seconds = float(poll_seconds)
        self.log_path = None if log_path is None else Path(log_path)
        self._lock = threading.Lock()
        self._latest: dict = {}
        self._last_error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="qwen36-telemetry", daemon=True)
        self._last_pressure = None
        self._swap4 = False
        self._swap8 = False

    def sample_once(self) -> dict:
        sample = self.source.sample()
        self.safety.set_memory_pressure_level(sample.get("memory_pressure_level"))
        self._record_events(sample)
        with self._lock:
            self._latest = dict(sample)
            self._last_error = None
        return sample

    def start(self) -> None:
        self.sample_once()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(2.0, self.poll_seconds + 1.0))

    def latest(self) -> dict:
        with self._lock:
            return {"sample": dict(self._latest), "error": self._last_error}
    def _append_event(self, event: dict) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def _record_events(self, sample: dict) -> None:
        pressure = sample.get("memory_pressure_level")
        if pressure != self._last_pressure:
            self._append_event({
                "timestamp": sample.get("timestamp"),
                "event": "memory_pressure_change",
                "previous": self._last_pressure,
                "current": pressure,
            })
            self._last_pressure = pressure
        swap = sample.get("swap_used_bytes")
        if swap is not None:
            if swap >= 4_000_000_000 and not self._swap4:
                self._append_event({"timestamp": sample.get("timestamp"), "event": "swap_crossed_4GB_observation", "swap_used_bytes": swap})
                self._swap4 = True
            if swap >= 8_000_000_000 and not self._swap8:
                self._append_event({"timestamp": sample.get("timestamp"), "event": "swap_crossed_8GB_observation", "swap_used_bytes": swap})
                self._swap8 = True
    def _run(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                self.sample_once()
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
