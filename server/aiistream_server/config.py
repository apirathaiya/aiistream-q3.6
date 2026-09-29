from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TelemetryConfig:
    poll_seconds: float = 5.0


@dataclass(frozen=True)
class SafetyConfig:
    memory_pressure_stop: bool = True
    throughput_collapse_stop: bool = True
    collapse_fraction: float = 0.50
    collapse_window_seconds: float = 60.0


@dataclass(frozen=True)
class ServiceConfig:
    port: int = 8081
    queue_limit: int = 4
    max_context: int = 32768
    max_context_hard_ceiling: int = 65536
    expert_read_path: str = "prefetch"
    telemetry: TelemetryConfig = TelemetryConfig()
    safety: SafetyConfig = SafetyConfig()

    def validate(self) -> "ServiceConfig":
        if not (1 <= self.port <= 65535):
            raise ConfigError("port must be in 1..65535")
        if self.port == 8080:
            raise ConfigError("port 8080 is reserved (the default port of other local model servers)")
        if self.expert_read_path not in ("prefetch", "direct", "parallel"):
            raise ConfigError("expert_read_path must be prefetch, direct or parallel")
        if self.queue_limit < 0:
            raise ConfigError("queue_limit must be >= 0")
        if not (1 <= self.max_context <= 65536):
            raise ConfigError("max_context must be in 1..65536")
        if not (self.max_context <= self.max_context_hard_ceiling <= 65536):
            raise ConfigError("max_context_hard_ceiling must be >= max_context and <= 65536")
        if self.telemetry.poll_seconds <= 0:
            raise ConfigError("telemetry.poll_seconds must be > 0")
        if not (0 < self.safety.collapse_fraction < 1):
            raise ConfigError("safety.collapse_fraction must be between 0 and 1")
        if self.safety.collapse_window_seconds <= 0:
            raise ConfigError("safety.collapse_window_seconds must be > 0")
        return self
def _reject_unknown(obj: dict, allowed: set[str], where: str) -> None:
    extra = sorted(set(obj) - allowed)
    if extra:
        raise ConfigError(f"unsupported config key(s) in {where}: {', '.join(extra)}")


def load_config(path: str | Path) -> ServiceConfig:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ConfigError("config root must be an object")
    _reject_unknown(raw, {
        "port", "queue_limit", "max_context", "max_context_hard_ceiling",
        "telemetry", "safety", "expert_read_path"
    }, "root")

    telemetry_raw = raw.get("telemetry", {})
    safety_raw = raw.get("safety", {})
    if not isinstance(telemetry_raw, dict) or not isinstance(safety_raw, dict):
        raise ConfigError("telemetry and safety must be objects")
    _reject_unknown(telemetry_raw, {"poll_seconds"}, "telemetry")
    _reject_unknown(safety_raw, {
        "memory_pressure_stop", "throughput_collapse_stop",
        "collapse_fraction", "collapse_window_seconds"
    }, "safety")
    cfg = ServiceConfig(
        port=int(raw.get("port", 8081)),
        queue_limit=int(raw.get("queue_limit", 4)),
        max_context=int(raw.get("max_context", 32768)),
        max_context_hard_ceiling=int(raw.get("max_context_hard_ceiling", 65536)),
        expert_read_path=raw.get("expert_read_path", "prefetch"),
        telemetry=TelemetryConfig(
            poll_seconds=float(telemetry_raw.get("poll_seconds", 5.0)),
        ),
        safety=SafetyConfig(
            memory_pressure_stop=bool(safety_raw.get("memory_pressure_stop", True)),
            throughput_collapse_stop=bool(safety_raw.get("throughput_collapse_stop", True)),
            collapse_fraction=float(safety_raw.get("collapse_fraction", 0.50)),
            collapse_window_seconds=float(safety_raw.get("collapse_window_seconds", 60.0)),
        ),
    )
    return cfg.validate()
