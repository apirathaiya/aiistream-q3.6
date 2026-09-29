from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    reason: str | None = None


class SafetyState:
    def __init__(self):
        self._lock = threading.Lock()
        self._memory_critical = False
        self._throughput_collapse = False
        self._collapse_detail = None

    def set_memory_pressure_level(self, level: int | None) -> None:
        with self._lock:
            self._memory_critical = (level == 4)

    def set_throughput_collapse(self, active: bool, detail=None) -> None:
        with self._lock:
            self._throughput_collapse = bool(active)
            self._collapse_detail = detail if active else None
    def admission_decision(self, *, memory_stop: bool, throughput_stop: bool) -> AdmissionDecision:
        with self._lock:
            if memory_stop and self._memory_critical:
                return AdmissionDecision(False, "memory_pressure_critical")
            if throughput_stop and self._throughput_collapse:
                return AdmissionDecision(False, "throughput_collapse")
            return AdmissionDecision(True, None)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "memory_pressure_critical": self._memory_critical,
                "throughput_collapse": self._throughput_collapse,
                "throughput_collapse_detail": self._collapse_detail,
            }


class ThroughputCollapseDetector:
    def __init__(self, safety: SafetyState, *, fraction: float = 0.5,
                 collapse_seconds: float = 60.0, early_seconds: float = 60.0):
        self.safety = safety
        self.fraction = float(fraction)
        self.collapse_seconds = float(collapse_seconds)
        self.early_seconds = float(early_seconds)
        self.reset()
    def reset(self, start_time: float | None = None) -> None:
        self.start_time = start_time
        self.early_rates: list[float] = []
        self.early_median: float | None = None
        self.below_since: float | None = None
        self.safety.set_throughput_collapse(False)

    def observe(self, rate: float, now: float) -> bool:
        if self.start_time is None:
            self.start_time = now
        elapsed = now - self.start_time
        if self.early_median is None:
            if elapsed <= self.early_seconds:
                self.early_rates.append(float(rate))
                return False
            if not self.early_rates:
                self.early_rates.append(float(rate))
            self.early_median = float(statistics.median(self.early_rates))

        threshold = self.early_median * self.fraction
        if rate < threshold:
            if self.below_since is None:
                self.below_since = now
            active = (now - self.below_since) >= self.collapse_seconds
            if active:
                self.safety.set_throughput_collapse(True, {
                    "early_median_tok_s": self.early_median,
                    "threshold_tok_s": threshold,
                    "current_tok_s": float(rate),
                    "below_for_seconds": now - self.below_since,
                })
            return active
        self.below_since = None
        self.safety.set_throughput_collapse(False)
        return False

    def finish(self) -> None:
        self.safety.set_throughput_collapse(False)


class AdmissionController:
    def __init__(self, *, queue_limit: int, safety: SafetyState,
                 memory_stop: bool = True, throughput_stop: bool = True):
        self.queue_limit = int(queue_limit)
        self.safety = safety
        self.memory_stop = bool(memory_stop)
        self.throughput_stop = bool(throughput_stop)
        self._lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._occupancy = 0
        self._active_generation = False

    @property
    def capacity(self) -> int:
        return 1 + self.queue_limit
    def try_admit(self):
        decision = self.safety.admission_decision(
            memory_stop=self.memory_stop,
            throughput_stop=self.throughput_stop,
        )
        if not decision.allowed:
            return None, decision.reason
        with self._lock:
            if self._occupancy >= self.capacity:
                return None, "queue_full"
            self._occupancy += 1
        return AdmissionTicket(self), None

    def _begin_generation(self):
        self._generation_lock.acquire()
        with self._lock:
            self._active_generation = True

    def _finish_generation(self):
        with self._lock:
            self._active_generation = False
        self._generation_lock.release()

    def _release_admission(self):
        with self._lock:
            self._occupancy -= 1

    def snapshot(self) -> dict:
        with self._lock:
            active = 1 if self._active_generation else 0
            return {
                "capacity": self.capacity,
                "occupancy": self._occupancy,
                "active_generation": bool(active),
                "queue_depth": max(0, self._occupancy - active),
            }
class AdmissionTicket:
    def __init__(self, controller: AdmissionController):
        self.controller = controller
        self.generation_acquired = False
        self.released = False

    def begin_generation(self) -> None:
        if self.released:
            raise RuntimeError("ticket already released")
        self.controller._begin_generation()
        self.generation_acquired = True

    def release(self) -> None:
        if self.released:
            return
        if self.generation_acquired:
            self.controller._finish_generation()
            self.generation_acquired = False
        self.controller._release_admission()
        self.released = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
