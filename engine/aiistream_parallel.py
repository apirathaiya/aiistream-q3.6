#!/usr/bin/env python3
"""AiiStream parallel expert reader: routed experts are read by a pool of eight workers."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import atexit
import os
import threading
import time

import mlx.core as mx
import numpy as np

import aiistream_model as p1

NP_DTYPES = p1.NP_DTYPES
_PROCESS_EXECUTOR = None
_PROCESS_EXECUTOR_CREATION_COUNT = 0
_PROCESS_EXECUTOR_LOCK = threading.Lock()


def _process_executor():
    global _PROCESS_EXECUTOR, _PROCESS_EXECUTOR_CREATION_COUNT
    with _PROCESS_EXECUTOR_LOCK:
        if _PROCESS_EXECUTOR is None:
            _PROCESS_EXECUTOR = ThreadPoolExecutor(
                max_workers=8, thread_name_prefix="p3s3-pread"
            )
            _PROCESS_EXECUTOR_CREATION_COUNT += 1
        return _PROCESS_EXECUTOR


def _shutdown_process_executor():
    global _PROCESS_EXECUTOR
    if _PROCESS_EXECUTOR is not None:
        _PROCESS_EXECUTOR.shutdown(wait=True, cancel_futures=False)
        _PROCESS_EXECUTOR = None


atexit.register(_shutdown_process_executor)


def _pread_exact_stats(fd: int, nbytes: int, offset: int):
    out = bytearray(nbytes)
    got = 0
    calls = 0
    sizes = Counter()
    short_reads = 0
    while got < nbytes:
        remaining = nbytes - got
        try:
            chunk = os.pread(fd, remaining, offset + got)
        except InterruptedError:
            continue
        if not chunk:
            raise IOError(f"short read at {offset}: {got}/{nbytes}")
        calls += 1
        sizes[len(chunk)] += 1
        if len(chunk) < remaining:
            short_reads += 1
        out[got:got + len(chunk)] = chunk
        got += len(chunk)
    return bytes(out), calls, sizes, short_reads


class ParallelShardIndex(p1.ShardIndex):
    """Expert reader with an optional eight-worker parallel read pool."""

    LAST_INSTANCE = None

    def __init__(self, *, parallel_pread: bool = True, workers: int = 8,
                 profile_timing: bool = False):
        super().__init__()
        self.parallel_pread = bool(parallel_pread)
        self.workers = int(workers)
        if self.parallel_pread and self.workers != 8:
            raise ValueError("parallel reads require exactly 8 workers")
        self.profile_timing = bool(profile_timing)
        self.executor = _process_executor() if self.parallel_pread else None
        self.executor_creation_count = (
            _PROCESS_EXECUTOR_CREATION_COUNT if self.parallel_pread else 0
        )
        self.read_path_s_total = 0.0
        self.read_path_s_by_call = defaultdict(float)
        self.read_path_invocations = 0
        self.checkpoint_read_started = False
        type(self).LAST_INSTANCE = self

    def close(self):
        super().close()

    def _read_tasks(self, tasks):
        if self.executor is None:
            return [_pread_exact_stats(fd, nbytes, offset)
                    for fd, nbytes, offset in tasks]
        futures = [
            self.executor.submit(_pread_exact_stats, fd, nbytes, offset)
            for fd, nbytes, offset in tasks
        ]
        return [future.result() for future in futures]

    def pread_experts(self, key: str, expert_ids: list[int], telemetry=None):
        shard_name = self.weight_map[key]
        _, fd, header, data_start = self.shards[shard_name]
        meta = header[key]
        shape = tuple(meta["shape"])
        start, end = meta["data_offsets"]
        per = (end - start) // shape[0]
        tasks = [
            (fd, per, data_start + start + int(expert) * per)
            for expert in expert_ids
        ]

        self.checkpoint_read_started = True
        t0 = time.perf_counter()
        results = self._read_tasks(tasks)
        read_wall = time.perf_counter() - t0
        self.read_path_s_total += read_wall
        self.read_path_invocations += 1
        call_key = getattr(telemetry, "current_call_key", None)
        if call_key is not None:
            self.read_path_s_by_call[call_key] += read_wall

        chunks = []
        for blob, calls, sizes, short_reads in results:
            chunks.append(blob)
            if telemetry is not None:
                if hasattr(telemetry, "pread_calls"):
                    telemetry.pread_calls += calls
                if hasattr(telemetry, "pread_sizes"):
                    telemetry.pread_sizes.update(sizes)
                telemetry.short_reads += short_reads

        if telemetry is not None and self.profile_timing:
            telemetry.bucket("pread_wait", read_wall)

        flat = b"".join(chunks)
        arr = np.frombuffer(
            flat, dtype=NP_DTYPES[meta["dtype"]]
        ).reshape((len(expert_ids),) + shape[1:])
        out = mx.array(arr)
        if meta["dtype"] == "BF16":
            out = out.view(mx.bfloat16)
        return out, len(flat), shard_name

    def read_path_snapshot(self, steady_first: int = 16,
                           steady_last: int = 95) -> dict:
        steady = 0.0
        for index in range(steady_first, steady_last + 1):
            steady += self.read_path_s_by_call.get(f"decode:{index}", 0.0)
        return {
            "parallel_pread_enabled": self.parallel_pread,
            "workers": 8 if self.parallel_pread else 0,
            "executor_creation_count": self.executor_creation_count,
            "read_path_timer_definition": (
                "wall time from immediately before exact per-expert read "
                "submission/serial loop until all expert byte buffers return; "
                "excludes b''.join, numpy reshape, mx.array and model compute"
            ),
            "read_path_s_total": self.read_path_s_total,
            "read_path_s_steady": steady,
            "read_path_invocations": self.read_path_invocations,
            "checkpoint_read_started": self.checkpoint_read_started,
        }
