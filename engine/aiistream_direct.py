#!/usr/bin/env python3
"""AiiStream direct expert reader: each requested expert is read from disk into its final buffer."""
from __future__ import annotations

from collections import Counter
import os
import time

import mlx.core as mx
import numpy as np

import aiistream_parallel as p3rt


def preadv_exact_into(fd: int, target, offset: int):
    """Exact positional read into an existing writable byte buffer."""
    view = target if isinstance(target, memoryview) else memoryview(target)
    view = view.cast("B") if view.format != "B" else view
    nbytes = len(view)
    got = 0
    calls = 0
    sizes = Counter()
    short_reads = 0
    while got < nbytes:
        remaining = nbytes - got
        try:
            n = os.preadv(fd, [view[got:]], int(offset) + got)
        except InterruptedError:
            continue
        if n == 0:
            raise IOError(f"short read at {offset}: {got}/{nbytes}")
        n = int(n)
        calls += 1
        sizes[n] += 1
        if n < remaining:
            short_reads += 1
        got += n
    return calls, sizes, short_reads


def _mx_from_raw(raw: np.ndarray, meta: dict, count: int):
    shape = tuple(meta["shape"])
    arr = np.frombuffer(
        raw, dtype=p3rt.NP_DTYPES[meta["dtype"]]
    ).reshape((int(count),) + shape[1:])
    out = mx.array(arr)
    if meta["dtype"] == "BF16":
        out = out.view(mx.bfloat16)
    return out


class DirectShardIndex(p3rt.ParallelShardIndex):
    """Demand-only expert reader with eight read workers."""

    LAST_INSTANCE = None

    def __init__(self, *, profile_timing: bool = False):
        super().__init__(
            parallel_pread=True, workers=8, profile_timing=profile_timing
        )
        self.short_reads_total = 0
        self.fetch_exceptions = 0
        type(self).LAST_INSTANCE = self

    def pread_experts(self, key: str, expert_ids: list[int], telemetry=None):
        shard_name = self.weight_map[key]
        _, fd, header, data_start = self.shards[shard_name]
        meta = header[key]
        shape = tuple(meta["shape"])
        start, end = meta["data_offsets"]
        per = (end - start) // shape[0]
        raw = np.empty(len(expert_ids) * per, dtype=np.uint8)
        offsets = [
            data_start + start + int(expert) * per
            for expert in expert_ids
        ]

        self.checkpoint_read_started = True
        mv = memoryview(raw)
        futures = []
        for i, offset in enumerate(offsets):
            futures.append(
                self.executor.submit(
                    preadv_exact_into,
                    fd,
                    mv[i * per:(i + 1) * per],
                    offset,
                )
            )

        t0 = time.perf_counter()
        try:
            results = [future.result() for future in futures]
        except Exception:
            self.fetch_exceptions += 1
            raise
        read_wall = time.perf_counter() - t0

        for calls, sizes, short_reads in results:
            self.short_reads_total += int(short_reads)
            if telemetry is not None:
                if hasattr(telemetry, "pread_calls"):
                    telemetry.pread_calls += int(calls)
                if hasattr(telemetry, "pread_sizes"):
                    telemetry.pread_sizes.update(sizes)
                telemetry.short_reads += int(short_reads)

        self.read_path_s_total += read_wall
        self.read_path_invocations += 1
        call_key = getattr(telemetry, "current_call_key", None)
        if call_key is not None:
            self.read_path_s_by_call[str(call_key)] += read_wall
        if self.profile_timing and telemetry is not None and hasattr(telemetry, "bucket"):
            telemetry.bucket("pread_wait", read_wall)

        out = _mx_from_raw(raw, meta, len(expert_ids))
        return out, len(raw), shard_name
