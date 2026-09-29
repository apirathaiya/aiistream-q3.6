#!/usr/bin/env python3
"""Paired A/B benchmark of the engine: direct (plain on-demand streaming) vs prefetch (default).

For every prompt and round, both modes generate the same greedy completion in alternating order,
calling the engine in-process (no HTTP). For each run it reports:
  - ttft_s: time to first token (prompt processing);
  - decode_tok_s: decode throughput = (tokens - 1) / (total time - ttft), prefill excluded;
  - total_s: total generation time.
It FAILS if the two modes ever produce different token IDs. The summary also records swap in use.

    python scripts/bench.py                 # 3 prompts x (1 warm-up + 2 scored) rounds, 256 tokens
    python scripts/bench.py --tokens 512 --rounds 3
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

import mlx.core as mx
from aiistream_server.runtime_engine import RuntimeEngine
from aiistream_server.safety import SafetyState

PROMPTS = [
    "Explain in detail how a hash table handles collisions, with examples.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Write a Python function that parses ISO-8601 dates, then explain each step.",
]


def run(mode: str, prompt: str, tokens: int) -> dict:
    engine = RuntimeEngine(repo_root=ROOT, safety=SafetyState(), expert_read_path=mode)
    try:
        prepared = engine.prepare([{"role": "user", "content": prompt}])
        mx.reset_peak_memory()
        r = engine.generate_prepared(prepared, temperature=0.0, seed=0, max_completion_tokens=tokens)
        peak_gb = mx.get_peak_memory() / 1e9
        decode_s = r.elapsed_s - (r.ttft_s or 0.0)
        return {"mode": mode, "tokens": r.completion_tokens, "token_ids": r.token_ids,
                "decode_tok_s": (r.completion_tokens - 1) / decode_s if decode_s > 0 else None,
                "ttft_s": r.ttft_s, "elapsed_s": r.elapsed_s, "mlx_peak_gb": peak_gb}
    finally:
        engine.close()
        del engine
        gc.collect()
        mx.clear_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=2, help="scored rounds (one extra warm-up round is always run)")
    ap.add_argument("--out", default=str(ROOT / "bench-results" / f"bench-{int(time.time())}.json"))
    a = ap.parse_args()
    rows, gains = [], []
    for rnd in range(a.rounds + 1):
        for i, prompt in enumerate(PROMPTS):
            order = ("direct", "prefetch") if (i + rnd) % 2 == 0 else ("prefetch", "direct")
            got = {m: run(m, prompt, a.tokens) for m in order}
            z, p = got["direct"], got["prefetch"]
            if z["token_ids"] != p["token_ids"]:
                print(f"IDENTITY FAILURE on prompt {i}, round {rnd}", file=sys.stderr)
                return 1
            gain = p["decode_tok_s"] / z["decode_tok_s"] - 1
            label = "warm-up" if rnd == 0 else f"round {rnd}"
            print(f"{label:8s} prompt {i}: decode direct {z['decode_tok_s']:.2f} | prefetch {p['decode_tok_s']:.2f} tok/s "
                  f"({gain:+.1%}); ttft {z['ttft_s']:.2f}/{p['ttft_s']:.2f} s; total {z['elapsed_s']:.1f}/{p['elapsed_s']:.1f} s "
                  f"| identical tokens: yes ({p['tokens']})", flush=True)
            for m in got.values():
                m.pop("token_ids")
            rows.append({"round": rnd, "prompt": i, "gain": gain, **{k: v for k, v in got.items()}})
            if rnd > 0:
                gains.append(gain)
    scored = [r for r in rows if r["round"] > 0]
    summary = {
        "paired_median_gain": statistics.median(gains),
        "positive_pairs": sum(g > 0 for g in gains), "pairs": len(gains),
        "direct_median_tok_s": statistics.median(r["direct"]["decode_tok_s"] for r in scored),
        "prefetch_median_tok_s": statistics.median(r["prefetch"]["decode_tok_s"] for r in scored),
        "direct_median_ttft_s": statistics.median(r["direct"]["ttft_s"] for r in scored),
        "prefetch_median_ttft_s": statistics.median(r["prefetch"]["ttft_s"] for r in scored),
        "direct_median_total_s": statistics.median(r["direct"]["elapsed_s"] for r in scored),
        "prefetch_median_total_s": statistics.median(r["prefetch"]["elapsed_s"] for r in scored),
        "prefetch_max_mlx_peak_gb": max(r["prefetch"]["mlx_peak_gb"] for r in scored),
        "metric": "engine decode throughput, in-process, prefill excluded; not end-to-end HTTP latency",
        "swap_used_at_end": subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip(),
        "tokens": a.tokens, "all_outputs_identical": True,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print("\nSUMMARY", json.dumps(summary, indent=1))
    print("saved", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
