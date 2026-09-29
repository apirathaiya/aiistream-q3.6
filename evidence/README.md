# Evidence

Apple M5 / 16 GB / macOS 27.0, `mlx==0.32.2`, `mlx-lm==0.31.3`. `golden_stock_mlx_lm.json`, `verify_identity.txt` and
`bench.json` were produced with the code in this repository; the two dated files were produced with the author's
lab and production tooling, which is not included.

| file | what it shows |
|---|---|
| `golden_stock_mlx_lm.json` | `scripts/make_golden.py`: token IDs and text SHA-256 from **stock mlx-lm running the unmodified checkpoint**, for the 8 identity cases. This is the independent reference. |
| `verify_identity.txt` | `scripts/verify_identity.py`: every case in all 3 read modes equals the stock reference and mlx-lm's loop on the engine model; seeded sampling differs from greedy; the tool cases produce a parsed tool call |
| `bench.json` | `scripts/bench.py`: paired decode-throughput A/B, prefetch vs direct, with TTFT, total time, MLX peak memory and swap in use |
| `harness_2026-09-28.json` | the lab measurement behind the 15.0 tok/s figure (15.0 vs 8.74 tok/s; held-out 15.1 vs 8.6), with per-pair gains and the SHA-256 of the source records. **Not produced by code in this repository** (the lab harness is not included); measured before the seeded-sampling fix; the limits are listed inside the file |
| `production_checks_2026-09-29.json` | checks after the seeded-sampling fix and disconnect cancellation: lab A/B of the fixed engine vs the 15.0 build (+1.45% median, 5/8, no measurable change); two 30-request memory soaks with a criterion fixed in advance (both pass); disconnect cancellation in 0.10 s with an identical next request; identity 8 cases × 3 modes equal to the stock reference. **Not produced by code in this repository** |

Speed depends on free RAM, because macOS caches model files. Re-run `scripts/bench.py` to measure your own machine.
