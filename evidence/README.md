# Evidence

Produced with the code in this repository on Apple M5 / 16 GB / macOS 27.0, `mlx==0.32.2`, `mlx-lm==0.31.3`.

| file | what it shows |
|---|---|
| `golden_stock_mlx_lm.json` | `scripts/make_golden.py`: token IDs and text SHA-256 from **stock mlx-lm running the unmodified checkpoint**, for the 8 identity cases. This is the independent reference. |
| `verify_identity.txt` | `scripts/verify_identity.py`: every case in all 3 read modes equals the stock reference and mlx-lm's loop on the engine model; seeded sampling differs from greedy; the tool cases produce a parsed tool call |
| `bench.json` | `scripts/bench.py`: paired decode-throughput A/B, prefetch vs direct, with TTFT, total time, MLX peak memory and swap in use |

Speed depends on free RAM, because macOS caches model files. Re-run `scripts/bench.py` to measure your own machine.
