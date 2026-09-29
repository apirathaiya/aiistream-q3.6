# AiiStream Q3.6

**Run Qwen3.6-35B-A3B on a 16 GB Mac with byte-identical output. No retraining, no reduced routing, no quality
trade-off.**

AiiStream Q3.6 runs a 35-billion-parameter Mixture-of-Experts model on a laptop that cannot hold it in memory:
- the ~18 GB of expert weights stay on the Mac's internal SSD;
- for each token, the engine reads only the experts the model actually uses;
- it anticipates what the model will need next, so most reads finish before they are needed.

The generated tokens are **identical, token for token,** to the unmodified 4-bit model. You can check this on your own
machine with one command.

*AiiStream is apirathaiya's expert-streaming engine; each supported model ships as its own release, and **Q3.6** is the
Qwen3.6-35B-A3B release. By apirathaiya. Apache-2.0: free for commercial and noncommercial use, with credit.*

---

## At a glance

| | |
|---|---|
| Model | Qwen3.6-35B-A3B, `mlx-community` 4-bit (40 MoE layers, 256 experts, top-8), pinned revision |
| Tested on | Apple M5, **16 GB** unified memory, internal SSD, macOS 27.0 (less RAM is expected to work; see [Running with less than 16 GB](#running-with-less-than-16-gb-of-ram)) |
| **Decode throughput** | **15.0 tok/s** (lab measurement, 512-token generations; **15.1** on held-out prompts) |
| vs. plain on-demand streaming | **+72.6%** faster decode on the same machine (12 of 12 paired runs faster) |
| Time to first token | ~2.3 s for a short prompt (`scripts/bench.py`) |
| Memory | MLX peak **1.7 GB** during generation (`scripts/bench.py`) |
| Memory over time | **stable**: 30 consecutive 300-token requests, server process ~2.0–2.2 GB, total growth ≤ 92 MB, criterion fixed before the first request |
| Client disconnect | generation stops **~0.1 s** after the client closes the connection; the next request is unaffected |
| Output | **byte-identical** to stock mlx-lm running the unmodified checkpoint (same token IDs and text SHA-256), including seeded sampling and tool calls |
| Routing | unchanged top-8: no expert skipping, no top-k reduction, no substitute experts |
| Training | none |
| Serving | OpenAI-compatible HTTP server on `127.0.0.1`, with streaming and tool calling |

With the shipped `scripts/bench.py`, which uses the server's generation loop, the same machine measures about
**11.8 tok/s** decode (+52% vs plain streaming). See [Performance notes](#performance-notes).

## Comparison with other SSD-streaming projects

Each row is the decode speed each project reports for a Qwen-family MoE model of this size class, on its own stated
hardware.
- **The machines differ, so this is not a controlled benchmark.** It is the comparison a reader can check today.
- "Peak memory" is each project's reported peak MLX memory.

| Project | Model as run | Routing | Output vs released model | Machine RAM | Peak memory | Decode tok/s |
|---|---|---|---|---:|---:|---:|
| **AiiStream Q3.6** | Qwen3.6-35B-A3B 4-bit | **top-8, exact** | **byte-identical** | **16 GB** | **1.7 GB** | **15.0**¹ |
| [Edge0](https://github.com/Edge0-AI/Edge0) | Qwen3.6-35B-A3B int4 + trained LoRA | top-4, trained prerouter | modified (−3.9 pts avg vs fp16, their eval) | 24 GB | 2.9 GiB | 14.9–17.7 |
| [mira-core](https://github.com/mabaeyens/mira-core/blob/main/docs/moe-offload-case-study.md) | Qwen3.6-35B-A3B 4-bit, 30% of experts resident | top-8, exact | lossless | 32 GB | 7.3 GB | 10.8 |
| [mira-core](https://github.com/mabaeyens/mira-core/blob/main/docs/moe-offload-case-study.md) | Qwen3.6-35B-A3B 8-bit, 30% resident | top-8, exact | lossless | 32 GB | 12.7 GB | 8.1 |
| [expert-sniper](https://github.com/walter-grace/expert-sniper) | Qwen3-30B-A3B 4-bit (128 experts), routing bias 0 | top-8, exact | lossless | 16 GB | — | 1.15 |

¹ Lab measurement. `scripts/bench.py` measures about 11.8 tok/s on the same machine.

**What the table shows:** AiiStream Q3.6 is the only entry that is byte-identical *and* runs on 16 GB. Sources were checked
2026-09-27/28: the Edge0 README Benchmark and Quality tables; mira-core `docs/moe-offload-case-study.md` §10–§11; the
expert-sniper README "Performance (v0.2)".

## How it works (overview)

Each generated token passes through 40 MoE layers, and each layer uses 8 of its 256 experts. AiiStream combines:

1. **Selective streaming.** Attention, routers and shared weights stay in memory. Routed experts are read from the SSD
   only when the model selects them.
2. **Look-ahead reading.** The engine estimates which experts upcoming layers will need, using signals the model already
   computes, and starts reading them early.
   - The model's own router still makes every decision.
   - A wrong guess costs time, never accuracy: a missing expert is simply read on demand.
3. **Efficient I/O.** Reads run in parallel, and data lands directly in memory the GPU can use, with no extra copies.
4. **Exactness checks.** Every change was accepted only if the output stayed byte-identical to the reference model, and
   the included scripts let you re-check that yourself.

## Quick start

```bash
git clone <this repo> aiistream && cd aiistream
python3 -m venv .venv && source .venv/bin/activate
pip install "mlx==0.32.2" "mlx-lm==0.31.3" huggingface_hub     # the versions validated for byte identity

scripts/download_model.sh      # ~20 GB; pinned revision, every file SHA-256-verified
scripts/serve.sh               # OpenAI-compatible server on http://127.0.0.1:8081

curl http://127.0.0.1:8081/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b-local","messages":[{"role":"user","content":"Hello"}]}'
```

Any OpenAI-compatible client can use `http://127.0.0.1:8081/v1` as its base URL. See [`server/README.md`](server/README.md)
for the API, tool calling, context limits and configuration.

## Check the claims on your machine

```bash
python scripts/verify_identity.py    # 8 cases x 3 read modes vs the stock-model reference (~20 min)
python scripts/bench.py              # paired A/B decode throughput: plain on-demand streaming vs AiiStream (~20 min)
```

## Performance notes

- **Free RAM matters.** macOS keeps recently read model files in its file cache, and AiiStream benefits from it.
  - With other memory-heavy apps open, or when the system is swapping, speed drops.
  - That cache memory is reclaimable, so it never blocks other apps.
- **The first request after a reboot is slower** while the file cache warms up.
- **What `bench.py` measures.** It calls the engine in-process, with no HTTP, and reports three numbers per run:
  - time to first token;
  - decode throughput, with prefill excluded;
  - total time.

  The headline figure is decode throughput. End-to-end latency through HTTP also includes prompt processing and
  client overhead.
- **The 15.0 tok/s figure (lab) and the 11.8 tok/s figure (`bench.py`).**
  - **15.0 tok/s** is the decode rate measured in the development lab harness: 8.74 tok/s for plain on-demand
    streaming against 15.0 for AiiStream, +72.6% paired median, 12 of 12 pairs faster, 4 prompts × 512 tokens,
    greedy, Apple M5 16 GB, macOS 27.0, 2026-09-28. On prompts not used during development: 15.1 vs 8.6 tok/s
    (+74.9%), with identical tokens in every pair. Summary and per-pair gains:
    [`evidence/harness_2026-09-28.json`](evidence/harness_2026-09-28.json).
  - That harness is **not included in this repository**. The 15.0 run used the build from before the seeded-sampling
    fix. A later paired A/B in the same harness measured the fixed engine against that build: **+1.45% median, 5 of 8
    pairs faster**, i.e. no measurable speed change
    ([`evidence/production_checks_2026-09-29.json`](evidence/production_checks_2026-09-29.json)).
  - **11.8 tok/s** is what `scripts/bench.py` measures on the same machine. It uses the server's generation loop and
    a different estimator. The cause of the gap to 15.0 is not yet identified.
  - The same build measured between 11.5 and 15.0 tok/s on different days, depending on free RAM.
- **Long reasoning.** The model may think before answering, and that thinking counts toward completion time.

## Running with less than 16 GB of RAM

**Not yet tested.** Every figure in this README comes from a 16 GB machine. But the measurements show why a Mac with
less RAM (for example 8 GB) should still run AiiStream Q3.6, only more slowly:

1. **The engine's own memory is small.** MLX peak memory during generation is about **1.7 GB**, and the server
   process is about **2 GB**. That is the attention, router and shared weights plus a fixed set of reusable read
   buffers. Expert weights are **never held in RAM**: they are read from the SSD for each token, into buffers that are
   reused rather than accumulated.
2. **The rest of the 16 GB was optional cache, not a requirement.** During the lab runs, macOS kept about 8.9 GB of
   the model file in its file cache, and that cache served about 91% of the expert bytes. The file cache is
   reclaimable memory: with less RAM it shrinks, more reads go to the SSD, and generation slows down. It does not stop
   working.
3. **The SSD had headroom.** In the same runs, the SSD delivered only about 9% of the expert bytes and used about 8%
   of its bandwidth. With a smaller cache, the SSD can take on more of the reads.

**What to expect, and what to do:**
- **Slower decode than on 16 GB.** How much slower has not been measured, because every lab figure was taken with a
  warm file cache. Run `scripts/bench.py` to measure your own machine.
- **Keep prompts short.** Long prompts need more memory while they are processed. On an earlier build of the engine,
  MLX peak memory rose to about 3.4 GB for an 8K-token prompt and 3.9 GB for a 32K-token prompt. On a smaller Mac,
  lower `max_context` in `server/config.json` (for example to `8192`).
- **Close memory-heavy apps.** Everything else that uses RAM competes with the file cache.
- **The server protects itself.** If macOS reports critical memory pressure, the server refuses new requests with
  HTTP 503 instead of starting one. A generation already in progress is never cut off.
- **Other Apple Silicon chips** (M1–M4) should work, because the engine uses only standard MLX operations, but they
  have not been tested. Byte identity is defined against stock mlx-lm; the shipped reference was produced on an M5.
  If `scripts/verify_identity.py` reports a difference on another chip, regenerate the reference on your machine with
  `scripts/make_golden.py` and run it again.
- You still need **~20 GB of free SSD space** for the model, whatever your RAM.

## When to use AiiStream Q3.6

- You want Qwen3.6-35B-A3B **exactly as released** on a 16 GB Mac (smaller Macs are expected to work, more slowly),
  or you want to keep RAM free on a bigger one.
- You need **reproducible** outputs: evaluations, regression tests, or seeded sampling that must match a reference.
- You want a local, private, OpenAI-compatible endpoint with tool calling.

## When not to use it

- You have 32 GB or more and only want speed. Keeping more of the model resident will be faster.
- You accept modified outputs for speed. Reduced-routing approaches decode faster at the same memory.
- You need batched, multi-user serving. AiiStream serves one request at a time.
- You are not on Apple Silicon. It is MLX-only.

## FAQ

**Is the output really identical, or just close?**
- **Identical:** the same token IDs and the same text SHA-256 as **stock mlx-lm running the unmodified checkpoint**,
  in every read mode.
- **The reference is independent of AiiStream.** `scripts/make_golden.py` runs mlx-lm's own model code with all 256
  experts of each layer, loaded layer by layer to fit in 16 GB. It is slow (~3 s/token) and was run once; its output is
  in `evidence/golden_stock_mlx_lm.json`.
- **Coverage:** greedy and seeded (temperature 0.7) sampling, plain and tool-calling prompts, 12–200 tokens.
  - The verifier fails unless seeded sampling actually differs from greedy, and unless the tool cases produce a
    parsed tool call.
- **Check it yourself:** `scripts/verify_identity.py`. The engine draws no random numbers of its own, so seeded
  sampling follows exactly the same random sequence as stock mlx-lm.

**Identical to what?**
- To `mlx-community/Qwen3.6-35B-A3B-4bit` (revision pinned in `scripts/model_pins.json`) run through mlx-lm's standard
  generation.
- AiiStream adds no loss on top of that checkpoint. It makes no claim about 4-bit versus BF16.

**Does it wear out the SSD?**
No. It only reads, and reads do not meaningfully wear flash.

**Can I switch the engine off?**
Yes. Set `expert_read_path` in `server/config.json` to `direct` (plain on-demand streaming) and restart. All modes
produce identical tokens.

## Requirements

- macOS on Apple Silicon. Tested with 16 GB of RAM; less RAM is expected to work more slowly (see
  [Running with less than 16 GB](#running-with-less-than-16-gb-of-ram)).
- ~20 GB of free SSD space.
- Python 3.12+.
- `mlx==0.32.2` and `mlx-lm==0.31.3`. These are pinned: other versions are not verified for byte identity.
- Xcode Command Line Tools, to build a tiny thermal-state helper on first start.

## Layout

```
engine/     the streaming engine (hash-pinned; the server refuses to start if a file changes)
server/     OpenAI-compatible HTTP server, config and tests
scripts/    download_model.sh, serve.sh, verify_identity.py, bench.py, make_golden.py
evidence/   identity and benchmark outputs produced with this code
model/      downloaded checkpoint and tokenizer (not in git)
```

## License

**Apache License 2.0** (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)).

- **Free to use, modify and redistribute, including commercially.**
- **Credit is required.** If you distribute AiiStream Q3.6 or a work based on it, keep the [`NOTICE`](NOTICE) file (Apache-2.0
  §4(d)) and credit "AiiStream Q3.6 by apirathaiya" in your documentation or credits.
- **Model weights are not included.** They are downloaded from Hugging Face under the Qwen model license.

## Citation

```bibtex
@software{aiistream_q36_2026,
  title  = {AiiStream Q3.6: Byte-Identical SSD Expert Streaming for Qwen3.6-35B-A3B on 16 GB Apple Silicon},
  author = {{apirathaiya}},
  year   = {2026},
  url    = {https://github.com/apirathaiya/aiistream-q3.6}
}
```
