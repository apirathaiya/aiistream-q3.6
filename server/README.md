# AiiStream Q3.6 server

Local OpenAI-compatible HTTP server for Qwen3.6-35B-A3B running on the AiiStream expert-streaming runtime.

**Transport rule:** the server never changes generation. It does not inject system prompts, rewrite roles, trim
messages, retry, resample or post-process model text (including `<think>` blocks). Its output is tested
token-for-token against mlx-lm's own generation loop (`tests/test_identity_real.py`, `../scripts/verify_identity.py`).

## Security

**Loopback only.** The server binds to `127.0.0.1` and has **no authentication and no TLS**. Do not expose it through
a proxy or tunnel.

## Thermal note

The server has no thermal governor. Long sustained generation on a fanless laptop may throttle.
`/health` reports the macOS thermal state.

## Start

```bash
../scripts/serve.sh            # builds the tiny macOS thermal-state helper on first run (needs Xcode CLT: swiftc)
```

Startup order: configuration is validated and every runtime file on the selected read path is SHA-256-checked
**before** the model loads; the port opens only after the model is loaded. A hash mismatch refuses startup.

## Read paths (`expert_read_path` in `config.json`)

| value | what it runs | pinned runtime files |
|---|---|---|
| `prefetch` (default) | AiiStream: reads likely experts ahead of time (`aiistream_prefetch.py`) | model, parallel, direct, prefetch |
| `direct` | reads each needed expert on demand, no prefetch (baseline for `bench.py`) | model, parallel, direct |
| `parallel` | simple parallel on-demand reads | model, parallel |

All three produce identical tokens; switching is a one-line config change plus restart.

## Configuration

`config.json` defaults:

```json
{
  "port": 8081,
  "queue_limit": 4,
  "max_context": 32768,
  "max_context_hard_ceiling": 65536,
  "expert_read_path": "prefetch",
  "telemetry": {"poll_seconds": 5},
  "safety": {
    "memory_pressure_stop": true,
    "throughput_collapse_stop": true,
    "collapse_fraction": 0.5,
    "collapse_window_seconds": 60
  }
}
```

Port 8080 is rejected by design, so it does not clash with another local model server on its default port.
## API

Endpoints:

- `POST /v1/chat/completions` — JSON or Server-Sent Events with `"stream": true`.
- `GET /v1/models` — the single local model.
- `GET /health` — queue, safety state and latest telemetry.

Supported chat request fields are `model`, `messages`, `stream`, `temperature`,
`max_completion_tokens`, `seed`, `tools`, plus `n: 1` as a compatibility constraint.
Unsupported fields are rejected explicitly; `tool_choice`, `parallel_tool_calls`,
`stream_options`, `n > 1`, batching, speculative/draft-model controls and
KV-quantisation controls are out of scope rather than silently ignored.

The service uses the checkpoint chat template directly with
`add_generation_prompt=True`. When `tools` is present it passes that array to the
checkpoint template; it does not add its own tool prompt or generation instruction.

Example:

```bash
curl --silent --show-error http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.6-35b-a3b-local",
    "messages":[{"role":"user","content":"Say hello."}],
    "temperature":0,
    "seed":7,
    "max_completion_tokens":64
  }'
```
For SSE add `"stream": true` and use `curl --no-buffer`.

## Tool calling

Tool definitions use the standard OpenAI JSON-Schema function shape, but the checkpoint's
generated call syntax is its own template format:

```text
<tool_call>
<function=NAME>
<parameter=PARAMETER>
value
</parameter>
</function>
</tool_call>
```

Multiple calls use one newline between complete `<tool_call>` blocks. Reasoning or other
model text may appear before the first call and is preserved exactly; the service does not
strip `<think>`.

Parameter values are emitted to API clients as **strings exactly as generated**. The
service does not coerce them against the supplied JSON Schema. Echoed OpenAI
`function.arguments` arrive as a JSON string; before rendering the next prompt the
service decodes that string back to a JSON object because the checkpoint template iterates
the argument mapping. Invalid JSON or a non-object is rejected rather than replaced.

If generated tool syntax is malformed, the service does not repair, retry, infer missing
parameters, or partially accept calls. The complete raw model text is returned as
`content`, there are no `tool_calls`, and `finish_reason` is `stop`.

For streaming responses, text before `<tool_call>` is emitted as normal content.
Tool-call framing itself is buffered and converted to indexed `tool_calls` deltas, so
clients do not display raw XML.

## Context policy

The measured input-context default is **32,768 tokens**. Rendered prompts at or below it
are accepted normally. The measured pass ceiling is **65,536 tokens**.

A rendered prompt above 32,768 and at or below 65,536 requires explicit opt-in:

```text
X-Qwen36-Allow-Long-Context: true
```

The response then includes `X-Qwen36-Long-Context-Warning` and HTTP `Warning: 299`.
The warning states the measured cost: at 65,536 input tokens we observed roughly
**9.6 minutes TTFT** and about **27% intra-request decode decay**. Prompts above 65,536
are refused. This limit concerns rendered input prompt tokens; the service separately
rejects prompt + requested completion tokens that exceed the checkpoint's own model maximum.

## Concurrency and admission

There is one model generation at a time. Up to four additional requests may wait, for a
total admission capacity of five. A sixth request receives HTTP **503** plus
`Retry-After: 5`. Queue admission is checked before prompt rendering/tokenization.

Two evidence-based safety conditions can also return 503 for **new** requests:

1. macOS `kern.memorystatus_vm_pressure_level == 4` (critical), and
2. decode throughput below 50% of that request's early-decode median for 60 continuous seconds.

**Neither condition can terminate the generation already in flight.** They gate admission only.

**Client disconnects.** If a client closes the connection (streaming or not), the server notices within one token
and stops that generation. The slot is freed for the next request and request-scoped buffers are reset. Requests
that are not cancelled are unaffected: their output is identical.
## Observability — not a governor

Every 5 seconds the service samples and exposes on `/health`:

- battery and virtual-battery temperature, charge and AC state;
- macOS `thermalState`;
- swap used;
- MLX active, peak and allocator-cache bytes;
- macOS memory-pressure level;
- queue and safety state.

Temperature, swap and MLX counters are observations only. There are no thermal thresholds,
worker stepping, hysteresis or governor state. Worker count is fixed at **8**.

Telemetry transition/events are written to `logs/telemetry_events.jsonl`. Swap crossings
are logged for observability but are not admission stops.

## Tests

Fast suite, with the real-model identity test skipped:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Real-model identity gate:

```bash
QWEN36_REAL_TEST=1 python3 -m unittest tests.test_identity_real.RealIdentityTest.test_service_vs_mlx_lm_reference_identity_and_eos_framing -v
```

That test sends real HTTP service requests, then drives the same model through
mlx-lm's own `stream_generate` loop with the same prompt, temperature, seed and token cap.
It covers both the ordinary path and a prompt rendered with `tools`; raw token-ID arrays
and SHA-256 of decoded model text must match exactly.
