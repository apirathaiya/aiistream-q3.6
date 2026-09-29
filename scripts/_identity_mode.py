#!/usr/bin/env python3
"""AiiStream identity matrix, one mode per process (used by verify_identity.py).

For each case it records the service output and mlx-lm's own generation loop run on the same
engine model (this proves the server loop adds nothing). verify_identity.py then compares every
mode against the stock-model reference in evidence/golden_stock_mlx_lm.json.
"""
import argparse, hashlib, json, sys, threading, time, urllib.request
from pathlib import Path

LA = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--service", default=str(Path(__file__).resolve().parents[1] / "server"))
parser.add_argument("--mode", required=True, choices=["prefetch", "direct", "parallel"])
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.service)))

from aiistream_server.app import ServiceApp
from aiistream_server.config import ServiceConfig, SafetyConfig, TelemetryConfig
from aiistream_server.runtime_engine import RuntimeEngine, MODEL_ID
from aiistream_server.safety import SafetyState
from aiistream_server.server import QwenHTTPServer
from tests.test_identity_real import mlx_reference, FakeMonitor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cases import CASES  # noqa: E402

safety = SafetyState()
engine = RuntimeEngine(repo_root=LA, safety=safety, expert_read_path=args.mode, capture_last_result=True)
print("ENGINE", args.mode, engine.runtime_hashes, flush=True)
cfg = ServiceConfig(port=args.port, expert_read_path=args.mode, telemetry=TelemetryConfig(5), safety=SafetyConfig())
app = ServiceApp(config=cfg, engine=engine, telemetry_monitor=FakeMonitor(), safety=safety)
server = QwenHTTPServer(("127.0.0.1", args.port), app)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
rows = []
try:
    for label, messages, temp, seed, tools, n in CASES:
        body = {"model": MODEL_ID, "messages": messages, "temperature": temp, "seed": seed,
                "max_completion_tokens": n, "stream": False}
        if tools is not None:
            body["tools"] = tools
        req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=900) as resp:
            http = json.loads(resp.read())
        wall = time.perf_counter() - t0
        res = engine.last_result
        ref = mlx_reference(engine, messages, temperature=temp, seed=seed, max_tokens=n, tools=tools)
        row = {"mode": args.mode, "case": label, "temperature": temp, "seed": seed, "max_tokens": n,
               "token_ids": res.token_ids, "token_ids_sha256": hashlib.sha256(json.dumps(res.token_ids).encode()).hexdigest(),
               "text_sha256": res.text_sha256, "finish_reason": res.finish_reason,
               "completion_tokens": res.completion_tokens, "has_tool_call_markup": "<tool_call>" in res.text,
               "http_finish_reason": http["choices"][0]["finish_reason"],
               "http_tool_calls": bool(http["choices"][0]["message"].get("tool_calls")),
               "ref_token_ids_sha256": hashlib.sha256(json.dumps(ref["token_ids"]).encode()).hexdigest(),
               "ref_text_sha256": ref["text_sha256"],
               "service_eq_mlx_lm": res.token_ids == ref["token_ids"] and res.text_sha256 == ref["text_sha256"],
               "http_wall_s": wall}
        rows.append(row)
        print("CASE", args.mode, label, "eq_mlx_lm=%s" % row["service_eq_mlx_lm"], "n=%d" % res.completion_tokens,
              "finish=%s" % res.finish_reason, "tool=%s" % row["http_tool_calls"], "wall=%.1f" % wall, flush=True)
finally:
    server.shutdown(); server.server_close(); thread.join(timeout=30)
    live = getattr(engine.shard, "live_slots", None)
    engine.close()
with open(args.output, "x") as f:
    json.dump({"mode": args.mode, "runtime_hashes": engine.runtime_hashes, "results": rows}, f, indent=1)
bad = [r["case"] for r in rows if not r["service_eq_mlx_lm"]]
print("DONE", args.mode, "mismatch_vs_mlx_lm", bad, flush=True)
sys.exit(1 if bad else 0)
