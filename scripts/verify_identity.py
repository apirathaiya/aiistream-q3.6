#!/usr/bin/env python3
"""Verify AiiStream's central claim on your machine: output identical to the unmodified model.

Runs 8 cases (greedy and seeded temperature 0.7, plain and tool-calling, 12-400 tokens) through the
real HTTP server in each read mode (prefetch, direct, parallel), then requires ALL of:
  1. every mode matches the stock mlx-lm reference of the unmodified model
     (evidence/golden_stock_mlx_lm.json, produced by scripts/make_golden.py);
  2. the server matches mlx-lm's own generation loop on the same engine model;
  3. the declared coverage really happened: seeded sampling differs from greedy, and the
     tool cases produced a parsed tool call.
Takes ~15 minutes; loads the model once per mode. Exit code 0 only if every check passes."""
import json, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cases import CASES, EXPECT_TOOL_CALL, SEEDED_VS_GREEDY  # noqa: E402

MODES = ("prefetch", "direct", "parallel")


def check(out: dict, golden: dict) -> list[str]:
    """Return a list of failures (empty means PASS). out[mode][case] -> row."""
    fails = []
    names = [c[0] for c in CASES]
    for mode in MODES:
        if mode not in out:
            fails.append(f"{mode}: no results")
            continue
        for case in names:
            r = out[mode].get(case)
            if r is None:
                fails.append(f"{mode}/{case}: missing")
                continue
            if not r["service_eq_mlx_lm"]:
                fails.append(f"{mode}/{case}: server differs from mlx-lm loop on the same model")
            g = golden.get(case)
            if g is None:
                fails.append(f"{case}: no stock reference")
            elif r["token_ids"] != g["token_ids"] or r["text_sha256"] != g["text_sha256"]:
                fails.append(f"{mode}/{case}: differs from the unmodified stock model")
    base = out.get("prefetch", {})
    for seeded, greedy in SEEDED_VS_GREEDY:
        if seeded in base and greedy in base and base[seeded]["token_ids"] == base[greedy]["token_ids"]:
            fails.append(f"coverage: {seeded} equals {greedy} (sampling path not exercised)")
    for case in EXPECT_TOOL_CALL:
        for mode in MODES:
            r = out.get(mode, {}).get(case)
            if r is not None and not r["http_tool_calls"]:
                fails.append(f"coverage: {mode}/{case} produced no tool call")
    return fails


def main() -> int:
    golden_path = ROOT / "evidence" / "golden_stock_mlx_lm.json"
    golden = json.loads(golden_path.read_text())["cases"]
    out = {}
    with tempfile.TemporaryDirectory() as d:
        for mode in MODES:
            path = Path(d) / f"{mode}.json"
            print(f"== {mode}", flush=True)
            subprocess.call([sys.executable, "-B", str(ROOT / "scripts" / "_identity_mode.py"),
                             "--mode", mode, "--port", "8095", "--output", str(path)])
            if path.exists():
                out[mode] = {r["case"]: r for r in json.loads(path.read_text())["results"]}
    for case in [c[0] for c in CASES]:
        r = out.get("prefetch", {}).get(case)
        if r:
            same = all(out.get(m, {}).get(case, {}).get("token_ids") == r["token_ids"] for m in MODES)
            print(f"{case:22s} tokens={r['completion_tokens']:3d} tool_call={r['http_tool_calls']!s:5s} "
                  f"identical_across_modes={same} matches_stock={r['token_ids'] == golden.get(case, {}).get('token_ids')}")
    fails = check(out, golden)
    for f in fails:
        print("FAIL:", f)
    print("RESULT:", "PASS - identical to the unmodified model in all modes" if not fails else f"FAIL ({len(fails)})")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
