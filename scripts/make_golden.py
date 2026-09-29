#!/usr/bin/env python3
"""Produce reference outputs from the UNMODIFIED model with stock mlx-lm (no AiiStream code).

The stock mlx-lm model is loaded lazily from model/. The full 4-bit model does not fit in 16 GB,
so each layer's full expert tensors (all 256 experts) are loaded right before that layer runs and
released right after it. The math is stock mlx-lm's own; only memory residency changes. This is
very slow (every token reads the whole model) and is meant to be run once.

    python scripts/make_golden.py [--out evidence/golden_stock_mlx_lm.json] [--cases a,b]
"""
from __future__ import annotations

import argparse, hashlib, json, os, platform, sys, tempfile, time
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm import load, stream_generate
from mlx_lm.models.switch_layers import SwitchGLU
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cases import CASES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "evidence" / "golden_stock_mlx_lm.json"))
ap.add_argument("--cases", default="")
args = ap.parse_args()
wanted = [c for c in CASES if not args.cases or c[0] in args.cases.split(",")]

with tempfile.TemporaryDirectory() as d:
    for sub in ("checkpoint", "hf-metadata"):
        for f in (ROOT / "model" / sub).iterdir():
            if not f.name.startswith("."):
                os.symlink(f.resolve(), Path(d) / f.name)
    weight_map = json.loads((Path(d) / "model.safetensors.index.json").read_text())["weight_map"]
    model, tokenizer = load(d, lazy=True)

    layer_of = {id(m): n for n, m in model.named_modules() if isinstance(m, SwitchGLU)}
    expected = {k.split(".mlp.switch_mlp.")[0] for k in weight_map if ".mlp.switch_mlp." in k}
    if len(layer_of) != len(expected) or not all(n.endswith(".mlp.switch_mlp") for n in layer_of.values()):
        sys.exit(f"could not map switch layers ({len(layer_of)} vs {len(expected)})")

    def release(module, prefix):
        shards = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for kind in ("weight", "scales", "biases"):
                key = f"{prefix}.{proj}.{kind}"
                shard = weight_map[key]
                if shard not in shards:
                    shards[shard] = mx.load(str(Path(d) / shard))
                setattr(getattr(module, proj), kind, shards[shard][key])

    stock_call = SwitchGLU.__call__

    def paged_call(self, x, indices):
        y = stock_call(self, x, indices)
        prefix = layer_of.get(id(self))
        if prefix is not None:
            mx.eval(y)
            release(self, prefix)
        return y

    SwitchGLU.__call__ = paged_call

    rows = {}
    for label, messages, temp, seed, tools, n in wanted:
        kwargs = {"tokenize": True, "add_generation_prompt": True}
        if tools is not None:
            kwargs["tools"] = tools
        prompt_ids = tokenizer.apply_chat_template(messages, **kwargs)
        mx.random.seed(seed)
        t0 = time.time()
        responses = list(stream_generate(model, tokenizer, prompt_ids, max_tokens=n,
                                         sampler=make_sampler(temp=temp)))
        final = responses[-1]
        ids = [int(r.token) for r in responses if r.finish_reason is None]
        if final.finish_reason == "length":
            ids.append(int(final.token))
        text = "".join(r.text for r in responses)
        rows[label] = {"token_ids": ids, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "finish_reason": final.finish_reason, "seconds": round(time.time() - t0, 1)}
        print("GOLDEN", label, len(ids), final.finish_reason, rows[label]["seconds"], "s", flush=True)

pins = json.loads((ROOT / "scripts" / "model_pins.json").read_text())
out = {"reference": "stock mlx-lm stream_generate on the unmodified checkpoint (per-layer expert paging only)",
       "model_revision": pins.get("revision"), "mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
       "machine": platform.machine(), "cases": rows}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
print("saved", args.out)
