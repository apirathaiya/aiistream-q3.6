#!/usr/bin/env python3
"""AiiStream model builder: non-expert weights stay resident; routed experts are streamed from disk."""
from __future__ import annotations

import gc, json, os, resource, sys, time
from collections import defaultdict
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import qwen3_5
from mlx_lm.models.switch_layers import SwiGLU, _gather_sort, _scatter_unsort
from mlx_lm.utils import load_tokenizer

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "model" / "checkpoint"
CONFIG_PATH = ROOT / "model" / "hf-metadata" / "config.json"
INDEX_PATH = ROOT / "model" / "hf-metadata" / "model.safetensors.index.json"
GROUP_SIZE, BITS, MODE = 64, 4, "affine"
NP_DTYPES = {"U8": np.uint8, "I8": np.int8, "U16": np.uint16, "U32": np.uint32,
             "F16": np.float16, "F32": np.float32, "BF16": np.uint16}


def read_header(path: Path):
    with path.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    return header, 8 + n


def pread_exact(fd: int, nbytes: int, offset: int, telemetry=None) -> bytes:
    out = bytearray(nbytes); got = 0
    while got < nbytes:
        try:
            chunk = os.pread(fd, nbytes - got, offset + got)
        except InterruptedError:
            continue
        if not chunk:
            raise IOError(f"short read at {offset}: {got}/{nbytes}")
        if telemetry is not None and len(chunk) < nbytes - got:
            telemetry.short_reads += 1
        out[got:got+len(chunk)] = chunk; got += len(chunk)
    return bytes(out)


class ShardIndex:
    def __init__(self):
        self.weight_map = json.loads(INDEX_PATH.read_text())["weight_map"]
        self.shards = {}
        for name in sorted(set(self.weight_map.values())):
            path = CHECKPOINT / name
            header, data_start = read_header(path)
            self.shards[name] = (path, os.open(path, os.O_RDONLY), header, data_start)

    def close(self):
        for _, fd, _, _ in self.shards.values():
            os.close(fd)

    def pread_experts(self, key: str, expert_ids: list[int], telemetry=None):
        shard_name = self.weight_map[key]
        path, fd, header, data_start = self.shards[shard_name]
        meta = header[key]; shape = tuple(meta["shape"])
        start, end = meta["data_offsets"]
        per = (end - start) // shape[0]
        chunks = []
        for expert in expert_ids:
            chunks.append(pread_exact(fd, per, data_start + start + expert * per, telemetry))
        flat = b"".join(chunks)
        arr = np.frombuffer(flat, dtype=NP_DTYPES[meta["dtype"]]).reshape((len(expert_ids),) + shape[1:])
        out = mx.array(arr)
        if meta["dtype"] == "BF16":
            out = out.view(mx.bfloat16)
        return out, len(flat), shard_name


class Telemetry:
    def __init__(self):
        self.bytes = 0; self.io_s = 0.0; self.short_reads = 0
        self.layer_bytes = defaultdict(int); self.layer_calls = defaultdict(int)
        self.route_sets = defaultdict(list)

    def record(self, layer, nbytes, dt, ids):
        self.bytes += nbytes; self.io_s += dt
        self.layer_bytes[layer] += nbytes; self.layer_calls[layer] += 1
        self.route_sets[layer].append(list(ids))

    def snapshot(self):
        return {"bytes": self.bytes, "io_s": self.io_s, "short_reads": self.short_reads,
                "mlx_active_mb": mx.get_active_memory()/1e6,
                "mlx_peak_mb": mx.get_peak_memory()/1e6,
                "rss_hwm_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6}


class SwitchProjection:
    """Quantized switch projection over streamed expert weights.

    Performs the same mx.gather_qmm call as mlx-lm's QuantizedSwitchLinear (bias=False),
    without constructing a randomly initialised layer, so decoding draws no random numbers
    beyond those of the sampler."""
    __slots__ = ("weight", "scales", "biases")

    def __init__(self, tensors):
        self.weight = tensors["weight"]; self.scales = tensors["scales"]; self.biases = tensors["biases"]

    def __call__(self, x, indices, sorted_indices=False):
        return mx.gather_qmm(x, self.weight, self.scales, self.biases, rhs_indices=indices,
                             transpose=True, group_size=GROUP_SIZE, bits=BITS, mode=MODE,
                             sorted_indices=sorted_indices)


def make_qsl(inp, out, k, tensors):
    return SwitchProjection(tensors)


class StreamingSwitchGLU(nn.Module):
    def __init__(self, layer_idx, args, shard, telemetry):
        super().__init__(); self.layer_idx = layer_idx; self.args = args
        self.shard = shard; self.telemetry = telemetry; self.activation = SwiGLU()

    def _load(self, expert_ids):
        prefix = f"language_model.model.layers.{self.layer_idx}.mlp.switch_mlp."
        specs = [("gate_proj", self.args.hidden_size, self.args.moe_intermediate_size),
                 ("up_proj", self.args.hidden_size, self.args.moe_intermediate_size),
                 ("down_proj", self.args.moe_intermediate_size, self.args.hidden_size)]
        mods = []; total = 0; t0 = time.perf_counter()
        for proj, inp, out in specs:
            t = {}
            for kind in ("weight", "scales", "biases"):
                arr, nb, _ = self.shard.pread_experts(prefix + proj + "." + kind, expert_ids, self.telemetry)
                t[kind] = arr; total += nb
            mods.append(make_qsl(inp, out, len(expert_ids), t))
        self.telemetry.record(self.layer_idx, total, time.perf_counter()-t0, expert_ids)
        return mods

    def __call__(self, x, indices):
        mx.eval(indices)
        src = np.array(indices); ids = sorted(set(int(v) for v in src.reshape(-1)))
        remap = {e:i for i,e in enumerate(ids)}
        idx = mx.array(np.vectorize(remap.__getitem__)(src).astype(np.int32))
        gate, up, down = self._load(ids)
        xe = mx.expand_dims(x, (-2, -3)); do_sort = idx.size >= 64; inv = None
        if do_sort: xe, idx, inv = _gather_sort(xe, idx)
        xu = up(xe, idx, sorted_indices=do_sort); xg = gate(xe, idx, sorted_indices=do_sort)
        y = down(self.activation(xu, xg), idx, sorted_indices=do_sort)
        if do_sort: y = _scatter_unsort(y, inv, indices.shape)
        return y.squeeze(-2)


def build_streaming_model(config, shard, telemetry):
    args = qwen3_5.ModelArgs.from_dict(config)
    model = qwen3_5.Model(args)
    targs = model.language_model.args
    for i, layer in enumerate(model.layers):
        layer.mlp.switch_mlp = StreamingSwitchGLU(i, targs, shard, telemetry)

    resident = {}
    for name in sorted(set(shard.weight_map.values())):
        w = mx.load(str(CHECKPOINT / name))
        resident.update({k:v for k,v in w.items() if ".mlp.switch_mlp." not in k and "mtp." not in k and not k.startswith("vision_")})
    resident = model.sanitize(resident)

    quant = config["quantization"]
    def pred(path, module):
        if path in quant and isinstance(quant[path], dict): return quant[path]
        if not hasattr(module, "to_quantized"): return False
        return f"{path}.scales" in resident
    nn.quantize(model, group_size=quant["group_size"], bits=quant["bits"], mode=quant.get("mode","affine"), class_predicate=pred)
    model.load_weights(list(resident.items()), strict=False)
    mx.eval(model.parameters()); model.eval()
    return model


def cache_signature(cache):
    out = []
    for i,c in enumerate(cache):
        state = c.state
        flat=[]
        for x in state:
            if x is None: flat.append(None)
            else:
                mx.eval(x); flat.append({"shape":list(x.shape),"sum":float(x.astype(mx.float32).sum())})
        out.append({"i":i,"type":type(c).__name__,"state":flat,
                    "offset":int(getattr(c,"offset",0)),
                    "lengths":None if getattr(c,"lengths",None) is None else np.array(c.lengths).tolist()})
    return out
