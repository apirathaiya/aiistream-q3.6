from __future__ import annotations

import hashlib
import importlib
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load_tokenizer

from .hashpin import PROMOTED_RUNTIME_SHA256, verify_runtime_imports
from .policy import RequestError
from .safety import SafetyState, ThroughputCollapseDetector

MODEL_ID = "qwen3.6-35b-a3b-local"


@dataclass(frozen=True)
class PreparedPrompt:
    messages: list[dict]
    prompt_ids: list[int]
    tools: list[dict] | None = None


@dataclass(frozen=True)
class GenerationResult:
    token_ids: list[int]
    text: str
    text_sha256: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float | None
    elapsed_s: float
def make_nonretaining_telemetry(p1_runtime):
    """Byte and read counters that do not retain per-token route IDs."""
    class NonRetainingTelemetry(p1_runtime.Telemetry):
        def record(self, layer, nbytes, dt, ids):
            self.bytes += nbytes
            self.io_s += dt
            self.layer_bytes[layer] += nbytes
            self.layer_calls[layer] += 1
    return NonRetainingTelemetry()


class RuntimeEngine:
    MODEL_ID = MODEL_ID

    def __init__(self, *, repo_root: str | Path, safety: SafetyState,
                 collapse_fraction: float = 0.5, collapse_window_seconds: float = 60.0,
                 expected_runtime_sha: str = PROMOTED_RUNTIME_SHA256,
                 expert_read_path: str = "prefetch",
                 capture_last_result: bool = False):
        self.repo_root = Path(repo_root)
        self.model_root = self.repo_root / "model"
        self.engine_dir = self.repo_root / "engine"
        self.runtime_path = self.engine_dir / "aiistream_parallel.py"
        self.expert_read_path = expert_read_path
        self.runtime_hashes = verify_runtime_imports(
            self.engine_dir, expert_read_path, p3_expected=expected_runtime_sha
        )
        self.runtime_sha256 = self.runtime_hashes["aiistream_parallel.py"]
        print(f"runtime hashes verified mode={expert_read_path} pins={self.runtime_hashes}", flush=True)
        self.safety = safety
        self.collapse_fraction = float(collapse_fraction)
        self.collapse_window_seconds = float(collapse_window_seconds)
        self.capture_last_result = bool(capture_last_result)
        self.last_result: GenerationResult | None = None
        self._last_lock = threading.Lock()

        if str(self.engine_dir) not in sys.path:
            sys.path.insert(0, str(self.engine_dir))
        self.p1 = importlib.import_module("aiistream_model")
        self.p3 = importlib.import_module("aiistream_parallel")

        config = json.loads(self.p1.CONFIG_PATH.read_text())
        text_cfg = config.get("text_config", config)
        self.model_max_context = int(text_cfg["max_position_embeddings"])
        self.telemetry = make_nonretaining_telemetry(self.p1)
        if self.expert_read_path == "parallel":
            self.shard = self.p3.ParallelShardIndex(parallel_pread=True, workers=8)
        elif self.expert_read_path == "direct":
            zero = importlib.import_module("aiistream_direct")
            self.shard = zero.DirectShardIndex()
        else:
            self.prefetch_mod = importlib.import_module("aiistream_prefetch")
            self.shard = self.prefetch_mod.PrefetchShard()
        self.model = self.p1.build_streaming_model(config, self.shard, self.telemetry)
        if self.expert_read_path == "prefetch":
            self.prefetch_mod.install_blocks(self.model, self.shard, self.telemetry)
        self.tokenizer = load_tokenizer(str(self.model_root / "hf-metadata"))
    @property
    def workers(self) -> int:
        return 8

    def close(self) -> None:
        self.shard.close()

    def prepare(self, messages: list[dict],
                tools: list[dict] | None = None) -> PreparedPrompt:
        template_messages = []
        for message_index, message in enumerate(messages):
            if message.get("role") != "assistant" or "tool_calls" not in message:
                template_messages.append(message)
                continue
            rendered_message = dict(message)
            rendered_calls = []
            for call_index, call in enumerate(message["tool_calls"]):
                rendered_call = dict(call)
                function = dict(call["function"])
                try:
                    arguments = json.loads(function["arguments"])
                except json.JSONDecodeError as exc:
                    raise RequestError(
                        f"messages[{message_index}].tool_calls[{call_index}]."
                        f"function.arguments is not valid JSON: {exc.msg}",
                        code="invalid_tool_arguments",
                    ) from exc
                if not isinstance(arguments, dict):
                    raise RequestError(
                        f"messages[{message_index}].tool_calls[{call_index}]."
                        "function.arguments must decode to a JSON object",
                        code="invalid_tool_arguments",
                    )
                function["arguments"] = arguments
                rendered_call["function"] = function
                rendered_calls.append(rendered_call)
            rendered_message["tool_calls"] = rendered_calls
            template_messages.append(rendered_message)

        kwargs = {"tokenize": True, "add_generation_prompt": True}
        if tools is not None:
            kwargs["tools"] = tools
        ids = self.tokenizer.apply_chat_template(template_messages, **kwargs)
        return PreparedPrompt(
            messages=messages, prompt_ids=[int(x) for x in ids], tools=tools
        )

    def generate_prepared(self, prepared: PreparedPrompt, *, temperature: float,
                          seed: int | None, max_completion_tokens: int,
                          on_text: Callable[[str], None] | None = None,
                          should_stop: Callable[[], bool] | None = None) -> GenerationResult:
        """Generate one completion. should_stop() is polled after every token; when it returns
        True (for example, the client disconnected) generation ends with finish_reason
        "cancelled". The request-scoped reset always runs, even on abort or exceptions."""
        try:
            return self._generate_prepared_inner(prepared, temperature=temperature,
                seed=seed, max_completion_tokens=max_completion_tokens, on_text=on_text,
                should_stop=should_stop)
        finally:
            shard = getattr(self, "shard", None)
            if shard is not None and hasattr(shard, "reset_request"):
                shard.reset_request()

    def _generate_prepared_inner(self, prepared: PreparedPrompt, *, temperature: float,
                          seed: int | None, max_completion_tokens: int,
                          on_text: Callable[[str], None] | None = None,
                          should_stop: Callable[[], bool] | None = None) -> GenerationResult:
        if seed is not None:
            mx.random.seed(seed)
        sampler = make_sampler(temp=float(temperature))
        prompt = mx.array(prepared.prompt_ids, dtype=mx.int32)
        cache = self.model.make_cache()
        detector = ThroughputCollapseDetector(
            self.safety,
            fraction=self.collapse_fraction,
            collapse_seconds=self.collapse_window_seconds,
            early_seconds=60.0,
        )
        detok = self.tokenizer.detokenizer
        detok.reset()
        tokens: list[int] = []
        finish_reason = "length"
        start = time.monotonic()
        first_token_time = None
        window_start = None
        window_tokens = 0
        runner = generate_step(
            prompt,
            self.model,
            max_tokens=int(max_completion_tokens),
            sampler=sampler,
            prompt_cache=cache,
            kv_bits=None,
        )
        try:
            for token, _logprobs in runner:
                mx.eval(token)
                mx.eval(mx.random.state)
                now = time.monotonic()
                tid = int(token)
                if first_token_time is None:
                    first_token_time = now

                if tid in self.tokenizer.eos_token_ids:
                    finish_reason = "stop"
                    break

                tokens.append(tid)
                detok.add_token(tid)
                segment = detok.last_segment
                if segment and on_text is not None:
                    on_text(segment)
                if should_stop is not None and should_stop():
                    finish_reason = "cancelled"
                    break

                if window_start is None:
                    window_start = now
                    detector.reset(start_time=now)
                else:
                    window_tokens += 1
                    elapsed = now - window_start
                    if elapsed >= 10.0:
                        detector.observe(window_tokens / elapsed, now)
                        window_start = now
                        window_tokens = 0
        finally:
            detector.finish()

        detok.finalize()
        tail = detok.last_segment
        if tail and on_text is not None:
            on_text(tail)
        text = self.tokenizer.decode(tokens)
        if detok.text != text:
            raise RuntimeError("streaming detokenizer diverged from tokenizer.decode")
        result = GenerationResult(
            token_ids=tokens,
            text=text,
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            finish_reason=finish_reason,
            prompt_tokens=len(prepared.prompt_ids),
            completion_tokens=len(tokens),
            ttft_s=None if first_token_time is None else first_token_time - start,
            elapsed_s=time.monotonic() - start,
        )
        if self.capture_last_result:
            with self._last_lock:
                self.last_result = result
        return result
