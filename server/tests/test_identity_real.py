import hashlib
import json
import os
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

from aiistream_server.app import ServiceApp
from aiistream_server.config import SafetyConfig, ServiceConfig, TelemetryConfig
from aiistream_server.runtime_engine import MODEL_ID, RuntimeEngine
from aiistream_server.safety import SafetyState
from aiistream_server.server import QwenHTTPServer


class FakeMonitor:
    def latest(self):
        return {"sample": {"memory_pressure_level": 1}, "error": None}


def mlx_reference(
    engine, messages, *, temperature, seed, max_tokens, sampler=None, tools=None
):
    """Reference output produced by mlx-lm's own stream_generate loop."""
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    if tools is not None:
        kwargs["tools"] = tools
    prompt_ids = engine.tokenizer.apply_chat_template(messages, **kwargs)
    engine.tokenizer.detokenizer.reset()
    mx.random.seed(seed)
    sampler = sampler or make_sampler(temp=temperature)
    cache = engine.model.make_cache()
    responses = list(stream_generate(
        engine.model,
        engine.tokenizer,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=cache,
        kv_bits=None,
    ))
    final = responses[-1]
    token_ids = [int(r.token) for r in responses if r.finish_reason is None]
    if final.finish_reason == "length":
        token_ids.append(int(final.token))
    text = "".join(r.text for r in responses)
    return {
        "token_ids": token_ids,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "finish_reason": final.finish_reason,
    }


@unittest.skipUnless(os.environ.get("QWEN36_REAL_TEST") == "1", "set QWEN36_REAL_TEST=1")
class RealIdentityTest(unittest.TestCase):
    def assert_no_control_tokens(self, engine, result, content):
        eos = set(engine.tokenizer.eos_token_ids)
        self.assertFalse(eos.intersection(result.token_ids))
        for special in engine.tokenizer.all_special_tokens:
            if special:
                self.assertNotIn(special, content)
    def test_service_vs_mlx_lm_reference_identity_and_eos_framing(self):
        root = Path(__file__).resolve().parents[2]
        service_root = Path(__file__).resolve().parents[1]
        safety = SafetyState()
        engine = RuntimeEngine(
            repo_root=root,
            safety=safety,
            collapse_fraction=0.5,
            collapse_window_seconds=60,
            capture_last_result=True,
        )
        cfg = ServiceConfig(
            port=8081, queue_limit=4, max_context=32768,
            max_context_hard_ceiling=65536,
            telemetry=TelemetryConfig(5), safety=SafetyConfig(),
        )
        app = ServiceApp(
            config=cfg, engine=engine,
            telemetry_monitor=FakeMonitor(), safety=safety,
        )
        server = QwenHTTPServer(("127.0.0.1", 0), app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def post(messages, *, temperature, seed, max_tokens, tools=None):
            body = {
                "model": MODEL_ID,
                "messages": messages,
                "temperature": temperature,
                "seed": seed,
                "max_completion_tokens": max_tokens,
                "stream": False,
            }
            if tools is not None:
                body["tools"] = tools
            req = urllib.request.Request(
                f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())

        evidence = {"reference": "mlx_lm.generate.stream_generate"}
        try:
            messages = [{"role": "user", "content": "Reply with one short sentence about the moon."}]
            temperature = 0.7
            seed = 424242
            max_tokens = 8
            service_body = post(
                messages, temperature=temperature, seed=seed, max_tokens=max_tokens
            )
            service_result = engine.last_result
            self.assertIsNotNone(service_result)
            reference = mlx_reference(
                engine, messages,
                temperature=temperature, seed=seed, max_tokens=max_tokens,
            )
            service_text = service_body["choices"][0]["message"]["content"]

            self.assertEqual(service_result.token_ids, reference["token_ids"])
            self.assertEqual(service_result.text_sha256, reference["text_sha256"])
            self.assertEqual(service_text, reference["text"])
            self.assertEqual(service_result.finish_reason, reference["finish_reason"])
            self.assertEqual(service_result.completion_tokens, len(service_result.token_ids))
            self.assertEqual(service_body["usage"]["completion_tokens"], len(reference["token_ids"]))
            self.assert_no_control_tokens(engine, service_result, service_text)
            evidence["seeded_identity"] = {
                "temperature": temperature,
                "seed": seed,
                "max_completion_tokens": max_tokens,
                "service_token_ids": service_result.token_ids,
                "reference_token_ids": reference["token_ids"],
                "service_text_sha256": service_result.text_sha256,
                "reference_text_sha256": reference["text_sha256"],
                "http_text_sha256": hashlib.sha256(service_text.encode()).hexdigest(),
                "finish_reason": service_result.finish_reason,
                "completion_tokens": service_result.completion_tokens,
            }

            tools = [{
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "List a directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }]
            tool_messages = [{
                "role": "user",
                "content": "Reply with one short sentence about the moon. Do not call tools.",
            }]
            tool_seed = 31337
            tool_max_tokens = 8
            tool_body = post(
                tool_messages,
                temperature=temperature,
                seed=tool_seed,
                max_tokens=tool_max_tokens,
                tools=tools,
            )
            tool_result = engine.last_result
            self.assertIsNotNone(tool_result)
            tool_reference = mlx_reference(
                engine,
                tool_messages,
                temperature=temperature,
                seed=tool_seed,
                max_tokens=tool_max_tokens,
                tools=tools,
            )
            self.assertEqual(tool_result.token_ids, tool_reference["token_ids"])
            self.assertEqual(tool_result.text_sha256, tool_reference["text_sha256"])
            self.assertEqual(tool_result.text, tool_reference["text"])
            self.assertEqual(
                tool_body["usage"]["completion_tokens"], len(tool_reference["token_ids"])
            )
            print(
                "TOOLS_IDENTITY",
                f"service_token_ids={tool_result.token_ids}",
                f"reference_token_ids={tool_reference['token_ids']}",
                f"service_text_sha256={tool_result.text_sha256}",
                f"reference_text_sha256={tool_reference['text_sha256']}",
            )
            evidence["tools_identity"] = {
                "temperature": temperature,
                "seed": tool_seed,
                "max_completion_tokens": tool_max_tokens,
                "service_token_ids": tool_result.token_ids,
                "reference_token_ids": tool_reference["token_ids"],
                "service_text": tool_result.text,
                "reference_text": tool_reference["text"],
                "service_text_sha256": tool_result.text_sha256,
                "reference_text_sha256": tool_reference["text_sha256"],
            }

            eos_id = int(next(iter(engine.tokenizer.eos_token_ids)))

            def forced_eos_sampler(_logprobs):
                return mx.array([eos_id])

            eos_messages = [{"role": "user", "content": "EOS framing probe."}]
            eos_seed = 7
            eos_max_tokens = 4
            with patch(
                "aiistream_server.runtime_engine.make_sampler",
                return_value=forced_eos_sampler,
            ):
                eos_body = post(
                    eos_messages,
                    temperature=0.7,
                    seed=eos_seed,
                    max_tokens=eos_max_tokens,
                )
            eos_result = engine.last_result
            self.assertIsNotNone(eos_result)
            eos_reference = mlx_reference(
                engine,
                eos_messages,
                temperature=0.7,
                seed=eos_seed,
                max_tokens=eos_max_tokens,
                sampler=forced_eos_sampler,
            )
            eos_text = eos_body["choices"][0]["message"]["content"]
            print(
                "EOS_PROBE",
                f"service_token_ids={eos_result.token_ids}",
                f"reference_token_ids={eos_reference['token_ids']}",
                f"service_completion_tokens={eos_body['usage']['completion_tokens']}",
                f"reference_completion_tokens={len(eos_reference['token_ids'])}",
                f"service_text={eos_text!r}",
                f"reference_text={eos_reference['text']!r}",
            )

            self.assert_no_control_tokens(engine, eos_result, eos_text)
            self.assertNotIn(eos_id, eos_result.token_ids)
            self.assertNotIn("<|im_end|>", eos_text)
            self.assertEqual(eos_result.token_ids, eos_reference["token_ids"])
            self.assertEqual(eos_result.text_sha256, eos_reference["text_sha256"])
            self.assertEqual(eos_text, eos_reference["text"])
            self.assertEqual(eos_result.finish_reason, "stop")
            self.assertEqual(eos_reference["finish_reason"], "stop")
            self.assertEqual(eos_result.completion_tokens, len(eos_result.token_ids))
            self.assertEqual(eos_body["usage"]["completion_tokens"], len(eos_reference["token_ids"]))
            self.assertEqual(eos_result.token_ids, [])
            self.assertEqual(eos_text, "")
            self.assertIsNotNone(eos_result.ttft_s)
            self.assertGreaterEqual(eos_result.ttft_s, 0.0)

            evidence["forced_eos_probe"] = {
                "eos_token_ids": list(engine.tokenizer.eos_token_ids),
                "service_token_ids": eos_result.token_ids,
                "reference_token_ids": eos_reference["token_ids"],
                "service_text": eos_text,
                "reference_text": eos_reference["text"],
                "service_finish_reason": eos_result.finish_reason,
                "reference_finish_reason": eos_reference["finish_reason"],
                "completion_tokens": eos_result.completion_tokens,
                "ttft_s": eos_result.ttft_s,
            }
            (service_root / "test-results" / "real_identity.json").write_text(
                json.dumps(evidence, indent=2) + "\n"
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
            engine.close()


if __name__ == "__main__":
    unittest.main()
