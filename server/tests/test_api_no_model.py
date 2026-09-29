import hashlib
import json
import threading
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace

from aiistream_server.app import ServiceApp
from aiistream_server.config import SafetyConfig, ServiceConfig, TelemetryConfig
from aiistream_server.runtime_engine import GenerationResult, MODEL_ID
from aiistream_server.safety import SafetyState
from aiistream_server.server import QwenHTTPServer


class FakeMonitor:
    def latest(self):
        return {"sample": {"battery_c": 30.0, "memory_pressure_level": 1}, "error": None}


class FakeEngine:
    MODEL_ID = MODEL_ID
    model_max_context = 262144

    def __init__(self):
        self.prepare_calls = 0

    def prepare(self, messages, tools=None):
        self.prepare_calls += 1
        n = 40000 if messages[0]["content"] == "LONG" else 3
        return SimpleNamespace(messages=messages, prompt_ids=list(range(n)), tools=tools)

    def generate_prepared(self, prepared, *, temperature, seed, max_completion_tokens, on_text=None, should_stop=None):
        marker = prepared.messages[0]["content"]
        if marker == "TOOL":
            text = (
                "<think>keep reasoning</think>\n\n"
                "<tool_call>\n<function=list_dir>\n<parameter=path>\n.\n"
                "</parameter>\n</function>\n</tool_call>"
            )
            pieces = [
                "<think>keep reasoning</think>\n\n<tool_",
                "call>\n<function=list_dir>\n<parameter=path>\n.\n",
                "</parameter>\n</function>\n</tool_call>",
            ]
        elif marker == "MALFORMED":
            text = "prefix\n<tool_call>broken"
            pieces = ["prefix\n<tool_", "call>broken"]
        else:
            text = "HELLO"
            pieces = ["HEL", "LO"]
        if on_text is not None:
            for piece in pieces:
                on_text(piece)
        return GenerationResult(
            token_ids=[1, 2], text=text,
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            finish_reason="length", prompt_tokens=len(prepared.prompt_ids),
            completion_tokens=2, ttft_s=0.01, elapsed_s=0.02,
        )
class ApiNoModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.safety = SafetyState()
        cls.engine = FakeEngine()
        cfg = ServiceConfig(
            port=8081, queue_limit=4, max_context=32768,
            max_context_hard_ceiling=65536,
            telemetry=TelemetryConfig(5), safety=SafetyConfig(),
        )
        cls.app = ServiceApp(config=cfg, engine=cls.engine, telemetry_monitor=FakeMonitor(), safety=cls.safety)
        cls.server = QwenHTTPServer(("127.0.0.1", 0), cls.app)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def post(self, body, headers=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        return urllib.request.urlopen(req, timeout=5)

    def base_body(self, **extra):
        body = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0,
            "max_completion_tokens": 4,
            "seed": 7,
        }
        body.update(extra)
        return body

    def tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "list a directory",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }]
    def test_nonstream_openai_shape(self):
        with self.post(self.base_body()) as resp:
            self.assertEqual(resp.status, 200)
            body = json.loads(resp.read())
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["model"], MODEL_ID)
        self.assertEqual(body["choices"][0]["message"]["content"], "HELLO")
        self.assertEqual(body["usage"]["completion_tokens"], 2)

    def test_stream_sse_framing(self):
        with self.post(self.base_body(stream=True)) as resp:
            self.assertEqual(resp.headers["Content-Type"], "text/event-stream; charset=utf-8")
            raw = resp.read().decode()
        self.assertIn('"delta":{"role":"assistant"}', raw)
        self.assertIn('"content":"HEL"', raw)
        self.assertIn('"content":"LO"', raw)
        self.assertTrue(raw.endswith("data: [DONE]\n\n"))

    def test_nonstream_tool_call_shape(self):
        body = self.base_body(
            messages=[{"role": "user", "content": "TOOL"}],
            tools=self.tools(),
        )
        with self.post(body) as resp:
            payload = json.loads(resp.read())
        choice = payload["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["content"], "<think>keep reasoning</think>\n\n")
        calls = choice["message"]["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "list_dir")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "."})

    def test_malformed_tool_output_is_raw_stop_without_calls(self):
        raw = "prefix\n<tool_call>broken"
        body = self.base_body(
            messages=[{"role": "user", "content": "MALFORMED"}],
            tools=self.tools(),
        )
        with self.post(body) as resp:
            payload = json.loads(resp.read())
        choice = payload["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(choice["message"]["content"], raw)
        self.assertNotIn("tool_calls", choice["message"])

    def test_stream_tool_xml_is_not_content(self):
        body = self.base_body(
            messages=[{"role": "user", "content": "TOOL"}],
            tools=self.tools(),
            stream=True,
        )
        with self.post(body) as resp:
            raw = resp.read().decode()
        self.assertNotIn("<tool_call>", raw)
        self.assertIn('"content":"<think>keep reasoning</think>\\n\\n"', raw)
        self.assertIn('"tool_calls":[{"index":0', raw)
        self.assertIn('"finish_reason":"tool_calls"', raw)
        self.assertTrue(raw.endswith("data: [DONE]\n\n"))

    def test_stream_malformed_restores_raw_and_stops(self):
        body = self.base_body(
            messages=[{"role": "user", "content": "MALFORMED"}],
            tools=self.tools(),
            stream=True,
        )
        with self.post(body) as resp:
            raw = resp.read().decode()
        self.assertIn('"content":"<tool_call>broken"', raw)
        self.assertNotIn('"tool_calls":', raw)
        self.assertIn('"finish_reason":"stop"', raw)

    def test_unsupported_field_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post(self.base_body(top_p=0.9))
        self.assertEqual(cm.exception.code, 400)
        err = json.loads(cm.exception.read())
        cm.exception.close()
        self.assertEqual(err["error"]["code"], "unsupported_field")

    def test_queue_full_is_503_before_prepare_or_tokenization(self):
        tickets = []
        for _ in range(self.app.admission.capacity):
            ticket, reason = self.app.admission.try_admit()
            self.assertIsNone(reason)
            tickets.append(ticket)
        before = self.engine.prepare_calls
        try:
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.post(self.base_body())
            self.assertEqual(cm.exception.code, 503)
            self.assertEqual(cm.exception.headers["Retry-After"], "5")
            err = json.loads(cm.exception.read())
            cm.exception.close()
            self.assertEqual(err["error"]["code"], "queue_full")
            self.assertEqual(self.engine.prepare_calls, before)
        finally:
            for ticket in tickets:
                ticket.release()

    def test_n_gt_one_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post(self.base_body(n=2))
        err = json.loads(cm.exception.read())
        cm.exception.close()
        self.assertEqual(err["error"]["code"], "unsupported_n")
    def test_long_context_requires_explicit_header_and_surfaces_cost(self):
        body = self.base_body(messages=[{"role": "user", "content": "LONG"}])
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post(body)
        err = json.loads(cm.exception.read())
        cm.exception.close()
        self.assertEqual(err["error"]["code"], "long_context_override_required")
        with self.post(body, headers={"X-Qwen36-Allow-Long-Context": "true"}) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("9.6 min TTFT", resp.headers["X-Qwen36-Long-Context-Warning"])

    def test_health_and_models(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as resp:
            health = json.loads(resp.read())
        self.assertEqual(health["workers"], 8)
        self.assertNotIn("governor", health)
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=5) as resp:
            models = json.loads(resp.read())
        self.assertEqual(models["data"][0]["id"], MODEL_ID)


if __name__ == "__main__":
    unittest.main()
