from __future__ import annotations

import argparse
import json
import select
import socket
import sys
import traceback
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .app import ServiceApp
from .config import ConfigError, load_config
from .policy import RequestError, parse_chat_request
from .runtime_engine import MODEL_ID, RuntimeEngine
from .safety import SafetyState
from .telemetry import SystemTelemetrySource, TelemetryMonitor
from .tool_calls import ToolCallStreamGate, openai_tool_calls, parse_tool_output

RETRY_AFTER_SECONDS = 5
LONG_CONTEXT_HEADER = "X-Qwen36-Allow-Long-Context"
WARNING_HEADER = "X-Qwen36-Long-Context-Warning"


def error_body(message: str, code: str) -> dict:
    return {"error": {"message": message, "type": "invalid_request_error", "code": code}}


class QwenHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, app: ServiceApp):
        self.app = app
        super().__init__(addr, QwenHandler)
class QwenHandler(BaseHTTPRequestHandler):
    server_version = "server/0.1"

    @property
    def app(self) -> ServiceApp:
        return self.server.app

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

    def _json(self, status: int, body: dict, *, extra_headers: dict | None = None):
        payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status: int, message: str, code: str, *, retry_after: bool = False):
        headers = {"Retry-After": RETRY_AFTER_SECONDS} if retry_after else None
        self._json(status, error_body(message, code), extra_headers=headers)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, self.app.health())
            return
        if self.path == "/v1/models":
            self._json(200, self.app.models())
            return
        self._error(404, "not found", "not_found")
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._error(404, "not found", "not_found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            chat = parse_chat_request(body, model_id=MODEL_ID)
        except RequestError as exc:
            self._error(exc.status, str(exc), exc.code)
            return
        except Exception as exc:
            self._error(400, f"invalid JSON request: {exc}", "invalid_json")
            return

        ticket, reason = self.app.admission.try_admit()
        if ticket is None:
            self._error(503, f"request not admitted: {reason}", reason or "admission_blocked", retry_after=True)
            return

        try:
            ticket.begin_generation()
            allow_long = self.headers.get(LONG_CONTEXT_HEADER, "").strip().lower() == "true"
            try:
                prepared = self.app.prepare(chat, allow_long_context=allow_long)
            except RequestError as exc:
                self._error(exc.status, str(exc), exc.code)
                return
            if chat.stream:
                self._stream(prepared)
            else:
                response, _result = self.app.nonstream_response(prepared, should_stop=self._client_gone)
                headers = {}
                if prepared.context.warning:
                    headers[WARNING_HEADER] = prepared.context.warning
                    headers["Warning"] = f'299 server "{prepared.context.warning}"'
                self._json(200, response, extra_headers=headers)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            if not self.wfile.closed:
                try:
                    self._error(500, f"generation failed: {type(exc).__name__}: {exc}", "generation_failed")
                except Exception:
                    pass
        finally:
            ticket.release()

    def _client_gone(self) -> bool:
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True

    def _stream(self, prepared):
        rid = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        if prepared.context.warning:
            self.send_header(WARNING_HEADER, prepared.context.warning)
            self.send_header("Warning", f'299 server "{prepared.context.warning}"')
        self.end_headers()
        client_alive = True

        def send_chunk(delta: dict, finish_reason=None):
            nonlocal client_alive
            if not client_alive:
                return
            chunk = {
                "id": rid, "object": "chat.completion.chunk", "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            data = ("data: " + json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n\n").encode()
            try:
                self.wfile.write(data)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                client_alive = False

        send_chunk({"role": "assistant"})
        try:
            stream_gate = None
            on_text = lambda segment: send_chunk({"content": segment})
            if prepared.chat.tools:
                stream_gate = ToolCallStreamGate(on_text)
                on_text = stream_gate.feed

            result = self.app.engine.generate_prepared(
                prepared.prepared_prompt,
                temperature=prepared.chat.temperature,
                seed=prepared.chat.seed,
                max_completion_tokens=prepared.chat.max_completion_tokens,
                on_text=on_text,
                should_stop=lambda: (not client_alive) or self._client_gone(),
            )

            finish_reason = result.finish_reason
            if prepared.chat.tools:
                parsed = parse_tool_output(result.text)
                if parsed.calls:
                    for index, call in enumerate(
                        openai_tool_calls(parsed.calls, rid)
                    ):
                        send_chunk({"tool_calls": [{
                            "index": index,
                            "id": call["id"],
                            "type": call["type"],
                            "function": call["function"],
                        }]})
                    finish_reason = "tool_calls"
                elif parsed.malformed:
                    stream_gate.flush_raw()
                    finish_reason = "stop"
                else:
                    stream_gate.flush_raw()

            send_chunk({}, finish_reason=finish_reason)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            if client_alive:
                payload = {"error": {"message": f"generation failed: {type(exc).__name__}: {exc}", "code": "generation_failed"}}
                try:
                    self.wfile.write(("data: " + json.dumps(payload, separators=(",", ":")) + "\n\n").encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    client_alive = False
        if client_alive:
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass


def build_app(*, config_path: str | Path, capture_last_result: bool = False):
    service_root = Path(__file__).resolve().parents[1]
    repo_root = service_root.parent
    config = load_config(config_path)
    safety = SafetyState()
    engine = RuntimeEngine(
        repo_root=repo_root,
        safety=safety,
        collapse_fraction=config.safety.collapse_fraction,
        collapse_window_seconds=config.safety.collapse_window_seconds,
        capture_last_result=capture_last_result,
        expert_read_path=config.expert_read_path,
    )
    source = SystemTelemetrySource(service_root / "bin" / "thermal_state")
    monitor = TelemetryMonitor(
        source, safety,
        poll_seconds=config.telemetry.poll_seconds,
        log_path=service_root / "logs" / "telemetry_events.jsonl",
    )
    monitor.start()
    app = ServiceApp(config=config, engine=engine, telemetry_monitor=monitor, safety=safety)
    return app, monitor, engine


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Local OpenAI-compatible Qwen3.6 service")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.json"))
    args = ap.parse_args(argv)
    try:
        app, monitor, engine = build_app(config_path=args.config)
    except (ConfigError, Exception) as exc:
        print(f"startup refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    server = None
    try:
        server = QwenHTTPServer(("127.0.0.1", app.config.port), app)
        print(f"server ready http://127.0.0.1:{app.config.port} model={MODEL_ID} workers=8 expert_read_path={app.config.expert_read_path}", flush=True)
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        monitor.stop()
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
