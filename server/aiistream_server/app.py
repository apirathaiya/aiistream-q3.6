from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from .config import ServiceConfig
from .policy import ChatRequest, ContextDecision, check_context_policy
from .safety import AdmissionController, SafetyState
from .tool_calls import openai_tool_calls, parse_tool_output


@dataclass(frozen=True)
class PreparedServiceRequest:
    chat: ChatRequest
    prepared_prompt: object
    context: ContextDecision


class ServiceApp:
    def __init__(self, *, config: ServiceConfig, engine, telemetry_monitor,
                 safety: SafetyState):
        self.config = config
        self.engine = engine
        self.telemetry_monitor = telemetry_monitor
        self.safety = safety
        self.admission = AdmissionController(
            queue_limit=config.queue_limit,
            safety=safety,
            memory_stop=config.safety.memory_pressure_stop,
            throughput_stop=config.safety.throughput_collapse_stop,
        )

    @property
    def model_id(self) -> str:
        return self.engine.MODEL_ID if hasattr(self.engine, "MODEL_ID") else getattr(self.engine, "model_id", "qwen3.6-35b-a3b-local")
    def prepare(self, chat: ChatRequest, *, allow_long_context: bool) -> PreparedServiceRequest:
        prepared = self.engine.prepare(chat.messages, tools=chat.tools)
        decision = check_context_policy(
            prompt_tokens=len(prepared.prompt_ids),
            max_completion_tokens=chat.max_completion_tokens,
            default_max=self.config.max_context,
            hard_ceiling=self.config.max_context_hard_ceiling,
            allow_long_context=allow_long_context,
            model_max=self.engine.model_max_context,
        )
        return PreparedServiceRequest(chat=chat, prepared_prompt=prepared, context=decision)

    def nonstream_response(self, req: PreparedServiceRequest, should_stop=None):
        result = self.engine.generate_prepared(
            req.prepared_prompt,
            temperature=req.chat.temperature,
            seed=req.chat.seed,
            max_completion_tokens=req.chat.max_completion_tokens,
            should_stop=should_stop,
        )
        created = int(time.time())
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        parsed = parse_tool_output(result.text) if req.chat.tools else None
        message = {"role": "assistant", "content": result.text}
        finish_reason = result.finish_reason
        if parsed is not None and parsed.calls:
            message = {
                "role": "assistant",
                "content": parsed.content if parsed.content else None,
                "tool_calls": openai_tool_calls(parsed.calls, response_id),
            }
            finish_reason = "tool_calls"
        elif parsed is not None and parsed.malformed:
            message = {"role": "assistant", "content": result.text}
            finish_reason = "stop"

        body = {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": self.model_id,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }
        return body, result

    def health(self) -> dict:
        return {
            "status": "ok",
            "model": self.model_id,
            "model_loaded": True,
            "workers": 8,
            "queue": self.admission.snapshot(),
            "safety": self.safety.snapshot(),
            "telemetry": self.telemetry_monitor.latest(),
            "context": {
                "default_max_context": self.config.max_context,
                "hard_ceiling": self.config.max_context_hard_ceiling,
            },
        }

    def models(self) -> dict:
        return {
            "object": "list",
            "data": [{
                "id": self.model_id,
                "object": "model",
                "owned_by": "local",
                "context_default": self.config.max_context,
                "context_hard_ceiling": self.config.max_context_hard_ceiling,
            }],
        }
