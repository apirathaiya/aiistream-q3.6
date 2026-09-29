from __future__ import annotations

from dataclasses import dataclass


LONG_CONTEXT_WARNING = (
    "Long-context override: measured 65,536-token input at ~9.6 min TTFT "
    "with ~27% intra-request decode decay; 32,768 is the recommended default."
)


class RequestError(ValueError):
    def __init__(self, message: str, *, code: str = "invalid_request", status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: list[dict]
    stream: bool = False
    temperature: float = 0.0
    max_completion_tokens: int = 256
    seed: int | None = None
    tools: list[dict] | None = None


ALLOWED_FIELDS = {
    "model", "messages", "stream", "temperature",
    "max_completion_tokens", "seed", "n", "tools",
}
def _message_error(index: int, message: str) -> RequestError:
    return RequestError(
        f"messages[{index}] {message}", code="unsupported_message_shape"
    )


def _validate_tool_call(value: object, *, message_index: int, call_index: int) -> None:
    where = f"messages[{message_index}].tool_calls[{call_index}]"
    if not isinstance(value, dict) or set(value) != {"id", "type", "function"}:
        raise RequestError(
            f"{where} must contain exactly id, type, function",
            code="unsupported_message_shape",
        )
    if not isinstance(value["id"], str) or not value["id"]:
        raise RequestError(f"{where}.id must be a non-empty string")
    if value["type"] != "function":
        raise RequestError(f"{where}.type must be 'function'")
    function = value["function"]
    if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
        raise RequestError(
            f"{where}.function must contain exactly name and arguments"
        )
    if not isinstance(function["name"], str) or not function["name"]:
        raise RequestError(f"{where}.function.name must be a non-empty string")
    if not isinstance(function["arguments"], str):
        raise RequestError(f"{where}.function.arguments must be a JSON string")
def _validate_message(message: object, index: int) -> None:
    if not isinstance(message, dict):
        raise RequestError(f"messages[{index}] must be an object")
    role = message.get("role")

    if set(message) == {"role", "content"}:
        if role not in {"system", "user", "assistant"}:
            raise _message_error(
                index, "plain-content role must be system, user, or assistant"
            )
        if not isinstance(message["content"], str):
            raise _message_error(index, "content must be a string")
        return

    if role == "assistant" and set(message) == {"role", "content", "tool_calls"}:
        calls = message["tool_calls"]
        if message["content"] is not None and not isinstance(message["content"], str):
            raise _message_error(index, "assistant content must be string or null")
        if not isinstance(calls, list) or not calls:
            raise _message_error(index, "assistant tool_calls must be a non-empty array")
        for call_index, call in enumerate(calls):
            _validate_tool_call(call, message_index=index, call_index=call_index)
        return

    if role == "tool" and set(message) == {"role", "tool_call_id", "content"}:
        if not isinstance(message["tool_call_id"], str) or not message["tool_call_id"]:
            raise _message_error(index, "tool_call_id must be a non-empty string")
        if not isinstance(message["content"], str):
            raise _message_error(index, "tool content must be a string")
        return

    raise _message_error(index, "has an unsupported key/role combination")
def _validate_tools(tools: object) -> list[dict] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise RequestError("tools must be an array", code="invalid_tools")
    validated = []
    for index, tool in enumerate(tools):
        where = f"tools[{index}]"
        if not isinstance(tool, dict) or set(tool) != {"type", "function"}:
            raise RequestError(
                f"{where} must contain exactly type and function", code="invalid_tools"
            )
        if tool["type"] != "function":
            raise RequestError(f"{where}.type must be 'function'", code="invalid_tools")
        function = tool["function"]
        if not isinstance(function, dict):
            raise RequestError(f"{where}.function must be an object", code="invalid_tools")
        extra = set(function) - {"name", "description", "parameters"}
        if "name" not in function or extra:
            raise RequestError(
                f"{where}.function supports only name, description?, parameters?",
                code="invalid_tools",
            )
        if not isinstance(function["name"], str) or not function["name"]:
            raise RequestError(
                f"{where}.function.name must be a non-empty string", code="invalid_tools"
            )
        if "description" in function and not isinstance(function["description"], str):
            raise RequestError(
                f"{where}.function.description must be a string", code="invalid_tools"
            )
        if "parameters" in function and not isinstance(function["parameters"], dict):
            raise RequestError(
                f"{where}.function.parameters must be an object", code="invalid_tools"
            )
        validated.append(tool)
    return validated


def parse_chat_request(body: object, *, model_id: str) -> ChatRequest:
    if not isinstance(body, dict):
        raise RequestError("request body must be a JSON object")
    extra = sorted(set(body) - ALLOWED_FIELDS)
    if extra:
        raise RequestError(
            f"unsupported field(s): {', '.join(extra)}", code="unsupported_field"
        )
    if body.get("n", 1) != 1:
        raise RequestError("n > 1 is out of scope", code="unsupported_n")
    if body.get("model") != model_id:
        raise RequestError(f"model must be {model_id!r}", code="model_not_found")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty array")
    for i, message in enumerate(messages):
        _validate_message(message, i)
    tools = _validate_tools(body.get("tools"))
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise RequestError("stream must be boolean")
    temperature = body.get("temperature", 0.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise RequestError("temperature must be a number")
    temperature = float(temperature)
    if not (temperature >= 0.0 and temperature < float("inf")):
        raise RequestError("temperature must be finite and >= 0")
    max_tokens = body.get("max_completion_tokens", 256)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise RequestError("max_completion_tokens must be a positive integer")
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise RequestError("seed must be an integer or null")
    return ChatRequest(
        model=model_id,
        messages=messages,
        stream=stream,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        seed=seed,
        tools=tools,
    )


@dataclass(frozen=True)
class ContextDecision:
    warning: str | None
def check_context_policy(*, prompt_tokens: int, max_completion_tokens: int,
                         default_max: int, hard_ceiling: int,
                         allow_long_context: bool, model_max: int) -> ContextDecision:
    if prompt_tokens > hard_ceiling:
        raise RequestError(
            f"rendered prompt is {prompt_tokens} tokens; hard ceiling is {hard_ceiling}",
            code="context_length_exceeded",
        )
    warning = None
    if prompt_tokens > default_max:
        if not allow_long_context:
            raise RequestError(
                f"rendered prompt is {prompt_tokens} tokens, above default {default_max}; "
                "retry with X-Qwen36-Allow-Long-Context: true to opt into the measured 65K cost",
                code="long_context_override_required",
            )
        warning = LONG_CONTEXT_WARNING
    if prompt_tokens + max_completion_tokens > model_max:
        raise RequestError(
            f"prompt + max_completion_tokens exceeds model maximum {model_max}",
            code="model_context_exceeded",
        )
    return ContextDecision(warning=warning)
