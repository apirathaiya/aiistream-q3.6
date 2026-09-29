from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable


TOOL_OPEN = "<tool_call>"
FUNCTION_OPEN = "<function="
PARAMETER_OPEN = "<parameter="
FUNCTION_CLOSE = "</function>\n</tool_call>"
PARAMETER_CLOSE = "\n</parameter>\n"


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, str]
    raw: str


@dataclass(frozen=True)
class ToolParseResult:
    content: str
    calls: list[ParsedToolCall]
    malformed: bool
    raw_text: str


def _valid_name(value: str) -> bool:
    return bool(value) and all(ch not in value for ch in "<>\r\n")
def parse_tool_output(text: str) -> ToolParseResult:
    """Parse the checkpoint template's exact XML-ish tool-call format.

    Parameter values remain strings exactly as emitted between the template's
    framing newlines. If a tool marker is present but any framing is invalid,
    no partial calls are returned and the complete raw text becomes content.
    """
    first = text.find(TOOL_OPEN)
    if first < 0:
        return ToolParseResult(text, [], False, text)

    prefix = text[:first]
    calls: list[ParsedToolCall] = []
    pos = first

    try:
        while True:
            start = pos
            if not text.startswith(TOOL_OPEN + "\n" + FUNCTION_OPEN, pos):
                raise ValueError("tool call must start with template framing")
            pos += len(TOOL_OPEN) + 1 + len(FUNCTION_OPEN)

            name_end = text.find(">\n", pos)
            if name_end < 0:
                raise ValueError("unterminated function name")
            name = text[pos:name_end]
            if not _valid_name(name):
                raise ValueError("invalid function name")
            pos = name_end + 2
            arguments: dict[str, str] = {}
            while not text.startswith(FUNCTION_CLOSE, pos):
                if not text.startswith(PARAMETER_OPEN, pos):
                    raise ValueError("expected parameter or function close")
                pos += len(PARAMETER_OPEN)
                parameter_end = text.find(">\n", pos)
                if parameter_end < 0:
                    raise ValueError("unterminated parameter name")
                parameter = text[pos:parameter_end]
                if not _valid_name(parameter) or parameter in arguments:
                    raise ValueError("invalid or duplicate parameter name")
                pos = parameter_end + 2

                value_end = text.find(PARAMETER_CLOSE, pos)
                if value_end < 0:
                    raise ValueError("unterminated parameter value")
                arguments[parameter] = text[pos:value_end]
                pos = value_end + len(PARAMETER_CLOSE)

            pos += len(FUNCTION_CLOSE)
            calls.append(ParsedToolCall(name, arguments, text[start:pos]))

            if pos == len(text):
                break
            if not text.startswith("\n" + TOOL_OPEN, pos):
                raise ValueError("tool calls must use the template's single-newline separator")
            pos += 1
        if not calls:
            raise ValueError("no tool calls parsed")
        return ToolParseResult(prefix, calls, False, text)
    except ValueError:
        return ToolParseResult(text, [], True, text)


def openai_tool_calls(calls: list[ParsedToolCall], response_id: str) -> list[dict]:
    stem = response_id.removeprefix("chatcmpl-")
    out = []
    for index, call in enumerate(calls):
        out.append({
            "id": f"call_{stem}_{index}",
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(
                    call.arguments, ensure_ascii=False, separators=(",", ":")
                ),
            },
        })
    return out


class ToolCallStreamGate:
    """Emit normal content while withholding possible tool-call framing."""

    def __init__(self, emit: Callable[[str], None]):
        self.emit = emit
        self.pending = ""
        self.blocked = False

    def feed(self, segment: str) -> None:
        if not segment:
            return
        self.pending += segment
        if self.blocked:
            return
        marker_index = self.pending.find(TOOL_OPEN)
        if marker_index >= 0:
            safe = self.pending[:marker_index]
            if safe:
                self.emit(safe)
            self.pending = self.pending[marker_index:]
            self.blocked = True
            return

        keep = 0
        max_keep = min(len(self.pending), len(TOOL_OPEN) - 1)
        for size in range(max_keep, 0, -1):
            if TOOL_OPEN.startswith(self.pending[-size:]):
                keep = size
                break
        safe = self.pending[:-keep] if keep else self.pending
        if safe:
            self.emit(safe)
        self.pending = self.pending[-keep:] if keep else ""

    def flush_raw(self) -> None:
        if self.pending:
            self.emit(self.pending)
            self.pending = ""
