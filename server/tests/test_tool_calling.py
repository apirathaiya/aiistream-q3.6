import json
import unittest

from aiistream_server.policy import RequestError, parse_chat_request
from aiistream_server.runtime_engine import MODEL_ID, RuntimeEngine
from aiistream_server.tool_calls import (
    ToolCallStreamGate,
    parse_tool_output,
)


TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "read a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}]


def call(name="read_file", params=None):
    params = params or [("path", "README.md")]
    pieces = ["<tool_call>\n", f"<function={name}>\n"]
    for key, value in params:
        pieces += [f"<parameter={key}>\n", value, "\n</parameter>\n"]
    pieces += ["</function>\n</tool_call>"]
    return "".join(pieces)
class ToolParserTests(unittest.TestCase):
    def test_single_call(self):
        raw = call()
        parsed = parse_tool_output(raw)
        self.assertFalse(parsed.malformed)
        self.assertEqual(parsed.content, "")
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(parsed.calls[0].name, "read_file")
        self.assertEqual(parsed.calls[0].arguments, {"path": "README.md"})

    def test_multiple_calls_preserve_order(self):
        raw = call("list_dir", [("path", ".")]) + "\n" + call(
            "read_file", [("path", "README.md")]
        )
        parsed = parse_tool_output(raw)
        self.assertEqual([c.name for c in parsed.calls], ["list_dir", "read_file"])
        self.assertEqual(parsed.calls[0].arguments, {"path": "."})
        self.assertEqual(parsed.calls[1].arguments, {"path": "README.md"})

    def test_call_after_think_preserves_prefix(self):
        prefix = "<think>\nreasoning stays here\n</think>\n\n"
        parsed = parse_tool_output(prefix + call())
        self.assertEqual(parsed.content, prefix)
        self.assertEqual(len(parsed.calls), 1)

    def test_multiline_and_angle_bracket_values_are_strings(self):
        raw = call("read_file", [
            ("path", "line one\nline two"),
            ("literal", "a < b > c"),
        ])
        parsed = parse_tool_output(raw)
        self.assertEqual(parsed.calls[0].arguments["path"], "line one\nline two")
        self.assertEqual(parsed.calls[0].arguments["literal"], "a < b > c")
        self.assertTrue(all(isinstance(v, str) for v in parsed.calls[0].arguments.values()))
    def test_malformed_returns_every_character_raw_and_no_calls(self):
        malformed = [
            "<tool_call>\n<function=read_file>\n<parameter=path>\nREADME.md",
            "<tool_call>\n<function=>\n</function>\n</tool_call>",
            call() + "\nNOT-ALLOWED-SUFFIX",
            "<tool_call>\n<function=x>\n<parameter=a>\n1\n</parameter>\n"
            "<parameter=a>\n2\n</parameter>\n</function>\n</tool_call>",
        ]
        for raw in malformed:
            with self.subTest(raw=raw):
                parsed = parse_tool_output(raw)
                self.assertTrue(parsed.malformed)
                self.assertEqual(parsed.calls, [])
                self.assertEqual(parsed.content, raw)
                self.assertEqual(parsed.raw_text, raw)

    def test_stream_gate_withholds_tool_xml(self):
        emitted = []
        gate = ToolCallStreamGate(emitted.append)
        prefix = "<think>ok</think>\n\n"
        raw = prefix + call()
        for piece in [raw[:12], raw[12:25], raw[25:31], raw[31:]]:
            gate.feed(piece)
        self.assertEqual("".join(emitted), prefix)
        self.assertNotIn("<tool_call>", "".join(emitted))

    def test_stream_gate_can_restore_malformed_raw(self):
        emitted = []
        gate = ToolCallStreamGate(emitted.append)
        raw = "prefix\n<tool_call>broken"
        for piece in ["prefix\n<tool_", "call>broken"]:
            gate.feed(piece)
        gate.flush_raw()
        self.assertEqual("".join(emitted), raw)
class PolicyToolTests(unittest.TestCase):
    def body(self, messages, **extra):
        value = {"model": MODEL_ID, "messages": messages}
        value.update(extra)
        return value

    def test_permitted_message_shapes(self):
        normal = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
        ]
        parse_chat_request(self.body(normal), model_id=MODEL_ID)

        assistant_tool = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
            }],
        }
        tool_result = {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "file contents",
        }
        parse_chat_request(
            self.body([{"role": "user", "content": "u"}, assistant_tool, tool_result]),
            model_id=MODEL_ID,
        )

    def test_assistant_plain_content_accepted(self):
        parsed = parse_chat_request(
            self.body([{"role": "assistant", "content": "plain reply"}]),
            model_id=MODEL_ID,
        )
        self.assertEqual(parsed.messages[0]["content"], "plain reply")

    def test_assistant_null_without_tool_calls_rejected(self):
        with self.assertRaises(RequestError):
            parse_chat_request(
                self.body([{"role": "assistant", "content": None}]),
                model_id=MODEL_ID,
            )

    def test_realistic_history_replay_accepts_assistant_plain_content(self):
        history = [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Turn one"},
            {"role": "assistant", "content": "Reply one"},
            {"role": "user", "content": "Turn two"},
        ]
        parsed = parse_chat_request(self.body(history), model_id=MODEL_ID)
        self.assertEqual(parsed.messages, history)
    def test_unknown_message_key_still_rejected(self):
        with self.assertRaises(RequestError):
            parse_chat_request(
                self.body([{"role": "user", "content": "u", "name": "surprise"}]),
                model_id=MODEL_ID,
            )

    def test_tools_shape_strict_and_out_of_scope_fields_rejected(self):
        parsed = parse_chat_request(
            self.body([{"role": "user", "content": "u"}], tools=TOOLS),
            model_id=MODEL_ID,
        )
        self.assertEqual(parsed.tools, TOOLS)
        for bad in [
            [{"type": "function", "function": {"name": "x", "extra": 1}}],
            [{"type": "not-function", "function": {"name": "x"}}],
            {"type": "function"},
        ]:
            with self.subTest(bad=bad):
                with self.assertRaises(RequestError):
                    parse_chat_request(
                        self.body([{"role": "user", "content": "u"}], tools=bad),
                        model_id=MODEL_ID,
                    )
        for field in ("tool_choice", "parallel_tool_calls", "stream_options"):
            with self.subTest(field=field):
                with self.assertRaises(RequestError):
                    parse_chat_request(
                        self.body([{"role": "user", "content": "u"}], **{field: True}),
                        model_id=MODEL_ID,
                    )
class CapturingTokenizer:
    def __init__(self):
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return [1, 2, 3]


class PromptRenderingTests(unittest.TestCase):
    def engine(self):
        engine = RuntimeEngine.__new__(RuntimeEngine)
        engine.tokenizer = CapturingTokenizer()
        return engine

    def assistant(self, arguments):
        return {
            "role": "assistant",
            "content": "<think>kept</think>",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": arguments},
            }],
        }

    def test_echoed_arguments_json_string_becomes_dict_before_template(self):
        engine = self.engine()
        original = self.assistant('{"path":"README.md","max_lines":"12"}')
        prepared = engine.prepare([original], tools=TOOLS)
        rendered_args = engine.tokenizer.messages[0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(rendered_args, {"path": "README.md", "max_lines": "12"})
        self.assertIsInstance(original["tool_calls"][0]["function"]["arguments"], str)
        self.assertEqual(engine.tokenizer.kwargs["tools"], TOOLS)
        self.assertEqual(prepared.tools, TOOLS)

    def test_invalid_or_nonobject_arguments_rejected_clearly(self):
        engine = self.engine()
        for arguments in ("{bad", '["not","object"]'):
            with self.subTest(arguments=arguments):
                with self.assertRaises(RequestError) as cm:
                    engine.prepare([self.assistant(arguments)], tools=TOOLS)
                self.assertEqual(cm.exception.code, "invalid_tool_arguments")


if __name__ == "__main__":
    unittest.main()
