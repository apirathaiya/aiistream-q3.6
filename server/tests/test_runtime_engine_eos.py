import threading
import unittest
from unittest.mock import patch

import mlx.core as mx

import aiistream_server.runtime_engine as runtime_engine
from aiistream_server.runtime_engine import PreparedPrompt, RuntimeEngine


class FakeDetokenizer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.text = ""
        self.last_segment = ""

    def add_token(self, token):
        piece = {101: "A", 102: "B", 999: "<EOS>"}[int(token)]
        self.last_segment = piece
        self.text += piece

    def finalize(self):
        self.last_segment = ""


class FakeTokenizer:
    eos_token_ids = [999]
    all_special_tokens = ["<EOS>"]

    def __init__(self):
        self.detokenizer = FakeDetokenizer()

    def decode(self, tokens):
        return "".join({101: "A", 102: "B", 999: "<EOS>"}[int(t)] for t in tokens)


class FakeModel:
    def make_cache(self):
        return []
class FakeDetector:
    instances = []

    def __init__(self, *args, **kwargs):
        self.reset_calls = []
        self.observe_calls = []
        self.finished = False
        type(self).instances.append(self)

    def reset(self, start_time=None):
        self.reset_calls.append(start_time)

    def observe(self, rate, now):
        self.observe_calls.append((rate, now))
        return False

    def finish(self):
        self.finished = True


def make_engine():
    engine = RuntimeEngine.__new__(RuntimeEngine)
    engine.model = FakeModel()
    engine.tokenizer = FakeTokenizer()
    engine.safety = object()
    engine.collapse_fraction = 0.5
    engine.collapse_window_seconds = 60.0
    engine.capture_last_result = False
    engine.last_result = None
    engine._last_lock = threading.Lock()
    return engine


def fake_generate(tokens):
    def _runner(*args, **kwargs):
        for token in tokens:
            yield mx.array(token), mx.array([0.0])
    return _runner


class RuntimeEngineEOSTests(unittest.TestCase):
    def setUp(self):
        FakeDetector.instances.clear()
    def test_eos_first_is_not_content_and_ttft_is_sane(self):
        engine = make_engine()
        streamed = []
        with (
            patch.object(runtime_engine, "generate_step", fake_generate([999])),
            patch.object(runtime_engine, "make_sampler", return_value=lambda x: x),
            patch.object(runtime_engine, "ThroughputCollapseDetector", FakeDetector),
            patch.object(runtime_engine.time, "monotonic", side_effect=[0.0, 0.25, 0.30]),
        ):
            result = engine.generate_prepared(
                PreparedPrompt(messages=[], prompt_ids=[1]),
                temperature=0.0,
                seed=None,
                max_completion_tokens=4,
                on_text=streamed.append,
            )

        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.token_ids, [])
        self.assertEqual(result.completion_tokens, 0)
        self.assertEqual(result.text, "")
        self.assertEqual(streamed, [])
        self.assertAlmostEqual(result.ttft_s, 0.25)
        detector = FakeDetector.instances[-1]
        self.assertEqual(detector.observe_calls, [])
        self.assertTrue(detector.finished)

    def test_eos_after_content_does_not_enter_throughput_window(self):
        engine = make_engine()
        streamed = []
        with (
            patch.object(runtime_engine, "generate_step", fake_generate([101, 999])),
            patch.object(runtime_engine, "make_sampler", return_value=lambda x: x),
            patch.object(runtime_engine, "ThroughputCollapseDetector", FakeDetector),
            patch.object(
                runtime_engine.time,
                "monotonic",
                side_effect=[0.0, 0.10, 11.20, 11.30],
            ),
        ):
            result = engine.generate_prepared(
                PreparedPrompt(messages=[], prompt_ids=[1]),
                temperature=0.0,
                seed=None,
                max_completion_tokens=4,
                on_text=streamed.append,
            )

        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.token_ids, [101])
        self.assertEqual(result.completion_tokens, 1)
        self.assertEqual(result.text, "A")
        self.assertEqual(streamed, ["A"])
        detector = FakeDetector.instances[-1]
        self.assertEqual(detector.observe_calls, [])
        self.assertEqual(detector.reset_calls, [0.10])

    def test_max_token_path_remains_length(self):
        engine = make_engine()
        with (
            patch.object(runtime_engine, "generate_step", fake_generate([101, 102])),
            patch.object(runtime_engine, "make_sampler", return_value=lambda x: x),
            patch.object(runtime_engine, "ThroughputCollapseDetector", FakeDetector),
            patch.object(
                runtime_engine.time,
                "monotonic",
                side_effect=[0.0, 0.10, 0.20, 0.30],
            ),
        ):
            result = engine.generate_prepared(
                PreparedPrompt(messages=[], prompt_ids=[1]),
                temperature=0.0,
                seed=None,
                max_completion_tokens=2,
            )

        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.token_ids, [101, 102])
        self.assertEqual(result.completion_tokens, 2)
        self.assertEqual(result.text, "AB")

    def test_should_stop_cancels_after_current_token(self):
        engine = make_engine()
        polls = []
        def should_stop():
            polls.append(1)
            return True
        with (
            patch.object(runtime_engine, "generate_step", fake_generate([101, 102])),
            patch.object(runtime_engine, "make_sampler", return_value=lambda x: x),
            patch.object(runtime_engine, "ThroughputCollapseDetector", FakeDetector),
            patch.object(runtime_engine.time, "monotonic", side_effect=[0.0, 0.10, 0.20, 0.30]),
        ):
            result = engine.generate_prepared(
                PreparedPrompt(messages=[], prompt_ids=[1]),
                temperature=0.0,
                seed=None,
                max_completion_tokens=2,
                should_stop=should_stop,
            )

        self.assertEqual(result.finish_reason, "cancelled")
        self.assertEqual(result.token_ids, [101])
        self.assertEqual(len(polls), 1)

    def test_should_stop_false_leaves_output_unchanged(self):
        engine = make_engine()
        with (
            patch.object(runtime_engine, "generate_step", fake_generate([101, 102])),
            patch.object(runtime_engine, "make_sampler", return_value=lambda x: x),
            patch.object(runtime_engine, "ThroughputCollapseDetector", FakeDetector),
            patch.object(runtime_engine.time, "monotonic", side_effect=[0.0, 0.10, 0.20, 0.30]),
        ):
            result = engine.generate_prepared(
                PreparedPrompt(messages=[], prompt_ids=[1]),
                temperature=0.0,
                seed=None,
                max_completion_tokens=2,
                should_stop=lambda: False,
            )

        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.token_ids, [101, 102])


if __name__ == "__main__":
    unittest.main()
