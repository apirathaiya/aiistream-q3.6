import json
import tempfile
import unittest
from pathlib import Path

from aiistream_server.config import ConfigError, load_config
from aiistream_server.hashpin import RuntimeHashError, verify_runtime_hash
from aiistream_server.policy import LONG_CONTEXT_WARNING, RequestError, check_context_policy
from aiistream_server.runtime_engine import RuntimeEngine
from aiistream_server.safety import SafetyState


class ConfigHashPolicyTests(unittest.TestCase):
    def test_config_validates_before_model_load(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.json"
            p.write_text(json.dumps({"port": 8080}))
            with self.assertRaisesRegex(ConfigError, "reserved"):
                load_config(p)

    def test_hard_ceiling_cannot_exceed_measured_65536(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.json"
            p.write_text(json.dumps({"max_context_hard_ceiling": 65537}))
            with self.assertRaises(ConfigError):
                load_config(p)

    def test_wrong_runtime_hash_refuses(self):
        repo_root = Path(__file__).resolve().parents[2]
        with self.assertRaisesRegex(RuntimeHashError, "runtime hash mismatch"):
            RuntimeEngine(
                repo_root=repo_root,
                safety=SafetyState(),
                expected_runtime_sha="0" * 64,
            )
    def test_context_default_override_and_hard_ceiling(self):
        ok = check_context_policy(
            prompt_tokens=32768, max_completion_tokens=16,
            default_max=32768, hard_ceiling=65536,
            allow_long_context=False, model_max=262144,
        )
        self.assertIsNone(ok.warning)
        with self.assertRaisesRegex(RequestError, "X-Qwen36-Allow-Long-Context"):
            check_context_policy(
                prompt_tokens=32769, max_completion_tokens=16,
                default_max=32768, hard_ceiling=65536,
                allow_long_context=False, model_max=262144,
            )
        override = check_context_policy(
            prompt_tokens=65536, max_completion_tokens=16,
            default_max=32768, hard_ceiling=65536,
            allow_long_context=True, model_max=262144,
        )
        self.assertEqual(override.warning, LONG_CONTEXT_WARNING)
        with self.assertRaisesRegex(RequestError, "hard ceiling"):
            check_context_policy(
                prompt_tokens=65537, max_completion_tokens=16,
                default_max=32768, hard_ceiling=65536,
                allow_long_context=True, model_max=262144,
            )


if __name__ == "__main__":
    unittest.main()
