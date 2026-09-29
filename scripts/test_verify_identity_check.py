"""Unit test for verify_identity.check(): it must fail when coverage or identity is missing."""
import copy, sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cases import CASES, EXPECT_TOOL_CALL  # noqa: E402
from verify_identity import MODES, check  # noqa: E402


def good():
    golden, out = {}, {m: {} for m in MODES}
    for i, (case, *_rest) in enumerate(CASES):
        ids = [i, i + 1]
        golden[case] = {"token_ids": ids, "text_sha256": f"h{i}"}
        for m in MODES:
            out[m][case] = {"token_ids": ids, "text_sha256": f"h{i}", "service_eq_mlx_lm": True,
                            "http_tool_calls": case in EXPECT_TOOL_CALL, "completion_tokens": 2}
    return out, golden


class CheckTests(unittest.TestCase):
    def test_good_passes(self):
        self.assertEqual(check(*good()), [])

    def test_all_identical_and_no_tool_calls_fails(self):
        out, golden = good()
        for m in MODES:
            for r in out[m].values():
                r["token_ids"] = [1]; r["text_sha256"] = "same"; r["http_tool_calls"] = False
        for g in golden.values():
            g["token_ids"] = [1]; g["text_sha256"] = "same"
        fails = check(out, golden)
        self.assertTrue(any("sampling path" in f for f in fails))
        self.assertTrue(any("no tool call" in f for f in fails))

    def test_differs_from_stock_fails(self):
        out, golden = good()
        golden = copy.deepcopy(golden); golden[CASES[0][0]]["token_ids"] = [99]
        self.assertTrue(any("unmodified stock model" in f for f in check(out, golden)))

    def test_missing_mode_fails(self):
        out, golden = good(); del out["direct"]
        self.assertTrue(any("direct: no results" in f for f in check(out, golden)))


if __name__ == "__main__":
    unittest.main()
