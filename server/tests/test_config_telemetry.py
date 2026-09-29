"""Config, hash-pin and telemetry checks (no model load)."""
import json
import tempfile
import unittest
from pathlib import Path
from aiistream_server.config import ConfigError, ServiceConfig, load_config
from aiistream_server.hashpin import RuntimeHashError, verify_runtime_imports, PINNED_RUNTIME_SHA256
from aiistream_server.runtime_engine import make_nonretaining_telemetry

ROOT=Path(__file__).resolve().parents[2]
EXPERIMENTS=ROOT/"engine"

class ConfigTelemetryChecks(unittest.TestCase):
    def test_both_modes_and_default_and_invalid(self):
        self.assertEqual(ServiceConfig().validate().expert_read_path,"prefetch")
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"cfg.json"
            for mode in ("parallel","direct","prefetch"):
                f.write_text(json.dumps({"expert_read_path":mode}))
                self.assertEqual(load_config(f).expert_read_path,mode)
            f.write_text(json.dumps({"expert_read_path":"other"}))
            with self.assertRaisesRegex(ConfigError,"expert_read_path"):
                load_config(f)

    def test_hash_mode_pins_and_negative_control(self):
        p=verify_runtime_imports(EXPERIMENTS,"parallel")
        self.assertEqual(set(p),{"aiistream_model.py","aiistream_parallel.py"})
        z=verify_runtime_imports(EXPERIMENTS,"direct")
        self.assertEqual(set(z),{"aiistream_model.py","aiistream_parallel.py","aiistream_direct.py"})
        self.assertEqual(z["aiistream_direct.py"],PINNED_RUNTIME_SHA256["aiistream_direct.py"])
        with self.assertRaises(RuntimeHashError):
            verify_runtime_imports(EXPERIMENTS,"parallel",p3_expected="0"*64)
        with tempfile.TemporaryDirectory() as d:
            for name in z:
                (Path(d)/name).write_text((EXPERIMENTS/name).read_text())
            (Path(d)/"aiistream_direct.py").write_text("corrupt\n")
            with self.assertRaises(RuntimeHashError):
                verify_runtime_imports(d,"direct")

    def test_10000_records_do_not_retain_routes_or_lose_counters(self):
        import sys
        sys.path.insert(0,str(EXPERIMENTS))
        import aiistream_model as p1
        tel=make_nonretaining_telemetry(p1)
        for i in range(10000):
            tel.record(i%40,128,0.002,[i%256,(i+1)%256])
        self.assertEqual(sum(len(x) for x in tel.route_sets.values()),0)
        self.assertEqual(tel.bytes,1_280_000)
        self.assertEqual(sum(tel.layer_calls.values()),10000)
        self.assertEqual(tel.short_reads,0)
        tel.short_reads+=1
        self.assertEqual(tel.short_reads,1)
        self.assertAlmostEqual(tel.io_s,20.0,places=7)
