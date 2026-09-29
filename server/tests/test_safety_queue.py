import unittest

from aiistream_server.safety import AdmissionController, SafetyState, ThroughputCollapseDetector


class SafetyQueueTests(unittest.TestCase):
    def test_memory_critical_gates_new_admission_not_inflight(self):
        safety = SafetyState()
        ctl = AdmissionController(queue_limit=4, safety=safety)
        running, reason = ctl.try_admit()
        self.assertIsNone(reason)
        running.begin_generation()
        safety.set_memory_pressure_level(4)
        blocked, reason = ctl.try_admit()
        self.assertIsNone(blocked)
        self.assertEqual(reason, "memory_pressure_critical")
        self.assertTrue(ctl.snapshot()["active_generation"])
        safety.set_memory_pressure_level(1)
        allowed, reason = ctl.try_admit()
        self.assertIsNone(reason)
        allowed.release()
        running.release()

    def test_queue_n_plus_one_rejected_before_tokenization(self):
        safety = SafetyState()
        ctl = AdmissionController(queue_limit=4, safety=safety)
        tickets = []
        tokenizations = 0
        for _ in range(5):
            t, reason = ctl.try_admit()
            self.assertIsNone(reason)
            tickets.append(t)
        sixth, reason = ctl.try_admit()
        if sixth is not None:
            tokenizations += 1
        self.assertIsNone(sixth)
        self.assertEqual(reason, "queue_full")
        self.assertEqual(tokenizations, 0)
        for t in tickets:
            t.release()
    def test_throughput_collapse_requires_full_60_seconds(self):
        safety = SafetyState()
        detector = ThroughputCollapseDetector(
            safety, fraction=0.5, collapse_seconds=60.0, early_seconds=60.0
        )
        detector.reset(start_time=0.0)
        for t in (10, 20, 30, 40, 50, 60):
            self.assertFalse(detector.observe(10.0, float(t)))
        self.assertFalse(detector.observe(4.0, 70.0))
        self.assertFalse(detector.observe(4.0, 129.9))
        self.assertTrue(detector.observe(4.0, 130.0))
        self.assertTrue(safety.snapshot()["throughput_collapse"])
        self.assertFalse(detector.observe(6.0, 140.0))
        self.assertFalse(safety.snapshot()["throughput_collapse"])

    def test_throughput_stop_never_invalidates_existing_ticket(self):
        safety = SafetyState()
        ctl = AdmissionController(queue_limit=1, safety=safety)
        running, _ = ctl.try_admit()
        running.begin_generation()
        safety.set_throughput_collapse(True, {"test": True})
        blocked, reason = ctl.try_admit()
        self.assertIsNone(blocked)
        self.assertEqual(reason, "throughput_collapse")
        self.assertTrue(ctl.snapshot()["active_generation"])
        running.release()


if __name__ == "__main__":
    unittest.main()
