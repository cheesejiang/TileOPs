"""CPU checks of single-device event timing and CLI validation."""

import contextlib
import importlib.util
import io
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "ascend" / "paged_cache_write.py"
SPEC = importlib.util.spec_from_file_location("single_device_bench", SOURCE)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class TimerTests(unittest.TestCase):
    def fake_torch(self, samples):
        log = []
        durations = iter(samples)

        class Buffer:
            def zero_(self):
                log.append("flush")

            def numel(self):
                return 1 << 20

        class Event:
            def __init__(self, **kwargs):
                self.name = "start" if not any(x == "event" for x in log) else "end"
                log.append("event")

            def record(self):
                log.append(self.name)

            def synchronize(self):
                log.append("event-sync")

            def elapsed_time(self, other):
                return next(durations)

        @contextlib.contextmanager
        def device(selected):
            log.append(("device", selected))
            yield

        def empty(size, *, dtype, device):
            log.append(("allocate", device))
            return Buffer()

        torch = types.SimpleNamespace(
            empty=empty,
            uint8="uint8",
            npu=types.SimpleNamespace(
                Event=Event,
                device=device,
                synchronize=lambda selected: log.append(("sync", selected)),
            ),
        )
        return torch, log

    def test_selected_device_reset_and_flush_are_outside_event_window(self):
        torch, log = self.fake_torch([2.0, 4.0])
        with patch.dict("sys.modules", {"torch": torch}):
            timer = BENCH.SingleDeviceEventTimer("npu:3", warmup=1, repeats=2, flush_mb=1)
            result = timer.measure(lambda: log.append("run"), reset=lambda: log.append("reset"))
        self.assertEqual(result["latency_ms"], 3.0)
        self.assertEqual(result["samples_ms"], [2.0, 4.0])
        self.assertEqual(result["clock_policy"], "unlocked")
        self.assertEqual(log.count(("sync", "npu:3")), 4)
        self.assertIn(("allocate", "npu:3"), log)
        self.assertIn(("device", "npu:3"), log)
        for index, item in enumerate(log):
            if item == "start" and log[index + 1] == "run":
                self.assertEqual(log[index - 3:index], ["reset", "flush", ("sync", "npu:3")])
                self.assertEqual(log[index:index + 4], ["start", "run", "end", "event-sync"])

    def test_no_flush_and_single_sample(self):
        torch, log = self.fake_torch([1.0])
        with patch.dict("sys.modules", {"torch": torch}):
            result = BENCH.SingleDeviceEventTimer(
                "npu:2", warmup=0, repeats=1, flush_mb=0
            ).measure(lambda: None)
        self.assertEqual(result["flush_bytes"], 0)
        self.assertEqual(result["stdev_ms"], 0)
        self.assertNotIn("flush", log)

    def test_invalid_samples_fail(self):
        for sample in (0, -1, float("nan"), float("inf")):
            with self.subTest(sample=sample):
                torch, _ = self.fake_torch([sample])
                with patch.dict("sys.modules", {"torch": torch}):
                    timer = BENCH.SingleDeviceEventTimer("npu:3", 0, 1, 0)
                    with self.assertRaises(RuntimeError):
                        timer.measure(lambda: None)

    def test_invalid_cli_before_runtime_import(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            BENCH.main(["--device", "-1", "--run-dir", "/unused"])
        self.assertEqual(raised.exception.code, 2)


class ReportTests(unittest.TestCase):
    def metadata(self, verify_only=False):
        return {
            "device": "npu:3", "device_name": "Ascend950",
            "torch": "test", "torch_npu": "test", "tilelang": "test",
            "protocol": "ascend-single-device-unlocked-events-v1",
            "arguments": {"verify_only": verify_only, "warmup": 3,
                          "repeats": 15, "flush_mb": 1024},
        }

    def render(self, rows, verify_only=False, complete=True):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            BENCH.write_summary(path, self.metadata(verify_only), rows, complete=complete)
            self.assertFalse((path / "summary.md.tmp").exists())
            return (path / "summary.md").read_text(encoding="utf-8")

    def test_measured_units_and_speedup(self):
        check = {"checked_outputs": 2, "total_outputs": 2}
        report = self.render([{
            "case_id": "decode|bf16", "status": "measured",
            "implementations": {
                "tileops": {"correctness": check, "measurement": {
                    "latency_ms": 0.002, "median_ms": 0.0015,
                }, "speedup_vs_generic": 2.0},
                "generic": {"correctness": check, "measurement": {
                    "latency_ms": 0.004, "median_ms": 0.003,
                }, "speedup_vs_generic": 1.0},
            },
        }])
        self.assertIn("通过（2/2 输出） | 2.000 | 1.500 | 2.000×", report)
        self.assertIn("decode&#124;bf16", report)
        self.assertIn("**2.000×**", report)
        self.assertIn("设备：npu:3", report)

    def test_verify_only_has_no_performance_numbers(self):
        report = self.render([{
            "case_id": "odd", "status": "verified",
            "implementations": {"tileops": {
                "correctness": {"checked_outputs": 2, "total_outputs": 2},
            }},
        }], verify_only=True)
        self.assertIn("通过（2/2 输出） | — | — | —", report)
        self.assertNotIn("加速比几何均值", report)

    def test_failure_and_partial_run_are_not_reported_as_passed(self):
        rows = [{"case_id": "broken", "status": "failed", "error": "RuntimeError: compile"}]
        report = self.render(rows)
        self.assertIn("状态：失败", report)
        self.assertIn("未完成，见失败详情", report)
        self.assertIn("    RuntimeError: compile", report)
        self.assertIn("状态：进行中", self.render(rows, complete=False))


if __name__ == "__main__":
    unittest.main()
