"""Launcher routing tests: standard library only, no runtime imports or device work."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class LauncherRoutingTest(unittest.TestCase):
    def test_site_guard_wraps_the_complete_runtime_command(self):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "ascend_launcher", root / "scripts/benchmark_gather_final.py"
        )
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            guard = temp / "host_guard.py"
            guard.write_text("# Site guard routing fixture; never executed.\n")
            run = temp / "run"
            argv = [
                "benchmark_gather_final.py",
                "--run-dir",
                str(run),
                "--owner",
                "test/site-guard",
                "--guard",
                str(guard),
                "--verify-only",
                "--op",
                "PagedKVCacheGatherFwdOp",
            ]

            def completed(command, **kwargs):
                self.assertEqual(command[:3], [sys.executable, str(guard), "run"])
                self.assertIn("benchmarks.ascend_gather_final.runner", command)
                self.assertIn("--verify-only", command)
                self.assertEqual(command[-2:], ["--op", "PagedKVCacheGatherFwdOp"])
                self.assertIn(str(root / "src"), kwargs["env"]["PYTHONPATH"])
                (run / "result.json").write_text(json.dumps({"status": "passed"}))
                row = {
                    "op": "PagedKVCacheGatherFwdOp", "case_id": "fp8-case",
                    "status": "measured",
                    "params": {"dst": [[1000, 576], "bfloat16"],
                               "cache": [[16, 64, 576], "float8_e4m3fn"]},
                    "implementations": {"tileops": {
                        "correctness": {"max_abs_err": 0.0},
                        "measurement": {"latency_ms": 0.01, "stdev_ms": 0.001,
                                        "protocol": "ascend-profiler-unlocked-v1"}}},
                }
                (run / "results.jsonl").write_text(json.dumps(row) + "\n")
                return SimpleNamespace(returncode=0)

            with (
                patch.object(sys, "argv", argv),
                patch.object(launcher.subprocess, "run", completed),
            ):
                self.assertEqual(launcher.main(), 0)
            receipt = json.loads((run / "receipt.json").read_text())
            self.assertEqual(receipt["guard_exit_code"], 0)
            self.assertEqual(receipt["command"][1], str(guard))
            report = (run / "summary.md").read_text()
            self.assertIn("fp8-case | tileops | measured | 10.000 | 1.000 | 172.8 | 0.0", report)

    def test_summary_handles_failed_and_verify_only_runs(self):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("ascend_report", root / "scripts/benchmark_gather_final.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            rows = [{"op": "Gather", "case_id": "check", "status": "verified",
                     "implementations": {"tileops": {"correctness": {"max_abs_err": 0}}}},
                    {"op": "Gather", "case_id": "bad|case", "status": "failed"}]
            (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n{broken")
            launcher.write_summary(run, {"status": "failed", "guard_exit_code": 1})
            report = (run / "summary.md").read_text()
            self.assertIn("不能作为有效性能结果", report)
            self.assertIn("仅正确性", report)
            self.assertIn("bad\\|case", report)
            self.assertIn("无法解析", report)

    def test_summary_filters_implementations_without_changing_raw_results(self):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("ascend_filtered_report", root / "scripts/benchmark_gather_final.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            raw = json.dumps({"op": "Gather", "case_id": "sample", "status": "measured",
                              "implementations": {name: {} for name in
                                                  ("tileops", "tilelang-auto1", "tilelang-batch2-db")}})
            (run / "results.jsonl").write_text(raw)
            launcher.write_summary(run, {"status": "passed", "guard_exit_code": 0},
                                   ["tileops", "tilelang-batch2-db"])
            report = (run / "summary.md").read_text()
            self.assertIn("| tileops |", report)
            self.assertIn("| tilelang-batch2-db |", report)
            self.assertNotIn("tilelang-auto1", report)
            self.assertEqual((run / "results.jsonl").read_text(), raw)


if __name__ == "__main__":
    unittest.main()
