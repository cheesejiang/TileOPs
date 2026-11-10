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
    def test_selected_card_runs_directly_without_guard(self):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "ascend_launcher", root / "scripts/benchmark_ascend.py"
        )
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            run = temp / "run"
            argv = [
                "benchmark_ascend.py",
                "--run-dir",
                str(run),
                "--owner",
                "test/site-guard",
                "--device",
                "3",
                "--verify-only",
                "--op",
                "all",
            ]

            def completed(command, **kwargs):
                self.assertEqual(command[:3], [sys.executable, "-m", "benchmarks.ascend.runner"])
                self.assertNotIn("npu_guard.py", " ".join(command))
                for name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "TILEOPS_NPU_DEVICE"):
                    self.assertEqual(kwargs["env"][name], "3")
                self.assertEqual(kwargs["timeout"], 1800)
                self.assertIn("benchmarks.ascend.runner", command)
                self.assertIn("--verify-only", command)
                self.assertEqual(command[-2:], ["--op", "all"])
                self.assertIn(str(root / "src"), kwargs["env"]["PYTHONPATH"])
                (run / "result.json").write_text(json.dumps({"status": "passed"}))
                return SimpleNamespace(returncode=0)

            with (
                patch.object(sys, "argv", argv),
                patch.object(launcher.subprocess, "run", completed),
            ):
                self.assertEqual(launcher.main(), 0)
            receipt = json.loads((run / "receipt.json").read_text())
            self.assertEqual(receipt["exit_code"], 0)
            self.assertFalse(receipt["exclusive"])
            self.assertEqual(receipt["physical_device"], 3)
            self.assertEqual(receipt["command"][1], "-m")

    def test_timeout_writes_nonexclusive_failure_receipt(self):
        from scripts import benchmark_ascend as launcher

        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            argv = [
                "benchmark_ascend.py", "--run-dir", str(run), "--device", "1",
                "--timeout", "5", "--op", "PagedKVCacheGatherFwdOp", "--unlocked",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(launcher.subprocess, "run", side_effect=launcher.subprocess.TimeoutExpired(
                    "benchmark", 5
                )),
            ):
                self.assertEqual(launcher.main(), 124)
            receipt = json.loads((run / "receipt.json").read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["exit_code"], 124)
            self.assertFalse(receipt["exclusive"])


if __name__ == "__main__":
    unittest.main()
