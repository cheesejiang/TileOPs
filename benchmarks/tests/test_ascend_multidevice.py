"""Multi-card development routing without importing an NPU runtime."""

import importlib.util
import contextlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MultiDeviceTest(unittest.TestCase):
    def test_launcher_routes_physical_card_and_unlocked_mode(self):
        launcher = load("launcher_multi", "scripts/benchmark_ascend.py")
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "result"
            argv = ["benchmark_ascend.py", "--device", "3", "--owner", "test",
                    "--run-dir", str(run), "--op", "PagedKVCacheGatherFwdOp", "--unlocked"]

            def completed(command, **kwargs):
                split = command.index("--")
                self.assertEqual(command[command.index("--device") + 1], "3")
                self.assertLess(command.index("--device"), split)
                self.assertGreater(command.index("--unlocked"), split)
                (run / "result.json").write_text(json.dumps({"status": "passed"}))
                return SimpleNamespace(returncode=0)

            with patch.object(sys, "argv", argv), patch.object(launcher.subprocess, "run", completed):
                self.assertEqual(launcher.main(), 0)

    def test_fd_scan_ignores_other_cards_but_keeps_target(self):
        guard = load("guard_multi", "scripts/npu_guard.py")
        guard.DEVICE, guard.PER_DEVICE = 3, True
        with tempfile.TemporaryDirectory() as temp:
            proc = Path(temp)
            for pid, node in [(101, "/dev/davinci2"), (102, "/dev/davinci_manager"),
                              (103, "/dev/davinci3")]:
                fd = proc / str(pid) / "fd"
                fd.mkdir(parents=True)
                (fd / "0").symlink_to(node)
            candidates, unreadable = guard.scan_device_fds(proc)
            self.assertEqual([c["pid"] for c in candidates], [103])
            self.assertEqual(unreadable, [])
            guard.PER_DEVICE = False
            self.assertEqual(len(guard.scan_device_fds(proc)[0]), 3)

    def test_disposable_probe_receives_selected_card(self):
        guard = load("guard_probe_multi", "scripts/npu_guard.py")
        guard.DEVICE, guard.PER_DEVICE = 5, True
        completed = SimpleNamespace(stdout=json.dumps({"state": "idle", "reasons": []}))
        with patch.object(guard.subprocess, "run", return_value=completed) as run:
            self.assertEqual(guard.probe()["state"], "idle")
        self.assertEqual(run.call_args.args[0][-2:], ["--device", "5"])

    def test_clock_readback_uses_selected_physical_card(self):
        clocks = load("clocks_multi", "benchmarks/ascend/clocks.py")
        policy = object.__new__(clocks.ClockPolicy)
        calls = []

        def query(device, pointer):
            calls.append(device)
            return 0

        policy.driver = SimpleNamespace(dsmi_get_aicore_info=query)
        with patch.dict(os.environ, {"TILEOPS_PHYSICAL_DEVICE": "3"}):
            policy.read()
        self.assertEqual(calls, [3, (1 << 16) | 3])

    def test_unlocked_policy_never_runs_clock_commands(self):
        clocks = load("clocks_unlocked", "benchmarks/ascend/clocks.py")
        with patch.object(clocks, "require_reservation"), patch.object(clocks.subprocess, "run") as run:
            with clocks.UnlockedPolicy() as policy:
                policy.check()
            run.assert_not_called()
        self.assertTrue(policy.unlocked)
        self.assertEqual(policy.samples, [{"locked": False, "frequency_verified": False}])

    def test_unlocked_measurement_cannot_claim_formal_protocol(self):
        timing = load("timing_unlocked", "benchmarks/ascend/timing.py")
        timer = object.__new__(timing.ProfilerTimer)
        timer.directory = Path("unused")
        timer.clocks = SimpleNamespace(unlocked=True, check=lambda: None)
        timer._prepare = lambda reset: None
        prof = SimpleNamespace(prof_if=SimpleNamespace(prof_path="unused"), step=lambda: None)
        parsed = {"profile_excluded_kernel_count": 0, "profile_kernel_count": 1,
                  "latency_ms": 0.01}
        torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
        with (patch.dict(sys.modules, {"torch": torch}),
              patch.object(timing, "require_reservation"),
              patch.object(timing, "_profile", side_effect=lambda path: contextlib.nullcontext(prof)),
              patch.object(timing, "_parse_ffts_profile", return_value=parsed)):
            result = timer.measure(lambda: None)
        self.assertEqual(result["protocol"], "ascend-profiler-unlocked-v1")
        self.assertFalse(result["formal"])
        self.assertEqual(result["repeats"], 15)
        self.assertEqual(len(result["samples_ms"]), 15)


if __name__ == "__main__":
    unittest.main()
