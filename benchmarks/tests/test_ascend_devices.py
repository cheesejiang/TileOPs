"""Physical-device routing checks without importing an accelerator runtime."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.ascend.clocks import ClockPolicy, _Frequency


def load_guard():
    path = Path(__file__).resolve().parents[2] / "scripts/npu_guard.py"
    spec = importlib.util.spec_from_file_location("device_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PhysicalDeviceTest(unittest.TestCase):
    def test_fd_scan_ignores_explicit_other_cards(self):
        guard = load_guard()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            nodes = {
                900001: ["/dev/davinci0", "/dev/davinci_manager"],
                900002: ["/dev/davinci3", "/dev/davinci_manager"],
                900003: ["/dev/devmm_svm"],
            }
            for pid, paths in nodes.items():
                folder = root / str(pid) / "fd"
                folder.mkdir(parents=True)
                for i, path in enumerate(paths):
                    (folder / str(i)).symlink_to(path)
            for device, expected in ((0, {900001, 900003}), (3, {900002, 900003})):
                candidates, unreadable = guard.scan_device_fds(root, device)
                self.assertEqual({p["pid"] for p in candidates}, expected)
                self.assertEqual(unreadable, [])

    def test_probe_subprocess_receives_selected_device(self):
        guard = load_guard()
        sample = {"state": "idle", "reasons": [], "device": 3}
        with patch.object(guard.subprocess, "run", return_value=SimpleNamespace(
            stdout=json.dumps(sample)
        )) as run:
            self.assertEqual(guard.probe(3), sample)
            self.assertEqual(run.call_args.args[0][-3:], ["_probe", "--device", "3"])

    def test_child_exposes_only_the_reserved_physical_card(self):
        guard = load_guard()
        child = SimpleNamespace(pid=900004, poll=lambda: 0)
        with (
            patch.object(guard.subprocess, "Popen", return_value=child) as spawn,
            patch.object(guard, "write_metadata"),
            patch.object(guard, "group_alive", return_value=False),
        ):
            self.assertEqual(guard.execute(-1, {"device": 3}, ["unused"], 1), 0)
            env = spawn.call_args.kwargs["env"]
            for name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "TILEOPS_NPU_DEVICE"):
                self.assertEqual(env[name], "3")

    def test_reservation_uses_selected_lock_and_rejects_device_mismatch(self):
        from benchmarks.ascend import guard

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "device-3.lock").write_text(json.dumps({"device": 0}))
            with (
                patch.object(guard, "LOCK_PATH", root / "device-0.lock"),
                patch.dict(guard.os.environ, {"TILEOPS_NPU_DEVICE": "3"}),
                patch.object(guard.fcntl, "flock", side_effect=BlockingIOError),
            ):
                with self.assertRaisesRegex(RuntimeError, "device does not match"):
                    guard.require_reservation()

    def test_clock_readback_uses_physical_card_for_both_components(self):
        ids = []

        def read(device, pointer):
            ids.append(device)
            info = pointer._obj
            self.assertIsInstance(info, _Frequency)
            info.rated_mhz = 1200
            info.current_mhz = 1000
            return 0

        policy = ClockPolicy.__new__(ClockPolicy)
        policy.device = 3
        policy.driver = SimpleNamespace(dsmi_get_aicore_info=read)
        self.assertEqual([p["mhz"] for p in policy.read()], [1000, 1000])
        self.assertEqual(ids, [3, (1 << 16) | 3])

    def test_nonzero_card_rejects_an_unscoped_clock_policy(self):
        with tempfile.TemporaryDirectory() as root:
            config = Path(root) / "clocks.json"
            config.write_text(json.dumps({
                "component_mhz": [1000, 1000],
                "lock": ["/site/lock"], "restore": ["/site/restore"],
            }))
            with self.assertRaisesRegex(ValueError, "Nonzero NPU"):
                ClockPolicy(config, device=3)


if __name__ == "__main__":
    unittest.main()
