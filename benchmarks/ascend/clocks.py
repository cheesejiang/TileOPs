"""Set a site's verified clock policy and independently check both NPU components.

A site supplies argv arrays for its supported lock/restore tool. The benchmark
never guesses a driver debug command. Missing policy or a failed readback aborts
measurement.
"""

import ctypes
import json
import subprocess
from pathlib import Path



class _Frequency(ctypes.Structure):
    _fields_ = [("rated_mhz", ctypes.c_uint), ("current_mhz", ctypes.c_uint)]


class ClockPolicy:
    def __init__(self, config: Path, *, device=0):
        self.device = device
        self.spec = json.loads(config.read_text())
        declared = self.spec.get("device")
        if declared is not None and declared != self.device:
            raise ValueError("Clock policy device does not match the selected physical NPU")
        self.targets = self.spec["component_mhz"]
        if len(self.targets) != 2 or any(type(v) is not int or v <= 0 for v in self.targets):
            raise ValueError("component_mhz must contain two positive integer frequencies")
        for name in ("lock", "restore"):
            argv = self.spec[name]
            if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
                raise ValueError(f"{name} must be an argv array")
            if not Path(argv[0]).is_absolute():
                raise ValueError(f"{name} executable must be an absolute path")
            if self.device != 0 and declared is None and not any("{device}" in v for v in argv):
                raise ValueError("Nonzero NPU requires a device-specific clock policy or {device} argv")
            self.spec[name] = [v.replace("{device}", str(self.device)) for v in argv]
        self.driver = ctypes.CDLL(
            self.spec.get("driver", "/usr/local/Ascend/driver/lib64/driver/libdrvdsmi_host.so")
        )
        self.driver.dsmi_get_aicore_info.argtypes = [ctypes.c_int, ctypes.POINTER(_Frequency)]
        self.driver.dsmi_get_aicore_info.restype = ctypes.c_int
        self.samples = []

    def read(self):
        values = []
        # The 950 exposes the two components in the high 16 bits of the DSMI device ID.
        for device in (0, 1):
            info = _Frequency()
            rc = self.driver.dsmi_get_aicore_info(
                (device << 16) | self.device, ctypes.byref(info)
            )
            if rc:
                raise RuntimeError(f"Cannot query component {device} frequency: {rc}")
            values.append({"rated_mhz": info.rated_mhz, "mhz": info.current_mhz})
        return values

    def check(self):
        values = self.read()
        self.samples.append(values)
        if [v["mhz"] for v in values] != self.targets:
            raise RuntimeError(f"Clock lock readback mismatch: {values}; expected {self.targets}")
        return values

    def __enter__(self):
        self.before = self.read()
        if any(
            target > v["rated_mhz"] for target, v in zip(self.targets, self.before, strict=True)
        ):
            raise ValueError("A benchmark clock must not exceed rated frequency")
        try:
            subprocess.run(self.spec["lock"], check=True, timeout=30)
            self.check()
        except BaseException:
            subprocess.run(self.spec["restore"], check=True, timeout=30)
            raise
        return self

    def __exit__(self, *_):
        subprocess.run(self.spec["restore"], check=True, timeout=30)
        self.after = self.read()
