#!/usr/bin/env python3
"""Run benchmarks on one selected physical NPU without acquiring a reservation."""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="new directory; existing output is never reused"
    )
    parser.add_argument("--owner", default=os.environ.get("USER", "benchmark"))
    parser.add_argument("--device", type=int, default=0, help="Physical NPU ID (default 0)")
    parser.add_argument("--timeout", type=int, default=1800)
    args, rest = parser.parse_known_args()
    if "--help" in rest:
        return 0
    if not 0 <= args.device < 65536:
        parser.error("--device must be in 0..65535")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    env.update(
        ASCEND_RT_VISIBLE_DEVICES=str(args.device),
        ASCEND_VISIBLE_DEVICES=str(args.device),
        TILEOPS_NPU_DEVICE=str(args.device),
    )
    command = [
        sys.executable,
        "-m",
        "benchmarks.ascend.runner",
        "--run-dir",
        str(run),
        *rest,
    ]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (run / "run.log").open("x") as log:
        try:
            result = subprocess.run(
                command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
    child = run / "result.json"
    summary = json.loads(child.read_text()) if child.exists() else {}
    success = exit_code == 0 and summary.get("status") == "passed"
    receipt = {
        "status": "passed" if success else "failed",
        "exit_code": exit_code,
        "exclusive": False,
        "owner": args.owner,
        "started_at": started,
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": command,
        "physical_device": args.device,
        "result": summary,
    }
    (run / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"run_dir": str(run), **receipt}, indent=2))
    return 0 if success else (exit_code or 1)


if __name__ == "__main__":
    sys.exit(main())
