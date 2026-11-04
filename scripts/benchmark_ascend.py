#!/usr/bin/env python3
"""Team benchmark entry point: reserve the host NPU before importing its runtime."""

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _gather_bandwidth(row, latency_ms):
    """Logical cache-read + destination-write bytes; not measured HBM traffic."""
    if row.get("op") != "PagedKVCacheGatherFwdOp" or not latency_ms or latency_ms <= 0:
        return "—"
    sizes = {"float16": 2, "bfloat16": 2, "float8_e4m3fn": 1}
    try:
        dst, cache = row["params"]["dst"], row["params"]["cache"]
        traffic = math.prod(dst[0]) * (sizes[dst[1]] + sizes[cache[1]])
    except (KeyError, TypeError, IndexError):
        return "—"
    return f"{traffic / latency_ms / 1e6:.1f}"


def write_summary(run, receipt):
    """Generate a CPU-only report after the final guard receipt is known."""
    valid = receipt["status"] == "passed"
    lines = ["# Ascend benchmark 汇总", "",
             f"运行状态：{receipt['status']}；guard 退出码：{receipt['guard_exit_code']}。", ""]
    if not valid:
        lines += ["本次运行失败，下表仅供诊断，耗时不能作为有效性能结果。", ""]
    lines += ["| 算子 | 规格 | 实现 | 状态 | 平均耗时（μs） | 标准差（μs） | 估算有效带宽（GB/s） | 最大绝对误差 | 协议 |",
              "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    path = run / "results.jsonl"
    rows = 0
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                lines.append(f"| — | results.jsonl 第 {number} 行不完整 | — | 无法解析 | — | — | — | — | — |")
                continue
            implementations = row.get("implementations") or {"—": {}}
            for name, result in implementations.items():
                measurement = result.get("measurement", {})
                correctness = result.get("correctness", {})
                latency = measurement.get("latency_ms")
                deviation = measurement.get("stdev_ms")
                cells = [row.get("op", "—"), row.get("case_id", "—"), name,
                         row.get("status", "—"),
                         f"{latency * 1000:.3f}" if latency is not None else "—",
                         f"{deviation * 1000:.3f}" if deviation is not None else "—",
                         _gather_bandwidth(row, latency),
                         correctness.get("max_abs_err", "—"),
                         measurement.get("protocol", "仅正确性" if row.get("status") == "verified" else "—")]
                lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
                rows += 1
    if not rows:
        lines += ["", "没有完整的 case 结果，请查看 run.log。"]
    lines += ["", "耗时是设备执行区间并集的平均值；1 ms = 1000 μs。",
              "Gather 有效带宽 = 输出元素数 ×（cache 元素字节数 + dst 元素字节数）/ 耗时；",
              "不包含元数据，不是硬件实测 HBM 带宽，其他算子不估算此项。",
              "unlocked 协议为不锁频的开发测量；仅正确性运行不包含耗时。", ""]
    report = "\n".join(lines)
    (run / "summary.md").write_text(report, encoding="utf-8")
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="new directory; existing output is never reused"
    )
    parser.add_argument("--owner", required=True)
    parser.add_argument("--device", type=int, help="physical NPU ID on a multi-card host")
    parser.add_argument("--wait", type=int, default=600)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--guard",
        type=Path,
        default=Path(os.environ.get("TILEOPS_NPU_GUARD", Path(__file__).parent / "npu_guard.py")),
        help="site-audited host guard Python entry; defaults to the bundled guard",
    )
    args, rest = parser.parse_known_args()
    if args.device is not None and args.device < 0:
        parser.error("--device must be nonnegative")
    if "--help" in rest:
        return 0
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    guard = args.guard.resolve(strict=True)
    command = [
        sys.executable,
        str(guard),
        "run",
        *(["--device", str(args.device)] if args.device is not None else []),
        "--owner",
        args.owner,
        "--wait",
        str(args.wait),
        "--timeout",
        str(args.timeout),
        "--",
        sys.executable,
        "-m",
        "benchmarks.ascend.runner",
        "--run-dir",
        str(run),
        *rest,
    ]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (run / "run.log").open("x") as log:
        result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    child = run / "result.json"
    summary = json.loads(child.read_text()) if child.exists() else {}
    success = result.returncode == 0 and summary.get("status") == "passed"
    receipt = {
        "status": "passed" if success else "failed",
        "guard_exit_code": result.returncode,
        "started_at": started,
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": command,
        "result": summary,
    }
    (run / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"run_dir": str(run), **receipt}, indent=2))
    write_summary(run, receipt)
    return 0 if success else (result.returncode or 1)


if __name__ == "__main__":
    sys.exit(main())
