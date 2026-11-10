"""Row-batch sweep for the seven paged cache write manifest workloads.

Use with the Ascend runner's --implementation-factory option. Every candidate
uses private cache pools and is checked against the independent reference
before timing. The generic implementation is the pre-optimization baseline.
"""

def candidates(op_name, case, *, sweep=True):
    if op_name != "PagedKVCacheWriteFwdOp":
        return {}
    import torch

    from benchmarks.baselines import private_inputs
    from tileops._kernels.ascend.attention.paged_cache_write import build, build_batched

    inputs = case.inputs
    k, _, pages, _, _, _, _ = inputs
    n, heads, dim = k.shape
    width = heads * dim
    physical = pages.shape[0] * pages.shape[1]
    dtype = str(k.dtype).removeprefix("torch.")
    quantized = pages.dtype == torch.float8_e4m3fn

    def implementation(program):
        def run(k, v, kp, vp, slots, ks, vs):
            kbytes = kp.view(torch.uint8) if quantized else kp
            vbytes = vp.view(torch.uint8) if quantized else vp
            args = [
                k.view(n, width),
                v.view(n, width),
                kbytes.view(physical, width),
                vbytes.view(physical, width),
                slots,
            ]
            if quantized:
                args.extend([ks, vs])
            program(*args)
            return kp, vp

        return private_inputs(run, inputs, 2, 3)

    result = {"generic": implementation(build(width, physical, dtype, quantized))}
    if sweep and width % 64 == 0 and n:
        for rows in (1, 4, 8, 16):
            if n % rows or rows * width > 16384:
                continue
            seen = set()
            for limit in (32, 64):
                cores = min(limit, n // rows)
                if cores in seen:
                    continue
                seen.add(cores)
                program = build_batched(width, physical, dtype, quantized, rows, cores)
                result[f"rows-{rows}-cores-{cores}"] = implementation(program)
    return result


def write_summary(directory, metadata, rows, *, complete):
    """Publish a readable report after each case, including partial/failed runs."""
    import math

    def cell(value):
        return str(value).replace("|", "&#124;").replace("\n", "<br>")

    options = metadata["arguments"]
    passed = sum(row.get("status") in ("verified", "measured") for row in rows)
    failed = sum(row.get("status") == "failed" for row in rows)
    state = "进行中" if not complete else ("失败" if failed else "通过")
    lines = [
        "# Paged KV Cache Write 性能与正确性报告", "",
        f"- 状态：{state}；已完成 {len(rows)} 个规格，通过 {passed}，失败 {failed}。",
        f"- 设备：{cell(metadata['device'])}（{cell(metadata['device_name'])}）。",
        f"- 环境：PyTorch {cell(metadata['torch'])}；torch_npu {cell(metadata['torch_npu'])}；"
        f"TileLang {cell(metadata['tilelang'])}。",
        f"- 模式：{'仅验证精度' if options['verify_only'] else '精度验证及性能测试'}；不锁频。",
        f"- 计时协议：`{metadata['protocol']}`；预热 {options['warmup']} 次，"
        f"采样 {options['repeats']} 次；计时前清缓存 {options['flush_mb']} MiB。",
        "- 基线：原始通用 Ascend 内核 `generic`；加速比 = 基线均值 / 当前实现均值。",
        "- 正确性使用独立参考检查全部输出及输入保持规则；失败规格不视为验证通过。",
        "",
        "| 规格 | 实现 | 正确性 | 平均耗时 (µs) | 中位耗时 (µs) | 相对基线加速比 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    speedups = []
    for row in rows:
        implementations = row.get("implementations", {})
        if row.get("status") == "failed" or not implementations:
            lines.append(f"| {cell(row['case_id'])} | — | 未完成，见失败详情 | — | — | — |")
            continue
        for name, result in implementations.items():
            check = result["correctness"]
            correctness = f"通过（{check['checked_outputs']}/{check['total_outputs']} 输出）"
            measurement = result.get("measurement")
            mean = f"{measurement['latency_ms'] * 1000:.3f}" if measurement else "—"
            median = f"{measurement['median_ms'] * 1000:.3f}" if measurement else "—"
            speedup = result.get("speedup_vs_generic")
            ratio = f"{speedup:.3f}×" if speedup is not None else "—"
            lines.append(
                f"| {cell(row['case_id'])} | {cell(name)} | {correctness} | {mean} | {median} | {ratio} |"
            )
            if name == "tileops" and speedup is not None:
                speedups.append(speedup)
    if speedups:
        geometric = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
        lines.extend(["", f"默认优化实现的加速比几何均值：**{geometric:.3f}×**"
                      f"（仅统计 {len(speedups)} 个成功测量的规格）。"])
    for row in rows:
        if row.get("error"):
            lines.extend(["", f"## 失败详情：{cell(row['case_id'])}", ""])
            lines.extend("    " + line for line in row["error"].splitlines())
    lines.extend([
        "", "耗时采用 NPU Event，可能包含主机提交间隙；未锁频结果不与锁频 profiler 协议直接比较。",
        "仅验证模式不产生性能数值。原始数据：[环境](environment.json)、"
        "[逐项结果](results.jsonl)、[最终状态](result.json)。", "",
    ])
    temporary = directory / "summary.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(directory / "summary.md")


class SingleDeviceEventTimer:
    """Unlocked NPU stream event timing on the process's selected device.

    Reset and optional cache eviction finish before each event window. These
    numbers use a different protocol from the shared locked profiler runner.
    """

    def __init__(self, device, warmup=3, repeats=15, flush_mb=1024):
        import torch

        if warmup < 0 or repeats < 1 or flush_mb < 0:
            raise ValueError("warmup/flush_mb must be nonnegative and repeats must be positive")
        self.device = device
        self.warmup = warmup
        self.repeats = repeats
        self.flush_buffer = (
            torch.empty(flush_mb * (1 << 20), dtype=torch.uint8, device=device)
            if flush_mb else None
        )

    def measure(self, fn, args=(), *, reset=None, name="call"):
        import torch

        with torch.npu.device(self.device):
            return self._measure(fn, args, reset=reset, name=name)

    def _measure(self, fn, args, *, reset, name):
        import math
        import statistics

        import torch

        def prepare():
            if reset is not None:
                reset()
            if self.flush_buffer is not None:
                self.flush_buffer.zero_()
            torch.npu.synchronize(self.device)

        for _ in range(self.warmup):
            prepare()
            output = fn(*args)
            torch.npu.synchronize(self.device)
            del output
        samples = []
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        # Materialize both lazy event handles outside the measurement windows.
        start.record()
        end.record()
        end.synchronize()
        for _ in range(self.repeats):
            prepare()
            start.record()
            output = fn(*args)
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
            del output
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError(f"Invalid NPU event duration for {name}: {elapsed}")
            samples.append(elapsed)
        return {
            "protocol": "ascend-single-device-unlocked-events-v1",
            "metric": "mean_npu_stream_event_ms",
            "device": str(self.device),
            "clock_policy": "unlocked",
            "warmup": self.warmup,
            "repeats": self.repeats,
            "flush_bytes": 0 if self.flush_buffer is None else self.flush_buffer.numel(),
            "latency_ms": statistics.mean(samples),
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0,
            "samples_ms": samples,
        }


def main(argv=None):
    import argparse
    import json
    import re
    from importlib.metadata import version
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Single-NPU paged cache write benchmark, no clock lock")
    parser.add_argument("--device", type=int, default=0, help="logical NPU index visible to this process")
    parser.add_argument("--run-dir", type=Path, required=True, help="new output directory")
    parser.add_argument("--case-regex", default=".*")
    parser.add_argument("--sweep", action="store_true", help="also compare row/core configurations")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--flush-mb", type=int, default=1024, help="cache eviction size; 0 disables it")
    args = parser.parse_args(argv)
    if args.device < 0 or args.warmup < 0 or args.repeats < 1 or args.flush_mb < 0:
        parser.error("invalid device or timing arguments")
    pattern = re.compile(args.case_regex)

    import torch
    import torch_npu

    # Select before allocations, backend registration and kernel compilation.
    if args.device >= torch.npu.device_count():
        parser.error(f"NPU {args.device} is not visible; count={torch.npu.device_count()}")
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)

    from benchmarks.api import cases
    from benchmarks.ascend.runner import benchmark_case
    from benchmarks.baselines import private_inputs
    from tileops.attention import PagedKVCacheWriteFwdOp
    from workloads.attention.paged_cache_write import paged_cache_write_result
    from workloads.device import set_run_device

    set_run_device(device)
    selected = [case for case in cases(PagedKVCacheWriteFwdOp) if pattern.search(case.id)]
    if not selected:
        parser.error("case regex matched no workloads")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "device": str(device),
        "device_name": torch.npu.get_device_name(args.device),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "tilelang": version("tilelang"),
        "clock_policy": "unlocked",
        "protocol": "ascend-single-device-unlocked-events-v1",
        "arguments": vars(args),
    }
    (args.run_dir / "environment.json").write_text(json.dumps(metadata, indent=2, default=str))
    report_rows = []
    write_summary(args.run_dir, metadata, report_rows, complete=False)
    timer = None if args.verify_only else SingleDeviceEventTimer(
        device, args.warmup, args.repeats, args.flush_mb
    )
    failed = 0
    with (args.run_dir / "results.jsonl").open("x") as output:
        for case in selected:
            row = {"case_id": case.id, "device": str(device)}
            try:
                # Seed the selected device only; do not initialize other NPUs.
                torch.random.default_generator.manual_seed(20261110)
                with torch.npu.device(device):
                    torch.npu.manual_seed(20261110)
                inputs = case.inputs
                op = PagedKVCacheWriteFwdOp(target="ascend")

                def run(*inputs):
                    return paged_cache_write_result(op, *inputs)

                implementations = {"tileops": private_inputs(run, inputs, 2, 3)}
                implementations.update(candidates("PagedKVCacheWriteFwdOp", case, sweep=args.sweep))
                row["implementations"] = benchmark_case(
                    case, implementations, timer, name=case.id, verify_only=args.verify_only
                )
                row["status"] = "verified" if args.verify_only else "measured"
                if not args.verify_only:
                    measurements = row["implementations"]
                    baseline = measurements["generic"]["measurement"]["latency_ms"]
                    for result in measurements.values():
                        result["speedup_vs_generic"] = baseline / result["measurement"]["latency_ms"]
            except Exception:
                import traceback

                row.update(status="failed", error=traceback.format_exc())
                failed += 1
            finally:
                case._release()
            output.write(json.dumps(row, default=str) + "\n")
            output.flush()
            report_rows.append(row)
            write_summary(args.run_dir, metadata, report_rows, complete=False)
            print(json.dumps(row, default=str), flush=True)
    summary = {"status": "failed" if failed else "passed", "cases": len(selected), "failed": failed}
    (args.run_dir / "result.json").write_text(json.dumps(summary, indent=2))
    write_summary(args.run_dir, metadata, report_rows, complete=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
