"""Verify experimental Ascend SIMT FP8 arithmetic against a CPU reference.

Run the entire foreground process under the site's reservation guard, including
torch_npu imports and compilation. Example from the TileOPs checkout:

    npu-guard run --owner jyw/fp8-simt-verify --wait 3600 --timeout 3600 -- \
        python scripts/verify_fp8_simt_arithmetic.py \
        --output /results/fp8-simt-verification.json

The reference decodes the actual FP8 inputs on CPU, computes in FP32, then rounds
once to the requested FP8 format. It checks exact finite results, infinity and
NaN masks, signed zeros, shape/dtype, and bytewise input preservation.

All 256 x 256 FP8 bit-pattern pairs are included for each selected operation and
format, including subnormals, overflow, NaNs, and E5M2 infinities. Differences in
overflow/saturation behavior are failures, not silently clamped or excluded.
This is an accuracy experiment, not a performance benchmark.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path


def _load_kernels():
    # Load this experimental file directly so verification does not import or
    # register unrelated kernels through tileops.kernels.ascend.__init__.
    path = (
        Path(__file__).resolve().parents[1]
        / "src/tileops/kernels/ascend/attention/fp8_simt_arithmetic.py"
    )
    spec = importlib.util.spec_from_file_location("fp8_simt_arithmetic_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cases(torch, dtype, block_size, seed):
    raw = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    yield (
        "all-bit-pattern-pairs",
        raw.repeat_interleave(256).view(dtype),
        raw.repeat(256).view(dtype),
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    shapes = [(0,), (1,), (31,), (32,), (127,), (128,), (129,)]
    shapes += [(block_size - 1,), (block_size,), (block_size + 1,), (7, 19), (17, 257)]
    for shape in dict.fromkeys(shapes):
        # Mix exact cancellation and generic values. Quantize on CPU so the
        # reference and kernel receive exactly the same representable inputs.
        a = (torch.randn(shape, generator=generator) * 8).to(dtype)
        b = (torch.randn(shape, generator=generator) * 8).to(dtype)
        if a.numel():
            b.view(torch.uint8).reshape(-1)[::7] = (
                a.view(torch.uint8).reshape(-1)[::7] ^ 0x80
            )
        yield "random-" + "x".join(map(str, shape)), a, b


def _check(torch, fn, operation, a_cpu, b_cpu, device, block_size, threads):
    # Transfer raw bytes so input generation needs no FP8 device-side operator.
    a = a_cpu.view(torch.uint8).to(device).view(a_cpu.dtype)
    b = b_cpu.view(torch.uint8).to(device).view(b_cpu.dtype)
    actual = fn(a, b, block_size=block_size, threads=threads)
    torch.npu.synchronize()

    if actual.shape != a_cpu.shape or actual.dtype != a_cpu.dtype or actual.device != a.device:
        raise AssertionError("output shape, dtype or device does not match the input")
    for name, tensor, original in (("a", a, a_cpu), ("b", b, b_cpu)):
        if not torch.equal(tensor.view(torch.uint8).cpu(), original.view(torch.uint8)):
            raise AssertionError(f"input {name} was modified (bytewise comparison)")

    result = actual.view(torch.uint8).cpu().view(a_cpu.dtype).float()
    af, bf = a_cpu.float(), b_cpu.float()
    unrounded = af + bf if operation == "add" else af * bf
    expected = unrounded.to(a_cpu.dtype).float()

    nan_mismatch = int((torch.isnan(result) != torch.isnan(expected)).sum().item())
    inf_mismatch = int(
        ((torch.isposinf(result) != torch.isposinf(expected))
         | (torch.isneginf(result) != torch.isneginf(expected))).sum().item()
    )
    finite_pair = torch.isfinite(result) & torch.isfinite(expected)
    finite_mismatch = int(((result != expected) & finite_pair).sum().item())
    finite_mask_mismatch = int(
        (torch.isfinite(result) != torch.isfinite(expected)).sum().item()
    )
    both_zero = (result == 0) & (expected == 0)
    zero_sign_mismatch = int(
        ((torch.signbit(result) != torch.signbit(expected)) & both_zero).sum().item()
    )
    max_abs_error = (
        float((result[finite_pair] - expected[finite_pair]).abs().max().item())
        if finite_pair.any().item() else 0.0
    )
    # Report expected FP8 quantization loss separately from kernel error.
    quantizable = torch.isfinite(unrounded) & torch.isfinite(expected)
    quantization_error = (
        float((expected[quantizable] - unrounded[quantizable]).abs().max().item())
        if quantizable.any().item() else 0.0
    )
    passed = not any((
        nan_mismatch, inf_mismatch, finite_mismatch, finite_mask_mismatch, zero_sign_mismatch,
    ))
    return {
        "status": "passed" if passed else "failed",
        "elements": math.prod(a_cpu.shape),
        "shape": list(a_cpu.shape),
        "max_abs_error_vs_rounded_reference": max_abs_error,
        "max_reference_quantization_error": quantization_error,
        "finite_value_mismatches": finite_mismatch,
        "finite_mask_mismatches": finite_mask_mismatch,
        "nan_mask_mismatches": nan_mismatch,
        "infinity_mismatches": inf_mismatch,
        "zero_sign_mismatches": zero_sign_mismatch,
        "inputs_preserved": True,
    }


def _write_report(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", action="append", choices=("float8_e4m3fn", "float8_e5m2"))
    parser.add_argument("--op", action="append", choices=("add", "mul"))
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, choices=(128, 256), default=128)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument(
        "--output", type=Path, required=True,
        help="new JSON report; existing files are not overwritten",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists; choose a new report filename")
    if not args.device.startswith("npu:"):
        parser.error("--device must select an NPU, for example npu:0")
    if not args.threads <= args.block_size <= 8192 or args.block_size % args.threads:
        parser.error("--block-size must be a multiple of --threads and no larger than 8192")

    report = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "block_size": args.block_size,
        "threads": args.threads,
        "seed": args.seed,
        "reference": "CPU FP32 arithmetic on decoded FP8 inputs, then one FP8 rounding",
        "rtol": 0,
        "atol": 0,
        "cases": [],
    }
    _write_report(report, args.output)
    try:
        # Imports that may initialize the NPU must remain inside npu-guard.
        import torch
        import torch_npu  # noqa: F401

        report["packages"] = {
            name: importlib.metadata.version(name)
            for name in ("torch", "torch_npu", "tilelang")
        }
        torch.npu.set_device(args.device)
        report["device_name"] = torch.npu.get_device_name()
        kernels = _load_kernels()
        for dtype_name in dict.fromkeys(args.dtype or ("float8_e4m3fn", "float8_e5m2")):
            dtype = getattr(torch, dtype_name)
            for operation in dict.fromkeys(args.op or ("add", "mul")):
                fn = getattr(kernels, "fp8_" + operation)
                for name, a, b in _cases(torch, dtype, args.block_size, args.seed):
                    record = {"dtype": dtype_name, "operation": operation, "case": name}
                    try:
                        record.update(_check(
                            torch, fn, operation, a, b, args.device,
                            args.block_size, args.threads,
                        ))
                    except Exception:
                        record.update(status="failed", error=traceback.format_exc())
                        report["cases"].append(record)
                        _write_report(report, args.output)
                        # Stop on runtime/compiler errors rather than repeatedly
                        # retrying an unusable kernel or a failed NPU context.
                        raise
                    report["cases"].append(record)
                    _write_report(report, args.output)
                    print(json.dumps(record, allow_nan=False), flush=True)
        report["status"] = (
            "passed" if all(c["status"] == "passed" for c in report["cases"]) else "failed"
        )
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        print(report["error"], flush=True)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_report(report, args.output)
    print(json.dumps({"status": report["status"], "report": str(args.output)}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
