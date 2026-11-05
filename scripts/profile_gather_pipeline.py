#!/usr/bin/env python3
"""Launch one llama-70b-cp gather for an external simulator/profiler.

No profiler nesting, warmup loop or cache eviction. Input transfers and optional
CPU result validation are setup work, not an isolated timing protocol.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("manual", "auto2", "gatherv2-core"), required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="compare the result to a CPU reference")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root)]

    import torch
    import torch_npu  # noqa: F401

    from tileops.kernels.ascend.attention.paged_cache_gather import AscendPagedKVCacheGatherKernel
    from tileops.kernels.ascend.attention.paged_cache_gather_auto import AscendPagedKVCacheGatherAutoKernel
    from tileops.kernels.attention.paged_cache_gather import PagedKVCacheGatherCall

    # Exact manifest geometry; values created on CPU to avoid random-generation NPU kernels.
    batch, pages, ps, width, length, first, table_width = 8, 4096, 16, 1024, 4096, 2048, 512
    generator = torch.Generator().manual_seed(2026)
    cache_cpu = torch.randn((pages * ps, width), generator=generator).to(torch.bfloat16)
    table_cpu = torch.randperm(pages, generator=generator, dtype=torch.int64).view(batch, table_width).int()
    cu_cpu = torch.arange(batch + 1, dtype=torch.int32) * length
    starts_cpu = torch.full((batch,), first, dtype=torch.int32)
    positions = torch.arange(length, dtype=torch.int64) + first
    slots_cpu = (table_cpu[:, positions // ps].long() * ps + positions % ps).reshape(-1)
    device = torch.device("npu:0")
    cache = cache_cpu.to(device).view(pages, ps, 8, 128)
    table, cu, starts = (t.to(device) for t in (table_cpu, cu_cpu, starts_cpu))
    dst = torch.empty((batch * length, 8, 128), dtype=torch.bfloat16, device=device)
    if args.implementation == "gatherv2-core":
        from benchmarks.ascend.gather_v2_baseline import GatherV2

        slots = slots_cpu.to(device)
        kernel = GatherV2()
        if args.source_dir:
            raise ValueError("CANN library source is not available from the Python adapter")
        torch.npu.synchronize()
        kernel(cache.flatten(0, 1), slots, dst)
    else:
        call = PagedKVCacheGatherCall(
            batch=batch, num_pages=pages, page_size=ps, table_width=table_width,
            row_width=width, dtype=torch.bfloat16, out_dtype=torch.bfloat16,
            has_starts=True, device=device,
        )
        kernel = (AscendPagedKVCacheGatherKernel(call) if args.implementation == "manual"
                  else AscendPagedKVCacheGatherAutoKernel(call, stages=2, tile_rows=16))
        if args.source_dir:
            args.source_dir.mkdir(parents=True, exist_ok=True)
            (args.source_dir / f"{args.implementation}.cpp").write_text(
                kernel._compiled.get_kernel_source(), encoding="utf-8")
        torch.npu.synchronize()
        kernel(dst, cache, table, cu, starts)
    torch.npu.synchronize()
    if args.check:
        expected = cache_cpu[slots_cpu].view(batch * length, 8, 128)
        torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(f"Completed one {args.implementation} call; this script does not report benchmark latency.")


if __name__ == "__main__":
    main()
