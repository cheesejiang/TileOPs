#!/usr/bin/env python3
"""Launch one llama-70b-cp gather for an external simulator/profiler.

No profiler nesting, warmup loop or cache eviction. Input transfers and optional
CPU result validation are setup work, not an isolated timing protocol.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("manual", "auto2", "simt", "batch2", "batch4", "batch2-db", "batch4-db", "batch16-r4", "overlap32-r4", "overlap60-r1", "precompute32-r4", "precompute60-r1", "gatherv2-core"), required=True)
    parser.add_argument("--store-l2-cache-ctrl", type=int, choices=(0, 4), default=4,
                        help="auto2/batch candidates: UB-to-GM cache policy")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--profile-dir", type=Path,
                        help="real-device profiler with automatic analysis; omit for simulator runs")
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
        flat_cache = cache.flatten(0, 1)

        def launch():
            kernel(flat_cache, slots, dst)
    else:
        call = PagedKVCacheGatherCall(
            batch=batch, num_pages=pages, page_size=ps, table_width=table_width,
            row_width=width, dtype=torch.bfloat16, out_dtype=torch.bfloat16,
            has_starts=True, device=device,
        )
        if args.implementation in ("precompute32-r4", "precompute60-r1"):
            from tileops.kernels.ascend.attention.paged_cache_gather_precompute import AscendPagedKVCacheGatherPrecomputeKernel
            kernel = AscendPagedKVCacheGatherPrecomputeKernel(
                call, total_rows=dst.shape[0],
                group_rows=60 if args.implementation == "precompute60-r1" else 32,
                tile_rows=1 if args.implementation == "precompute60-r1" else 4)
        elif args.implementation == "simt":
            from tileops.kernels.ascend.attention.paged_cache_gather_simt import AscendPagedKVCacheGatherSimtKernel
            kernel = AscendPagedKVCacheGatherSimtKernel(call)
        elif args.implementation in ("batch2", "batch4", "batch2-db", "batch4-db", "batch16-r4", "overlap32-r4", "overlap60-r1"):
            from tileops.kernels.ascend.attention.paged_cache_gather_batch import AscendPagedKVCacheGatherBatchKernel
            kernel = AscendPagedKVCacheGatherBatchKernel(
                call, group_tiles=60 if args.implementation == "overlap60-r1" else (8 if args.implementation == "overlap32-r4" else (16 if args.implementation == "batch16-r4" else (4 if args.implementation.startswith("batch4") else 2))),
                tile_rows=1 if args.implementation == "overlap60-r1" else (4 if args.implementation in ("batch16-r4", "overlap32-r4") else 16),
                double_buffer=args.implementation.endswith("-db") or args.implementation in ("overlap32-r4", "overlap60-r1"),
                store_l2_cache_ctrl=args.store_l2_cache_ctrl)
        else:
            kernel = (AscendPagedKVCacheGatherKernel(call) if args.implementation == "manual"
                  else AscendPagedKVCacheGatherAutoKernel(
                      call, stages=2, tile_rows=16, store_l2_cache_ctrl=args.store_l2_cache_ctrl))
        if args.source_dir:
            args.source_dir.mkdir(parents=True, exist_ok=True)
            source = kernel._compiled.get_kernel_source()
            source_path = args.source_dir / f"{args.implementation}.cpp"
            source_path.write_text(source, encoding="utf-8")
            if args.implementation == "batch16-r4":
                import hashlib
                import re

                # Check emitted source, rather than just the Python flags.
                event = lambda src, dst: rf"asc_sync_(?:notify|wait)\(PIPE_{src}, PIPE_{dst},"
                read_done = re.search(event("MTE2", "MTE3"), source)
                write_done = re.search(event("MTE3", "MTE2"), source)
                stores = [m.start() for m in re.finditer("asc_copy_ub2gm_align\\(", source)]
                if not (read_done and write_done and stores
                        and read_done.start() < min(stores)
                        and max(stores) < write_done.start()):
                    raise RuntimeError("batch16-r4 generated source does not match single-group read/write ordering")
                manifest = {
                    "implementation": args.implementation,
                    "double_buffer": False, "buffers": 1,
                    "group_rows": 64, "load_rows": 4,
                    "full_group_loads": 16, "full_group_store_bytes": 131072,
                    "store_l2_cache_ctrl": args.store_l2_cache_ctrl,
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "source_order_checked": True,
                    "note": "Static source-order check only; not proof of device timing.",
                }
                (args.source_dir / "batch16-r4.launch.json").write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(manifest, indent=2), flush=True)
        def launch():
            kernel(dst, cache, table, cu, starts)

    torch.npu.synchronize()
    if args.profile_dir:
        report = args.profile_dir.resolve()
        report.mkdir(parents=True, exist_ok=False)
        # Warm up outside the window, including the ACLNN first-call setup.
        launch()
        torch.npu.synchronize()
        print(f"Collecting real-device profile into {report}", flush=True)
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(report), analyse_flag=True),
        ) as prof:
            launch()
            torch.npu.synchronize()
            prof.step()
        print(f"Profiler raw directory: {prof.prof_if.prof_path}", flush=True)
        print("Generated reports:", flush=True)
        for path in sorted(report.rglob("*")):
            if path.is_file() and path.suffix in (".csv", ".json", ".db"):
                print(path, flush=True)
                if "op_summary" in path.name.lower() and path.suffix == ".csv":
                    with path.open(encoding="utf-8-sig", newline="") as handle:
                        for row_data in csv.DictReader(handle):
                            names = {k: v for k, v in row_data.items()
                                     if "name" in k.lower() or "type" in k.lower()}
                            print(json.dumps(names, ensure_ascii=False), flush=True)
        print("If no operator report was generated, retain the raw directory and analysis warnings.", flush=True)
    else:
        launch()
    torch.npu.synchronize()
    if args.check:
        expected = cache_cpu[slots_cpu].view(batch * length, 8, 128)
        torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print(f"Completed {args.implementation}; this script does not report benchmark latency.")


if __name__ == "__main__":
    main()
