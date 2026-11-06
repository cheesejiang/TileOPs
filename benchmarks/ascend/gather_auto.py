"""Compare the default manual gather with experimental automatic schedules."""

import math

import torch

from benchmarks.api import Implementation
from tileops.kernels.ascend.attention.paged_cache_gather_auto import AscendPagedKVCacheGatherAutoKernel
from tileops.kernels.attention.paged_cache_gather import PagedKVCacheGatherCall


class AutoGather:
    """Public-layout adapter; compile on first use, including strided-output staging."""

    def __init__(self, stages=1, tile_rows=16, segmented=False, group_tiles=None, double_buffer=False, simt=False, store_l2_cache_ctrl=4, precompute=False):
        self.precompute = precompute
        self.store_l2_cache_ctrl = store_l2_cache_ctrl
        self.simt = simt
        self.double_buffer = double_buffer
        self.group_tiles = group_tiles
        self.stages = stages
        self.tile_rows = tile_rows
        self.segmented = segmented
        self.kernels = {}

    def __call__(self, dst, cache, table, cu, starts=None, scale=None):
        if not dst.numel():
            return None
        call = PagedKVCacheGatherCall(
            batch=table.shape[0], num_pages=cache.shape[0], page_size=cache.shape[1],
            table_width=table.shape[1], row_width=math.prod(dst.shape[1:]),
            dtype=cache.dtype, out_dtype=dst.dtype, has_starts=starts is not None,
            device=dst.device,
        )
        cache_key = (call, dst.shape[0]) if self.precompute else call
        kernel = self.kernels.get(cache_key)
        if kernel is None:
            if self.precompute:
                from tileops.kernels.ascend.attention.paged_cache_gather_precompute import AscendPagedKVCacheGatherPrecomputeKernel
                kernel = AscendPagedKVCacheGatherPrecomputeKernel(call, total_rows=dst.shape[0])
            elif self.simt:
                from tileops.kernels.ascend.attention.paged_cache_gather_simt import AscendPagedKVCacheGatherSimtKernel
                kernel = AscendPagedKVCacheGatherSimtKernel(call)
            elif self.group_tiles is not None:
                from tileops.kernels.ascend.attention.paged_cache_gather_batch import AscendPagedKVCacheGatherBatchKernel
                kernel = AscendPagedKVCacheGatherBatchKernel(call, group_tiles=self.group_tiles, double_buffer=self.double_buffer, tile_rows=self.tile_rows, store_l2_cache_ctrl=self.store_l2_cache_ctrl)
            else:
                kernel = AscendPagedKVCacheGatherAutoKernel(
                    call, stages=self.stages, tile_rows=self.tile_rows, segmented=self.segmented,
                    store_l2_cache_ctrl=self.store_l2_cache_ctrl)
            self.kernels[cache_key] = kernel
        output = dst if dst.is_contiguous() else torch.empty_like(dst, memory_format=torch.contiguous_format)
        kernel(output, cache, table.contiguous(), cu.contiguous(),
               starts.contiguous() if starts is not None else None,
               scale.contiguous() if scale is not None else None)
        if output is not dst:
            dst.copy_(output)
        return None


class _Run:
    def __init__(self, inputs, stages, tile_rows=16, segmented=False, group_tiles=None, double_buffer=False, simt=False, store_l2_cache_ctrl=4, precompute=False):
        self.dst = torch.empty_like(inputs[0])
        self.inputs = inputs[1:]
        self.op = AutoGather(stages, tile_rows, segmented, group_tiles, double_buffer, simt, store_l2_cache_ctrl, precompute)
        self.provenance = {"schedule": "simt-direct" if simt else ("batched" if group_tiles else "automatic"), "stages": stages,
                           "group_tiles": group_tiles, "double_buffer": double_buffer,
                           "tile_rows": tile_rows, "store_l2_cache_ctrl": store_l2_cache_ctrl,
                           "segmented": segmented, "precompute": precompute,
                           "experimental": True, "explicit_events": group_tiles is not None}

    def __call__(self):
        self.op(self.dst, *self.inputs)
        return self.dst

    def reset(self):
        self.dst.fill_(float("nan"))


def make_implementations(op_name, case):
    if op_name != "PagedKVCacheGatherFwdOp":
        return {}
    result = {}
    for stages in (1, 2):
        run = _Run(case.inputs, stages)
        result[f"tilelang-auto{stages}"] = Implementation(run, args=(), reset=run.reset)
    # A larger tile cannot take the contiguous fast path on a 16-row page.
    if case.inputs[1].shape[1] >= 32:
        run = _Run(case.inputs, stages=2, tile_rows=32)
        result["tilelang-auto2-r32"] = Implementation(run, args=(), reset=run.reset)
    run = _Run(case.inputs, stages=2, tile_rows=16, segmented=True)
    result["tilelang-auto2-segments"] = Implementation(run, args=(), reset=run.reset)
    if case.inputs[1].dtype in (torch.float16, torch.bfloat16):
        for group_tiles in (2, 4):
            run = _Run(case.inputs, stages=1, group_tiles=group_tiles)
            result[f"tilelang-batch{group_tiles}"] = Implementation(run, args=(), reset=run.reset)
        for group_tiles in (2, 4):
            run = _Run(case.inputs, stages=1, group_tiles=group_tiles, double_buffer=True)
            result[f"tilelang-batch{group_tiles}-db"] = Implementation(run, args=(), reset=run.reset)
        run = _Run(case.inputs, stages=1, group_tiles=16, tile_rows=4)
        result["tilelang-batch16-r4"] = Implementation(run, args=(), reset=run.reset)
    return result


def make_batch4_implementations(op_name, case):
    """Focused comparison without compiling unrelated experimental schedules."""
    if op_name != "PagedKVCacheGatherFwdOp" or case.inputs[1].dtype not in (torch.float16, torch.bfloat16):
        return {}
    result = {}
    for double_buffer in (False, True):
        run = _Run(case.inputs, stages=1, group_tiles=4, double_buffer=double_buffer)
        name = "tilelang-batch4-db" if double_buffer else "tilelang-batch4"
        result[name] = Implementation(run, args=(), reset=run.reset)
    return result


def make_load_granularity_implementations(op_name, case):
    """Single-buffer 64-row groups: compare four 16-row and sixteen 4-row loads."""
    if op_name != "PagedKVCacheGatherFwdOp" or case.inputs[1].dtype not in (torch.float16, torch.bfloat16):
        return {}
    result = {}
    for group_tiles, tile_rows, name in (
        (4, 16, "tilelang-batch4"), (16, 4, "tilelang-batch16-r4"),
    ):
        run = _Run(case.inputs, stages=1, group_tiles=group_tiles, tile_rows=tile_rows)
        result[name] = Implementation(run, args=(), reset=run.reset)
    return result


def make_simt_implementations(op_name, case):
    """Only SIMT and auto2; use --implementations to exclude the default binder."""
    if op_name != "PagedKVCacheGatherFwdOp":
        return {}
    result = {}
    for name, simt in (("tilelang-auto2", False), ("tilelang-simt", True)):
        run = _Run(case.inputs, stages=2, simt=simt)
        result[name] = Implementation(run, args=(), reset=run.reset)
    return result


def make_overlap_implementations(op_name, case):
    """128 KiB payload at width 1024 BF16: serial 64 rows versus 2x32 rows."""
    if op_name != "PagedKVCacheGatherFwdOp" or case.inputs[1].dtype not in (torch.float16, torch.bfloat16):
        return {}
    result = {}
    for name, group_tiles, double_buffer in (
        ("tilelang-batch16-r4", 16, False),
        ("tilelang-overlap32-r4", 8, True),
    ):
        run = _Run(case.inputs, stages=1, group_tiles=group_tiles,
                   tile_rows=4, double_buffer=double_buffer)
        result[name] = Implementation(run, args=(), reset=run.reset)
    return result


def make_store_policy_implementations(op_name, case):
    """Same overlap32-r4 schedule, changing only the UB-to-GM L2 policy."""
    if op_name != "PagedKVCacheGatherFwdOp" or case.inputs[1].dtype not in (torch.float16, torch.bfloat16):
        return {}
    result = {}
    for mode in (4, 0):
        run = _Run(case.inputs, stages=1, group_tiles=8, tile_rows=4,
                   double_buffer=True, store_l2_cache_ctrl=mode)
        name = f"tilelang-overlap32-r4-store{mode}"
        result[name] = Implementation(run, args=(), reset=run.reset)
    return result


def make_store0_overlap_implementations(op_name, case):
    """Compare single 64-row and alternating 32-row regions, both store mode 0."""
    if op_name != "PagedKVCacheGatherFwdOp" or case.inputs[1].dtype not in (torch.float16, torch.bfloat16):
        return {}
    result = {}
    for name, group_tiles, double_buffer in (
        ("tilelang-batch16-r4-store0", 16, False),
        ("tilelang-overlap32-r4-store0", 8, True),
    ):
        run = _Run(case.inputs, stages=1, group_tiles=group_tiles, tile_rows=4,
                   double_buffer=double_buffer, store_l2_cache_ctrl=0)
        result[name] = Implementation(run, args=(), reset=run.reset)
    return result


def make_auto2_store0_implementations(op_name, case):
    """Compare automatic scheduling with manual overlap under the same store policy."""
    if op_name != "PagedKVCacheGatherFwdOp" or case.inputs[1].dtype not in (torch.float16, torch.bfloat16):
        return {}
    automatic = _Run(case.inputs, stages=2, tile_rows=16, store_l2_cache_ctrl=0)
    overlap = _Run(case.inputs, stages=1, group_tiles=8, tile_rows=4,
                   double_buffer=True, store_l2_cache_ctrl=0)
    return {
        "tilelang-auto2-store0": Implementation(automatic, args=(), reset=automatic.reset),
        "tilelang-overlap32-r4-store0": Implementation(overlap, args=(), reset=overlap.reset),
    }


def make_gatherv2_layout_implementations(op_name, case):
    """Compare 2x32-row/4-row loads with 2x60-row/1-row loads, both store0."""
    if op_name != "PagedKVCacheGatherFwdOp" or case.inputs[1].dtype not in (torch.float16, torch.bfloat16):
        return {}
    result = {}
    for name, group_tiles, tile_rows in (
        ("tilelang-overlap32-r4-store0", 8, 4),
        ("tilelang-overlap60-r1-store0", 60, 1),
    ):
        run = _Run(case.inputs, stages=1, group_tiles=group_tiles, tile_rows=tile_rows,
                   double_buffer=True, store_l2_cache_ctrl=0)
        result[name] = Implementation(run, args=(), reset=run.reset)
    return result


def make_precompute_implementations(op_name, case):
    """Address preparation stays inside the measured kernel in both candidates."""
    if op_name != "PagedKVCacheGatherFwdOp" or case.inputs[1].dtype not in (torch.float16, torch.bfloat16):
        return {}
    result = {}
    for name, precompute in (("tilelang-overlap32-r4-store0", False),
                             ("tilelang-precompute32-r4-store0", True)):
        run = _Run(case.inputs, stages=1, group_tiles=8, tile_rows=4,
                   double_buffer=True, store_l2_cache_ctrl=0, precompute=precompute)
        result[name] = Implementation(run, args=(), reset=run.reset)
    return result
