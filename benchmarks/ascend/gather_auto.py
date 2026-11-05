"""Compare the default manual gather with experimental automatic schedules."""

import math

import torch

from benchmarks.api import Implementation
from tileops.kernels.ascend.attention.paged_cache_gather_auto import AscendPagedKVCacheGatherAutoKernel
from tileops.kernels.attention.paged_cache_gather import PagedKVCacheGatherCall


class AutoGather:
    """Public-layout adapter; compile on first use, including strided-output staging."""

    def __init__(self, stages=1, tile_rows=16, segmented=False, group_tiles=None, double_buffer=False):
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
        kernel = self.kernels.get(call)
        if kernel is None:
            if self.group_tiles is not None:
                from tileops.kernels.ascend.attention.paged_cache_gather_batch import AscendPagedKVCacheGatherBatchKernel
                kernel = AscendPagedKVCacheGatherBatchKernel(call, group_tiles=self.group_tiles, double_buffer=self.double_buffer, tile_rows=self.tile_rows)
            else:
                kernel = AscendPagedKVCacheGatherAutoKernel(
                    call, stages=self.stages, tile_rows=self.tile_rows, segmented=self.segmented)
            self.kernels[call] = kernel
        output = dst if dst.is_contiguous() else torch.empty_like(dst, memory_format=torch.contiguous_format)
        kernel(output, cache, table.contiguous(), cu.contiguous(),
               starts.contiguous() if starts is not None else None,
               scale.contiguous() if scale is not None else None)
        if output is not dst:
            dst.copy_(output)
        return None


class _Run:
    def __init__(self, inputs, stages, tile_rows=16, segmented=False, group_tiles=None, double_buffer=False):
        self.dst = torch.empty_like(inputs[0])
        self.inputs = inputs[1:]
        self.op = AutoGather(stages, tile_rows, segmented, group_tiles, double_buffer)
        self.provenance = {"schedule": "batched" if group_tiles else "automatic", "stages": stages,
                           "group_tiles": group_tiles, "double_buffer": double_buffer,
                           "tile_rows": tile_rows,
                           "segmented": segmented,
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
