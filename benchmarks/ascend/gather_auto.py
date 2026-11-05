"""Compare the default manual gather with experimental automatic schedules."""

import math

import torch

from benchmarks.api import Implementation
from tileops.kernels.ascend.attention.paged_cache_gather_auto import AscendPagedKVCacheGatherAutoKernel
from tileops.kernels.attention.paged_cache_gather import PagedKVCacheGatherCall


class AutoGather:
    """Public-layout adapter; compile on first use, including strided-output staging."""

    def __init__(self, stages=1):
        self.stages = stages
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
            kernel = AscendPagedKVCacheGatherAutoKernel(call, stages=self.stages)
            self.kernels[call] = kernel
        output = dst if dst.is_contiguous() else torch.empty_like(dst, memory_format=torch.contiguous_format)
        kernel(output, cache, table.contiguous(), cu.contiguous(),
               starts.contiguous() if starts is not None else None,
               scale.contiguous() if scale is not None else None)
        if output is not dst:
            dst.copy_(output)
        return None


class _Run:
    def __init__(self, inputs, stages):
        self.dst = torch.empty_like(inputs[0])
        self.inputs = inputs[1:]
        self.op = AutoGather(stages)
        self.provenance = {"schedule": "automatic", "stages": stages,
                           "experimental": True, "explicit_events": False}

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
    return result
