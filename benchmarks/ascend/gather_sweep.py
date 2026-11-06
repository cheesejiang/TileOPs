"""Only requested group/load combinations; all seven gather manifest cases.

Configure comma-separated row counts with TILEOPS_GATHER_GROUP_ROWS and
TILEOPS_GATHER_LOAD_ROWS. Defaults: 16/32/48 and 1/2/4/8/16.
"""
import math
import os

import torch

from benchmarks.api import Implementation
from tileops.kernels.ascend.attention.paged_cache_gather_sweep import AscendPagedKVCacheGatherSweepKernel
from tileops.kernels.attention.paged_cache_gather import PagedKVCacheGatherCall


def _selection(variable, default, allowed):
    values = tuple(dict.fromkeys(int(v.strip()) for v in os.environ.get(variable, default).split(',')))
    if not values or any(v not in allowed for v in values):
        raise ValueError(f'{variable}: expected comma-separated values in {sorted(allowed)}')
    return values


class _Run:
    def __init__(self, inputs, group_rows, load_rows):
        dst, cache, table, cu, starts, scale = inputs
        self.dst = torch.empty_like(dst, memory_format=torch.contiguous_format)
        self.inputs = inputs[1:]
        call = PagedKVCacheGatherCall(
            batch=table.shape[0], num_pages=cache.shape[0], page_size=cache.shape[1],
            table_width=table.shape[1], row_width=math.prod(dst.shape[1:]),
            dtype=cache.dtype, out_dtype=dst.dtype, has_starts=starts is not None,
            device=dst.device,
        )
        # Compile lazily, inside verification, so errors are recorded by runner.
        self.call, self.kernel = call, None
        self.group_rows, self.load_rows = group_rows, load_rows
        self.provenance = dict(schedule='overlap-sweep', group_rows=group_rows,
                               load_rows=load_rows, buffers=2, store_l2_cache_ctrl=0,
                               precompute=False, experimental=True)

    def __call__(self):
        if self.kernel is None:
            self.kernel = AscendPagedKVCacheGatherSweepKernel(self.call, self.group_rows, self.load_rows)
        self.kernel(self.dst, *self.inputs)
        return self.dst

    def reset(self):
        self.dst.fill_(float('nan'))


def make_implementations(op_name, case):
    if op_name != 'PagedKVCacheGatherFwdOp':
        return {}
    groups = _selection('TILEOPS_GATHER_GROUP_ROWS', '16,32,48', {16,32,48,64})
    loads = _selection('TILEOPS_GATHER_LOAD_ROWS', '1,2,4,8,16', {1,2,4,8,16})
    result = {}
    for group in groups:
        for load in loads:
            if group % load:
                raise ValueError('group rows must be divisible by load rows')
            run = _Run(case.inputs, group, load)
            result[f'tilelang-sweep-g{group}-r{load}-store0'] = Implementation(run, args=(), reset=run.reset)
    return result
