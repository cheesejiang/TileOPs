"""Parameterized overlap correctness including live metadata replay."""
import dataclasses

import pytest
import torch

from tests.ops.test_paged_cache_gather_ascend import _cases, _request
from tileops.kernels.ascend.attention.paged_cache_gather_sweep import AscendPagedKVCacheGatherSweepKernel
from workloads.attention.paged_cache_gather import PagedKVCacheGatherCall as GatherWorkload
from workloads.device import run_device, run_device_available


@pytest.mark.smoke
@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.case_id)
@pytest.mark.parametrize("cache_page", [False, True])
def test_pagecache_manifest(case, cache_page):
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires Ascend and --tileops-device=npu")
    workload = GatherWorkload(case)
    args = workload.gen_inputs()
    kernel = AscendPagedKVCacheGatherSweepKernel(
        dataclasses.replace(_request(case), device=args[0].device),
        group_rows=32, tile_rows=8, cache_page=cache_page)
    for replay in range(2):
        if replay:
            args[2].copy_(args[2].flip(-1))
            if args[4] is not None:
                args[4].zero_()
            if args[5] is not None:
                args[5].fill_(0.75)
        expected = workload.ref_program(*args)
        args[0].fill_(float("nan"))
        kernel(*args)
        torch.testing.assert_close(args[0], expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_pagecache_request_key_empty_segments_and_replay(dtype):
    from workloads.attention.paged_cache_gather import PagedKVCacheGatherWorkload
    from tileops.kernels.attention.paged_cache_gather import PagedKVCacheGatherCall

    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires Ascend and --tileops-device=npu")
    workload = PagedKVCacheGatherWorkload(
        [0, 9, 0, 7, 41, 0, 19], (576,), 16, dtype,
        torch.bfloat16 if dtype == torch.float8_e4m3fn else None,
        starts=[0, 0, 0, 0, 15, 0, 1],
    )
    args = workload.gen_inputs()
    dst, cache, table, cu, starts, scale = args
    call = PagedKVCacheGatherCall(
        batch=table.shape[0], num_pages=cache.shape[0], page_size=16,
        table_width=table.shape[1], row_width=576, dtype=dtype,
        out_dtype=dst.dtype, has_starts=True, device=dst.device,
    )
    kernel = AscendPagedKVCacheGatherSweepKernel(call, 32, 8, cache_page=True)
    for replay in range(2):
        if replay:
            table.copy_(table.flip(-1))
            starts.zero_()
        expected = workload.ref_program(*args)
        dst.fill_(float('nan'))
        kernel(*args)
        torch.testing.assert_close(dst, expected, rtol=0, atol=0, equal_nan=True)
