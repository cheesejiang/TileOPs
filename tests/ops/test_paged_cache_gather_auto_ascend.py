"""Experimental automatic schedules: all manifest cases and live metadata."""

import dataclasses

import pytest
import torch

from tests.ops.test_paged_cache_gather_ascend import _cases, _request
from tileops.kernels.ascend.attention.paged_cache_gather_auto import AscendPagedKVCacheGatherAutoKernel
from tileops.kernels.attention.paged_cache_gather import PagedKVCacheGatherCall
from workloads.attention.paged_cache_gather import PagedKVCacheGatherCall as GatherWorkload
from workloads.attention.paged_cache_gather import PagedKVCacheGatherWorkload
from workloads.device import run_device, run_device_available


@pytest.mark.smoke
@pytest.mark.parametrize("stages,tile_rows,segmented", [
    (1, 16, False), (2, 16, False), (2, 32, False), (2, 16, True),
])
@pytest.mark.parametrize("case", _cases(), ids=lambda case: case.case_id)
def test_automatic_schedule(case, stages, tile_rows, segmented):
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires Ascend and --tileops-device=npu")
    workload = GatherWorkload(case)
    args = list(workload.gen_inputs())
    call = dataclasses.replace(_request(case), device=args[0].device)
    if tile_rows > call.page_size:
        pytest.skip("32-row candidate requires pages with at least 32 rows")
    kernel = AscendPagedKVCacheGatherAutoKernel(
        call, stages=stages, tile_rows=tile_rows, segmented=segmented)
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
@pytest.mark.parametrize("dtype,out_dtype", [
    (torch.bfloat16, None), (torch.float8_e4m3fn, torch.bfloat16),
])
def test_segments_cross_empty_requests_and_pages(dtype, out_dtype):
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires Ascend and --tileops-device=npu")
    # One tile spans several request boundaries, including empty requests.
    # The final tile is partial, and several starts are one row before a page end.
    workload = PagedKVCacheGatherWorkload(
        [0, 1, 0, 2, 13, 0, 19], (576,), 16, dtype, out_dtype,
        starts=[0, 15, 0, 15, 7, 0, 15],
    )
    args = workload.gen_inputs()
    dst, cache, table, cu, starts, scale = args
    call = PagedKVCacheGatherCall(
        batch=table.shape[0], num_pages=cache.shape[0], page_size=16,
        table_width=table.shape[1], row_width=576, dtype=dtype,
        out_dtype=dst.dtype, has_starts=True, device=dst.device,
    )
    expected = workload.ref_program(*args)
    kernel = AscendPagedKVCacheGatherAutoKernel(call, stages=2, segmented=True)
    kernel(dst, cache, table, cu, starts, scale)
    torch.testing.assert_close(dst, expected, rtol=0, atol=0, equal_nan=True)
