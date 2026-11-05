"""Experimental automatic schedules: all manifest cases and live metadata."""

import dataclasses

import pytest
import torch

from tests.ops.test_paged_cache_gather_ascend import _cases, _request
from tileops.kernels.ascend.attention.paged_cache_gather_auto import AscendPagedKVCacheGatherAutoKernel
from workloads.attention.paged_cache_gather import PagedKVCacheGatherCall as GatherWorkload
from workloads.device import run_device, run_device_available


@pytest.mark.smoke
@pytest.mark.parametrize("stages,tile_rows", [(1, 16), (2, 16), (2, 32)])
@pytest.mark.parametrize("case", _cases(), ids=lambda case: case.case_id)
def test_automatic_schedule(case, stages, tile_rows):
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires Ascend and --tileops-device=npu")
    workload = GatherWorkload(case)
    args = list(workload.gen_inputs())
    call = dataclasses.replace(_request(case), device=args[0].device)
    if tile_rows > call.page_size:
        pytest.skip("32-row candidate requires pages with at least 32 rows")
    kernel = AscendPagedKVCacheGatherAutoKernel(call, stages=stages, tile_rows=tile_rows)
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
