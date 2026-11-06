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
@pytest.mark.parametrize("group_rows", [16, 32, 48])
@pytest.mark.parametrize("load_rows", [1, 2, 4, 8, 16])
def test_sweep_manifest(case, group_rows, load_rows):
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires Ascend and --tileops-device=npu")
    workload = GatherWorkload(case)
    args = workload.gen_inputs()
    kernel = AscendPagedKVCacheGatherSweepKernel(
        dataclasses.replace(_request(case), device=args[0].device),
        group_rows=group_rows, tile_rows=load_rows)
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
