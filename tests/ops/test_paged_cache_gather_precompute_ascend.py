"""Precomputed-address correctness including live metadata replay."""
import dataclasses

import pytest
import torch

from tests.ops.test_paged_cache_gather_ascend import _cases, _request
from tileops.kernels.ascend.attention.paged_cache_gather_precompute import AscendPagedKVCacheGatherPrecomputeKernel
from workloads.attention.paged_cache_gather import PagedKVCacheGatherCall as GatherWorkload
from workloads.device import run_device, run_device_available


@pytest.mark.smoke
@pytest.mark.parametrize("case", [c for c in _cases() if "fp8" not in c.case_id], ids=lambda c: c.case_id)
def test_precompute_manifest(case):
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires Ascend and --tileops-device=npu")
    workload = GatherWorkload(case)
    args = workload.gen_inputs()
    kernel = AscendPagedKVCacheGatherPrecomputeKernel(
        dataclasses.replace(_request(case), device=args[0].device), total_rows=args[0].shape[0])
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
