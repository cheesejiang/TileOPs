"""Ascend gather: manifest coverage, live metadata and write-only output views."""

import dataclasses
import math

import pytest
import torch

from tileops.kernels.ascend.attention.paged_cache_gather import AscendPagedKVCacheGatherKernel
from tileops.kernels.attention.paged_cache_gather import PagedKVCacheGatherCall
from tileops.manifest import load_adts, load_manifest, load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
from tileops.ops import PagedKVCacheGatherFwdOp
from workloads.attention.paged_cache_gather import PagedKVCacheGatherCall as GatherWorkload
from workloads.device import run_device, run_device_available


def _cases():
    name = "PagedKVCacheGatherFwdOp"
    plan = entry_plan(name, load_manifest()[name], load_adts(), resolve=False)
    return [instantiate(plan, row, dtype) for row in load_workloads(name)
            for dtype in row["dtype_cases"]]


def _request(case):
    i = case.indices
    return PagedKVCacheGatherCall(
        batch=i["B"], num_pages=i["NP"], page_size=i["PS"], table_width=i["W"],
        row_width=math.prod(i["E"]),
        dtype=getattr(torch, case.tensors["cache"][1]),
        out_dtype=getattr(torch, case.tensors["dst"][1]),
        has_starts=case.present("seq_starts"),
    )


@pytest.mark.smoke
def test_manifest_coverage():
    cases = _cases()
    assert len(cases) == 7
    for case in cases:
        assert AscendPagedKVCacheGatherKernel.refusal(_request(case)) is None, case.case_id
    PagedKVCacheGatherFwdOp()


@pytest.mark.smoke
@pytest.mark.parametrize("changes", [
    {"page_size": 7}, {"row_width": 577}, {"dtype": torch.float32},
    {"batch": 257}, {"table_width": 0}, {"out_dtype": torch.float32},
])
def test_refuses_unsupported_geometry(changes):
    call = dataclasses.replace(_request(_cases()[0]), **changes)
    assert AscendPagedKVCacheGatherKernel.refusal(call)


@pytest.mark.smoke
@pytest.mark.parametrize("case", _cases(), ids=lambda case: case.case_id)
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
def test_public_path(case, strided):
    if torch.device(run_device()).type != "npu" or not run_device_available():
        pytest.skip("requires Ascend and --tileops-device=npu")
    workload = GatherWorkload(case)
    args = list(workload.gen_inputs())
    backing = None
    if strided:
        dst = args[0]
        backing = torch.full((*dst.shape[:-1], dst.shape[-1] * 2), -19,
                             dtype=dst.dtype, device=dst.device)
        args[0] = backing[..., ::2]
    op = PagedKVCacheGatherFwdOp(out_dtype=getattr(torch, case.params["out_dtype"])
                               if case.params["out_dtype"] else None)
    for replay in range(2):
        # Same shapes and op instance, different device metadata and scale.
        if replay:
            args[2].copy_(args[2].flip(-1))
            if args[4] is not None:
                args[4].zero_()
            if args[5] is not None:
                args[5].fill_(0.75)
        expected = workload.ref_program(*args)
        args[0].fill_(float("nan"))
        assert op(*args) is None
        torch.testing.assert_close(args[0], expected, rtol=0, atol=0, equal_nan=True)
        if backing is not None:
            assert bool((backing[..., 1::2] == -19).all())
