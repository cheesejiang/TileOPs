"""Grouped DMA: exact results, metadata replay and split request/page tails."""
import dataclasses

import pytest
import torch

from tests.ops.test_paged_cache_gather_ascend import _cases, _request
from tileops.kernels.ascend.attention.paged_cache_gather_batch import AscendPagedKVCacheGatherBatchKernel
from tileops.kernels.attention.paged_cache_gather import PagedKVCacheGatherCall
from workloads.attention.paged_cache_gather import PagedKVCacheGatherCall as GatherWorkload
from workloads.attention.paged_cache_gather import PagedKVCacheGatherWorkload
from workloads.device import run_device, run_device_available


def _require_npu():
    if torch.device(run_device()).type != 'npu' or not run_device_available():
        pytest.skip('requires Ascend and --tileops-device=npu')


@pytest.mark.smoke
@pytest.mark.parametrize('group_tiles,double_buffer,tile_rows', [(2, False, 16), (4, False, 16), (2, True, 16), (4, True, 16), (16, False, 4), pytest.param(8, True, 4, id="overlap32-r4")])
@pytest.mark.parametrize('case', [c for c in _cases() if 'fp8' not in c.case_id], ids=lambda c: c.case_id)
@pytest.mark.parametrize("store_l2_cache_ctrl", [4, 0])
def test_batch_manifest(case, group_tiles, double_buffer, tile_rows, store_l2_cache_ctrl):
    _require_npu()
    workload = GatherWorkload(case)
    args = workload.gen_inputs()
    call = dataclasses.replace(_request(case), device=args[0].device)
    kernel = AscendPagedKVCacheGatherBatchKernel(
        call, group_tiles, double_buffer=double_buffer, tile_rows=tile_rows,
        store_l2_cache_ctrl=store_l2_cache_ctrl)
    for replay in range(2):
        if replay:
            args[2].copy_(args[2].flip(-1))
            if args[4] is not None:
                args[4].zero_()
        expected = workload.ref_program(*args)
        args[0].fill_(float('nan'))
        kernel(*args)
        torch.testing.assert_close(args[0], expected, rtol=0, atol=0)


@pytest.mark.smoke
@pytest.mark.parametrize('group_tiles,double_buffer,tile_rows', [(2, False, 16), (4, False, 16), (2, True, 16), (4, True, 16), (16, False, 4), pytest.param(8, True, 4, id="overlap32-r4")])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_batch_split_boundaries(group_tiles, double_buffer, tile_rows, dtype):
    _require_npu()
    workload = PagedKVCacheGatherWorkload(
        [0, 1, 0, 2, 13, 0, 19], (576,), 16, dtype, None,
        starts=[0, 15, 0, 15, 7, 0, 15],
    )
    args = workload.gen_inputs()
    dst, cache, table, cu, starts, scale = args
    call = PagedKVCacheGatherCall(
        batch=table.shape[0], num_pages=cache.shape[0], page_size=16,
        table_width=table.shape[1], row_width=576, dtype=dtype,
        out_dtype=dtype, has_starts=True, device=dst.device,
    )
    expected = workload.ref_program(*args)
    AscendPagedKVCacheGatherBatchKernel(call, group_tiles, double_buffer=double_buffer, tile_rows=tile_rows)(*args)
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
