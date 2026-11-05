"""Experimental non-FP8 gather: prepare addresses, then issue grouped DMA.

Not registered as the public default. One payload buffer, explicit group barriers.
"""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.kernels.ascend.attention.paged_cache_gather import (
    AscendPagedKVCacheGatherKernel, _vector_cores,
)
from tileops.kernels.kernel_base import Kernel


@functools.lru_cache(maxsize=64)
def _compile(batch, pages, page_size, table_width, width, dtype, starts, blocks, group_tiles):
    tile_rows = 16
    group_rows = tile_rows * group_tiles
    total = T.dynamic("total")

    @T.prim_func
    def main(
        dst: T.Tensor((total, width), dtype),
        cache: T.Tensor((pages * page_size, width), dtype),
        table: T.Tensor((batch, table_width), "int32"),
        cu: T.Tensor((batch + 1,), "int32"),
        start: T.Tensor((batch,) if starts else (batch + 1,), "int32"),
        scale: T.Tensor((pages * page_size, width), dtype),
    ):
        with T.Kernel(blocks) as bx:
            src = T.alloc_shared((group_rows, width), dtype)
            offsets = T.alloc_shared((batch + 1,), "int32")
            beginnings = T.alloc_shared((batch,), "int32")
            # At worst every row is a separate page/request segment. These
            # descriptors are scalar-written/read; payload DMA never reads them.
            addresses = T.alloc_shared((group_rows,), "int64")
            lengths = T.alloc_shared((group_rows,), "int32")
            destinations = T.alloc_shared((group_rows,), "int32")
            T.copy(cu, offsets)
            if starts:
                T.copy(start, beginnings)
            T.ascend_set_flag("MTE2_S", 0)
            T.ascend_wait_flag("MTE2_S", 0)
            # Partition contiguous tile intervals across cores, including tails.
            tiles_per_core = T.ceildiv(T.ceildiv(total, tile_rows), blocks)
            first = bx * tiles_per_core * tile_rows
            end = T.min(total, (bx + 1) * tiles_per_core * tile_rows)
            request = T.alloc_var("int32", init=0)
            lo = T.alloc_var("int32", init=0)
            hi = T.alloc_var("int32", init=batch)
            if first < end:
                while lo < hi:
                    mid = (lo + hi) // 2
                    if offsets[mid + 1] <= first:
                        lo = mid + 1
                    else:
                        hi = mid
                request = lo
                for group in T.serial(T.ceildiv(end - first, group_rows)):
                    row = first + group * group_rows
                    valid = T.min(group_rows, end - row)
                    prepared = T.alloc_var("int32", init=0)
                    segments = T.alloc_var("int32", init=0)
                    # Prepare all addresses before issuing any payload read.
                    while prepared < valid:
                        while offsets[request + 1] <= row + prepared:
                            request = request + 1
                        pos = row + prepared - offsets[request]
                        if starts:
                            pos = pos + beginnings[request]
                        count = T.min(
                            tile_rows - prepared % tile_rows,
                            T.min(valid - prepared, T.min(
                                offsets[request + 1] - row - prepared,
                                page_size - pos % page_size)),
                        )
                        addresses[segments] = T.cast(
                            table[request, pos // page_size], "int64"
                        ) * page_size + pos % page_size
                        lengths[segments] = count
                        destinations[segments] = prepared
                        prepared = prepared + count
                        segments = segments + 1
                    # No per-segment completion barrier: independent UB regions.
                    for segment in T.serial(segments):
                        physical = addresses[segment]
                        target = destinations[segment]
                        count = lengths[segment]
                        if count == tile_rows:
                            T.copy(cache[physical:physical + tile_rows, :],
                                   src[target:target + tile_rows, :])
                        else:
                            T.copy(cache[physical:physical + count, :],
                                   src[target:target + count, :])
                    T.ascend_set_flag("MTE2_MTE3", 0)
                    T.ascend_wait_flag("MTE2_MTE3", 0)
                    if valid == group_rows:
                        T.copy(src, dst[row:row + group_rows, :])
                    else:
                        T.copy(src[0:valid, :], dst[row:row + valid, :])
                    # Complete output before the next group reuses payload UB.
                    T.ascend_set_flag("MTE3_MTE2", 0)
                    T.ascend_wait_flag("MTE3_MTE2", 0)
            T.ascend_set_flag("MTE3_S", 0)
            T.ascend_wait_flag("MTE3_S", 0)

    return tilelang.compile(main, target="ascend", out_idx=[], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    })


class AscendPagedKVCacheGatherBatchKernel(AscendPagedKVCacheGatherKernel):
    """Inherit the public launch checks; replace only the compiled schedule."""

    def __init__(self, call, group_tiles=2):
        reason = self.refusal(call)
        if reason:
            raise ValueError(reason)
        if call.dtype not in (torch.float16, torch.bfloat16) or call.out_dtype != call.dtype:
            raise ValueError("batched gather requires matching FP16/BF16 cache and output")
        if group_tiles not in (2, 4):
            raise ValueError("group_tiles must be 2 or 4")
        Kernel.__init__(self, device_index=call.device.index if call.device is not None else None)
        self.call, self.dtype = call, call.dtype
        self._compiled = _compile(
            call.batch, call.num_pages, call.page_size, call.table_width, call.row_width,
            self.dtype_str, call.has_starts, _vector_cores(self.device_index), group_tiles,
        )
