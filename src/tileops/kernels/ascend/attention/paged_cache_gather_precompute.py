"""Experimental per-core full address precomputation followed by 32/60-row overlapping DMA."""

import functools

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.kernels.ascend.attention.paged_cache_gather import (
    AscendPagedKVCacheGatherKernel, _vector_cores,
)
from tileops.kernels.kernel_base import Kernel


@functools.lru_cache(maxsize=64)
def _compile(batch, pages, page_size, table_width, width, dtype, starts, blocks, total_rows, group_rows, tile_rows):
    buffers, store_l2_cache_ctrl = 2, 0
    max_core_rows = ((total_rows + 16 * blocks - 1) // (16 * blocks)) * 16
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
            # Two disjoint 32-row regions; all physical rows are prepared first.
            src = T.alloc_shared((buffers, group_rows, width), dtype)
            offsets = T.alloc_shared((batch + 1,), "int32")
            beginnings = T.alloc_shared((batch,), "int32")
            row_addresses = T.alloc_shared((max_core_rows,), "int64")
            T.copy(cu, offsets)
            if starts:
                T.copy(start, beginnings)
            T.ascend_set_flag("MTE2_S", 0)
            T.ascend_wait_flag("MTE2_S", 0)
            # Partition contiguous tile intervals across cores, including tails.
            tiles_per_core = T.ceildiv(T.ceildiv(total, 16), blocks)
            first = bx * tiles_per_core * 16
            end = T.min(total, (bx + 1) * tiles_per_core * 16)
            groups = T.ceildiv(T.max(end - first, 0), group_rows)
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
                # All mapping work is inside the timed device kernel, before
                # any payload DMA. Empty requests are skipped by advancing cu.
                for local_row in T.serial(end - first):
                    output_row = first + local_row
                    while offsets[request + 1] <= output_row:
                        request = request + 1
                    position = output_row - offsets[request]
                    if starts:
                        position = position + beginnings[request]
                    row_addresses[local_row] = T.cast(
                        table[request, position // page_size], "int64"
                    ) * page_size + position % page_size
                for group in T.serial(groups):
                    row = first + group * group_rows
                    valid = T.min(group_rows, end - row)
                    if group_rows == 32:
                        if group > 0:
                            previous_row = row - group_rows
                            for slot in T.unroll(buffers):
                                if (group - 1) % buffers == slot:
                                    T.ascend_wait_flag("MTE2_MTE3", slot)
                                    T.copy(src[slot, :, :],
                                           dst[previous_row:previous_row + group_rows, :],
                                           l2_cache_ctrl=store_l2_cache_ctrl)
                                    T.ascend_set_flag("MTE3_MTE2", slot)
                    for slot in T.unroll(buffers):
                        if group % buffers == slot:
                            if group >= buffers:
                                T.ascend_wait_flag("MTE3_MTE2", slot)
                            for segment in T.serial(T.ceildiv(valid, tile_rows)):
                                target = segment * tile_rows
                                physical = row_addresses[group * group_rows + target]
                                contiguous = T.alloc_var("int32", init=1)
                                count = T.min(tile_rows, valid - target)
                                for r in T.serial(count):
                                    if row_addresses[group * group_rows + target + r] != physical + r:
                                        contiguous = 0
                                if contiguous == 1:
                                    T.copy(cache[physical:physical + count, :],
                                           src[slot, target:target + count, :])
                                else:
                                    for r in T.serial(count):
                                        source_row = row_addresses[group * group_rows + target + r]
                                        T.copy(cache[source_row, :], src[slot, target + r, :])
                            T.ascend_set_flag("MTE2_MTE3", slot)
                    if group_rows == 60:
                        if group > 0:
                            previous_row = row - group_rows
                            for slot in T.unroll(buffers):
                                if (group - 1) % buffers == slot:
                                    T.ascend_wait_flag("MTE2_MTE3", slot)
                                    T.copy(src[slot, :, :],
                                           dst[previous_row:previous_row + group_rows, :],
                                           l2_cache_ctrl=store_l2_cache_ctrl)
                                    T.ascend_set_flag("MTE3_MTE2", slot)
                # Only the final group can contain fewer than group_rows rows.
                final_row = first + (groups - 1) * group_rows
                final_valid = end - final_row
                for slot in T.unroll(buffers):
                    if (groups - 1) % buffers == slot:
                        T.ascend_wait_flag("MTE2_MTE3", slot)
                        if final_valid == group_rows:
                            T.copy(src[slot, :, :], dst[final_row:final_row + group_rows, :],
                                   l2_cache_ctrl=store_l2_cache_ctrl)
                        else:
                            T.copy(src[slot, 0:final_valid, :], dst[final_row:end, :],
                                   l2_cache_ctrl=store_l2_cache_ctrl)
                        T.ascend_set_flag("MTE3_MTE2", slot)
                for slot in T.unroll(buffers):
                    if groups > slot:
                        T.ascend_wait_flag("MTE3_MTE2", slot)
            T.ascend_set_flag("MTE3_S", 0)
            T.ascend_wait_flag("MTE3_S", 0)

    return tilelang.compile(main, target="ascend", out_idx=[], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    })


class AscendPagedKVCacheGatherPrecomputeKernel(AscendPagedKVCacheGatherKernel):
    """Inherit the public launch checks; replace only the compiled schedule."""

    def __init__(self, call, total_rows, group_rows=32, tile_rows=4):
        reason = self.refusal(call)
        if reason:
            raise ValueError(reason)
        if call.dtype not in (torch.float16, torch.bfloat16) or call.out_dtype != call.dtype:
            raise ValueError("precomputed gather requires matching FP16/BF16 tensors")
        if (group_rows, tile_rows) not in ((32, 4), (60, 1)):
            raise ValueError("supported precompute layouts: (32,4), (60,1)")
        if total_rows <= 0:
            raise ValueError("total_rows must be positive")
        Kernel.__init__(self, device_index=call.device.index if call.device is not None else None)
        self.call, self.dtype, self.total_rows = call, call.dtype, total_rows
        self._compiled = _compile(
            call.batch, call.num_pages, call.page_size, call.table_width, call.row_width,
            self.dtype_str, call.has_starts, _vector_cores(self.device_index), total_rows, group_rows, tile_rows,
        )

    def forward(self, dst, cache, block_table, cu_seq_lens, seq_starts=None, scale=None):
        if dst.shape[0] != self.total_rows:
            raise ValueError("output row count differs from precomputed address buffer specialization")
        return super().forward(dst, cache, block_table, cu_seq_lens, seq_starts, scale)
