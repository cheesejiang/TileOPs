"""Parameterized overlap schedule for FP16/BF16 and scaled FP8 gather.

Not registered as the public default. One or two payload buffers, explicit per-buffer events.
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
def _compile(batch, pages, page_size, table_width, width, dtype, starts, blocks, group_rows, tile_rows, out_dtype):
    buffers = 2
    quantized = dtype == "float8_e4m3fn"
    ready_event = "V_MTE3" if quantized else "MTE2_MTE3"
    store_l2_cache_ctrl = 0
    total = T.dynamic("total")

    @T.prim_func
    def main(
        dst: T.Tensor((total, width), out_dtype),
        cache: T.Tensor((pages * page_size, width), dtype),
        table: T.Tensor((batch, table_width), "int32"),
        cu: T.Tensor((batch + 1,), "int32"),
        start: T.Tensor((batch,) if starts else (batch + 1,), "int32"),
        scale: T.Tensor((1,) if quantized else (pages * page_size, width),
                        "float32" if quantized else dtype),
    ):
        with T.Kernel(blocks) as bx:
            # A single allocation split into disjoint per-group UB regions.
            # batch4: A=rows 0..63, B=rows 64..127 (flattened).
            src = T.alloc_shared((buffers, group_rows, width), dtype)
            converted = T.alloc_shared((buffers, group_rows, width) if quantized else (1, 1, 1), out_dtype)
            scale_ub = T.alloc_shared((1,), "float32")
            offsets = T.alloc_shared((batch + 1,), "int32")
            beginnings = T.alloc_shared((batch,), "int32")
            # At worst every row is a separate page/request segment. These
            # descriptors are scalar-written/read; payload DMA never reads them.
            addresses = T.alloc_shared((group_rows,), "int64")
            lengths = T.alloc_shared((group_rows,), "int32")
            destinations = T.alloc_shared((group_rows,), "int32")
            if quantized:
                T.copy(scale, scale_ub)
            T.copy(cu, offsets)
            if starts:
                T.copy(start, beginnings)
            T.ascend_set_flag("MTE2_S", 0)
            T.ascend_wait_flag("MTE2_S", 0)
            # Partition contiguous tile intervals across cores, including tails.
            tiles_per_core = T.ceildiv(T.ceildiv(total, 16), blocks)
            first = bx * tiles_per_core * 16
            end = T.min(total, (bx + 1) * tiles_per_core * 16)
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
                groups = T.ceildiv(end - first, group_rows)
                for group in T.serial(groups):
                    row = first + group * group_rows
                    valid = T.min(group_rows, end - row)
                    # Issue the previous output BEFORE preparing current
                    # metadata (which contains dependent GM scalar reads).
                    # Otherwise metadata preparation can delay MTE3 launch.
                    if group > 0:
                        previous_row = row - group_rows
                        for slot in T.unroll(buffers):
                            if (group - 1) % buffers == slot:
                                T.ascend_wait_flag(ready_event, slot)
                                if quantized:
                                    T.copy(converted[slot, :, :],
                                           dst[previous_row:previous_row + group_rows, :], l2_cache_ctrl=store_l2_cache_ctrl)
                                else:
                                    T.copy(src[slot, :, :],
                                           dst[previous_row:previous_row + group_rows, :], l2_cache_ctrl=store_l2_cache_ctrl)
                                T.ascend_set_flag("MTE3_MTE2", slot)
                                if quantized:
                                    T.ascend_set_flag("MTE3_V", slot)
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
                    # Constant event IDs and buffer slices in each unrolled branch.
                    for slot in T.unroll(buffers):
                        if group % buffers == slot:
                            if group >= buffers:
                                T.ascend_wait_flag("MTE3_MTE2", slot)
                            for segment in T.serial(segments):
                                physical = addresses[segment]
                                target = destinations[segment]
                                count = lengths[segment]
                                if count == tile_rows:
                                    T.copy(cache[physical:physical + tile_rows, :],
                                           src[slot, target:target + tile_rows, :])
                                else:
                                    T.copy(cache[physical:physical + count, :],
                                           src[slot, target:target + count, :])
                            if quantized:
                                if group >= buffers:
                                    T.ascend_wait_flag("MTE3_V", slot)
                                T.ascend_set_flag("MTE2_V", slot)
                                T.ascend_wait_flag("MTE2_V", slot)
                                with T.SimtVF(threads=256):
                                    for rr, cc in T.Parallel(group_rows, width):
                                        if rr < valid:
                                            converted[slot, rr, cc] = T.cast(src[slot, rr, cc], "float32") * scale_ub[0]
                            T.ascend_set_flag(ready_event, slot)
                # Drain the final group; only this group may be partial.
                final_row = first + (groups - 1) * group_rows
                final_valid = end - final_row
                for slot in T.unroll(buffers):
                    if (groups - 1) % buffers == slot:
                        T.ascend_wait_flag(ready_event, slot)
                        if quantized:
                            if final_valid == group_rows:
                                T.copy(converted[slot, :, :], dst[final_row:final_row + group_rows, :], l2_cache_ctrl=store_l2_cache_ctrl)
                            else:
                                T.copy(converted[slot, 0:final_valid, :], dst[final_row:end, :], l2_cache_ctrl=store_l2_cache_ctrl)
                        else:
                            if final_valid == group_rows:
                                T.copy(src[slot, :, :], dst[final_row:final_row + group_rows, :], l2_cache_ctrl=store_l2_cache_ctrl)
                            else:
                                T.copy(src[slot, 0:final_valid, :], dst[final_row:end, :], l2_cache_ctrl=store_l2_cache_ctrl)
                        T.ascend_set_flag("MTE3_MTE2", slot)
                        if quantized:
                            T.ascend_set_flag("MTE3_V", slot)
                # Consume the last release event for each used buffer.
                for slot in T.unroll(buffers):
                    if groups > slot:
                        T.ascend_wait_flag("MTE3_MTE2", slot)
                        if quantized:
                            T.ascend_wait_flag("MTE3_V", slot)
            T.ascend_set_flag("MTE3_S", 0)
            T.ascend_wait_flag("MTE3_S", 0)

    return tilelang.compile(main, target="ascend", out_idx=[], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    })


class AscendPagedKVCacheGatherSweepKernel(AscendPagedKVCacheGatherKernel):
    """Inherit the public launch checks; replace only the compiled schedule."""

    def __init__(self, call, group_rows=32, tile_rows=4):
        reason = self.refusal(call)
        if reason:
            raise ValueError(reason)
        if group_rows not in (16, 32, 48, 64) or tile_rows not in (1, 2, 4, 8, 16):
            raise ValueError("unsupported sweep dimensions")
        if group_rows % tile_rows:
            raise ValueError("group_rows must be divisible by tile_rows")
        Kernel.__init__(self, device_index=call.device.index if call.device is not None else None)
        self.call, self.dtype = call, call.dtype
        self._compiled = _compile(
            call.batch, call.num_pages, call.page_size, call.table_width, call.row_width,
            self.dtype_str, call.has_starts, _vector_cores(self.device_index),
            group_rows, tile_rows, self.dtype_to_str(call.out_dtype),
        )
