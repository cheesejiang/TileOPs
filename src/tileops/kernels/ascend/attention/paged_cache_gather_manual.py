"""Historical manual gather schedule; unregistered profiling experiment."""

import functools

import tilelang
import tilelang.ascend.language as T

from tileops.kernels.ascend.attention.paged_cache_gather import AscendPagedKVCacheGatherKernel, _vector_cores
from tileops.kernels.kernel_base import Kernel


@functools.lru_cache(maxsize=64)
def _compile(batch, pages, page_size, table_width, width, dtype, out_dtype, starts, blocks):
    # Whole-page DMA for the large MLA cases; smaller tiles limit the cost
    # of unaligned starts while retaining a full page for 16-row Llama pages.
    tile_rows = 16 if starts else page_size
    quantized = dtype == "float8_e4m3fn"
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
            src = T.alloc_shared((tile_rows, width), dtype)
            converted = T.alloc_shared((tile_rows, width), out_dtype)
            # Metadata is small; cache it once per core, outside the tile loop.
            offsets = T.alloc_shared((batch + 1,), "int32")
            beginnings = T.alloc_shared((batch,), "int32")
            scale_ub = T.alloc_shared((1,), "float32")
            T.copy(cu, offsets)
            if starts:
                T.copy(start, beginnings)
            if quantized:
                T.copy(scale, scale_ub)
            T.ascend_set_flag("MTE2_S", 0)
            T.ascend_wait_flag("MTE2_S", 0)
            for task in T.serial(T.ceildiv(T.ceildiv(total, tile_rows), blocks)):
                row = (task * blocks + bx) * tile_rows
                if row < total:
                    request = T.alloc_var("int32", init=0)
                    # At most 256 requests in the manifest; binary search skips
                    # repeated offsets belonging to empty requests.
                    lo = T.alloc_var("int32", init=0)
                    hi = T.alloc_var("int32", init=batch)
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if offsets[mid + 1] <= row:
                            lo = mid + 1
                        else:
                            hi = mid
                    request = lo
                    pos = row - offsets[request]
                    if starts:
                        pos = pos + beginnings[request]
                    if row + tile_rows <= offsets[request + 1] and pos % page_size + tile_rows <= page_size:
                        physical = T.cast(table[request, pos // page_size], "int64") * page_size + pos % page_size
                        T.copy(cache[physical:physical + tile_rows, :], src)
                    else:
                        # Unaligned starts and request/page tails are uncommon;
                        # load each valid row, leaving all full tiles coalesced.
                        for r in T.serial(tile_rows):
                            if row + r < total:
                                while offsets[request + 1] <= row + r:
                                    request = request + 1
                                p = row + r - offsets[request]
                                if starts:
                                    p = p + beginnings[request]
                                physical = T.cast(table[request, p // page_size], "int64") * page_size + p % page_size
                                T.copy(cache[physical, :], src[r, :])
                    if quantized:
                        T.ascend_set_flag("MTE2_V", 0)
                        T.ascend_wait_flag("MTE2_V", 0)
                        with T.SimtVF(threads=256):
                            for r, c in T.Parallel(tile_rows, width):
                                if row + r < total:
                                    converted[r, c] = T.cast(src[r, c], "float32") * scale_ub[0]
                        T.ascend_set_flag("V_MTE3", 0)
                        T.ascend_wait_flag("V_MTE3", 0)
                    else:
                        T.ascend_set_flag("MTE2_MTE3", 0)
                        T.ascend_wait_flag("MTE2_MTE3", 0)
                    if row + tile_rows <= total:
                        if quantized:
                            T.copy(converted, dst[row:row + tile_rows, :])
                        else:
                            T.copy(src, dst[row:row + tile_rows, :])
                    else:
                        for r in T.serial(tile_rows):
                            if row + r < total:
                                if quantized:
                                    T.copy(converted[r, :], dst[row + r, :])
                                else:
                                    T.copy(src[r, :], dst[row + r, :])
                    # A subsequent tile must not overwrite an in-flight DMA.
                    T.ascend_set_flag("MTE3_MTE2", 0)
                    T.ascend_wait_flag("MTE3_MTE2", 0)
                    if quantized:
                        T.ascend_set_flag("MTE3_V", 0)
                        T.ascend_wait_flag("MTE3_V", 0)
            T.ascend_set_flag("MTE3_S", 0)
            T.ascend_wait_flag("MTE3_S", 0)

    return tilelang.compile(main, target="ascend", out_idx=[], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    })


class AscendPagedKVCacheGatherManualKernel(AscendPagedKVCacheGatherKernel):
    """Historical manual schedule, unregistered, for explicit profiling only."""

    def __init__(self, call):
        reason = self.refusal(call)
        if reason:
            raise ValueError(reason)
        Kernel.__init__(self, device_index=call.device.index if call.device is not None else None)
        self.call, self.dtype = call, call.dtype
        self._compiled = _compile(
            call.batch, call.num_pages, call.page_size, call.table_width, call.row_width,
            self.dtype_str, self.dtype_to_str(call.out_dtype), call.has_starts,
            _vector_cores(self.device_index),
        )

