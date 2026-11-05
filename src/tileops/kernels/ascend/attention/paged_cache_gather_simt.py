"""Experimental direct GM SIMT gather, without payload UB or DMA group events."""

import functools

import tilelang
import tilelang.ascend.language as T

from tileops.kernels.ascend.attention.paged_cache_gather import (
    AscendPagedKVCacheGatherKernel, _vector_cores,
)
from tileops.kernels.kernel_base import Kernel


@functools.lru_cache(maxsize=64)
def _compile(batch, pages, page_size, table_width, width, dtype, out_dtype, starts, blocks):
    total = T.dynamic("total")
    tile_rows = 16
    quantized = dtype == "float8_e4m3fn"

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
            for task in T.serial(T.ceildiv(T.ceildiv(total, tile_rows), blocks)):
                base = (task * blocks + bx) * tile_rows
                with T.SimtVF(threads=256):
                    # Flattening makes adjacent logical lanes access adjacent
                    # payload elements. All metadata remains live device data.
                    for element in T.Parallel(tile_rows * width):
                        row = base + element // width
                        column = element % width
                        if row < total:
                            lo = T.alloc_var("int32", init=0)
                            hi = T.alloc_var("int32", init=batch)
                            while lo < hi:
                                mid = (lo + hi) // 2
                                if cu[mid + 1] <= row:
                                    lo = mid + 1
                                else:
                                    hi = mid
                            position = T.alloc_var("int32", init=row - cu[lo])
                            if starts:
                                position = position + start[lo]
                            physical = T.cast(table[lo, position // page_size], "int64") * page_size + position % page_size
                            if quantized:
                                dst[row, column] = T.cast(cache[physical, column], "float32") * scale[0]
                            else:
                                dst[row, column] = cache[physical, column]

    return tilelang.compile(main, target="ascend", out_idx=[])


class AscendPagedKVCacheGatherSimtKernel(AscendPagedKVCacheGatherKernel):
    """Reuse host validation and flat launch signature, replace device schedule."""

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
