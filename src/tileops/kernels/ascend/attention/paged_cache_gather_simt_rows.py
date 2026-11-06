"""Experimental SIMT gather: map each row once, explicit contiguous lane ownership."""
import functools

import tilelang
import tilelang.ascend.language as T

from tileops.kernels.ascend.attention.paged_cache_gather import AscendPagedKVCacheGatherKernel, _vector_cores
from tileops.kernels.kernel_base import Kernel


@functools.lru_cache(maxsize=64)
def _compile(batch, pages, page_size, table_width, width, dtype, out_dtype, starts, blocks):
    total = T.dynamic("total")
    tile_rows, threads, lanes = 32, 256, 32
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
            offsets = T.alloc_shared((batch + 1,), "int32")
            beginnings = T.alloc_shared((batch,), "int32")
            scale_ub = T.alloc_shared((1,), "float32")
            physical_rows = T.alloc_shared((tile_rows,), "int64")
            T.copy(cu, offsets)
            if starts:
                T.copy(start, beginnings)
            if quantized:
                T.copy(scale, scale_ub)
            for task in T.serial(T.ceildiv(T.ceildiv(total, tile_rows), blocks)):
                base = (task * blocks + bx) * tile_rows
                # Only tile_rows logical iterations: one mapping per valid row,
                # not one mapping per output element. Compiler manages UB deps.
                with T.SimtVF(threads=threads):
                    for rr in T.Parallel(tile_rows):
                        row = base + rr
                        if row < total:
                            lo = T.alloc_var("int32", init=0)
                            hi = T.alloc_var("int32", init=batch)
                            while lo < hi:
                                mid = (lo + hi) // 2
                                if offsets[mid + 1] <= row:
                                    lo = mid + 1
                                else:
                                    hi = mid
                            position = T.alloc_var("int32", init=row - offsets[lo])
                            if starts:
                                position = position + beginnings[lo]
                            physical_rows[rr] = T.cast(table[lo, position // page_size], "int64") * page_size + position % page_size
                with T.SimtVF(threads=threads):
                    tx = T.get_thread_binding()
                    # Adjacent 32 threads own adjacent columns of one row.
                    # Hoist row-address loads out of the per-column loop.
                    for row_round in T.serial(T.ceildiv(tile_rows, threads // lanes)):
                        rr = row_round * (threads // lanes) + tx // lanes
                        row = base + rr
                        if rr < tile_rows and row < total:
                            physical = physical_rows[rr]
                            multiplier = T.alloc_var("float32", init=1.0)
                            if quantized:
                                multiplier = scale_ub[0]
                            for column_round in T.serial(T.ceildiv(width, lanes)):
                                column = column_round * lanes + tx % lanes
                                if column < width:
                                    if quantized:
                                        dst[row, column] = T.cast(cache[physical, column], "float32") * multiplier
                                    else:
                                        dst[row, column] = cache[physical, column]

    return tilelang.compile(main, target="ascend", out_idx=[])


class AscendPagedKVCacheGatherSimtRowsKernel(AscendPagedKVCacheGatherKernel):
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
