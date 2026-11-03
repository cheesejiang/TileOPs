"""Persistent, page-aware cache gathering on Ascend 950.

Request lengths, page IDs, starts and scale remain device data. Output tiles
provide balanced work even when one request contains most of the tokens.
"""

import functools
import math
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.attention.paged_cache_gather import (
    PagedKVCacheGatherCall,
    PagedKVCacheGatherFwdInterface,
)
from tileops.kernels.kernel_base import Kernel


def _vector_cores(device_index):
    from tilelang.carver.arch.driver import get_num_vector_cores

    count = get_num_vector_cores(device_id=device_index)
    if not count:
        raise RuntimeError("cannot determine Ascend vector core count")
    return int(count)


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


class AscendPagedKVCacheGatherKernel(Kernel, PagedKVCacheGatherFwdInterface):
    """Gather supported row geometries without copying metadata to the host."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: PagedKVCacheGatherCall):
        if call.page_size not in (16, 64) or call.row_width not in (576, 1024):
            return "requires page size 16/64 and row width 576/1024"
        if not 0 < call.batch <= 256 or min(call.num_pages, call.table_width) <= 0:
            return "requires 1..256 requests and nonempty page pool/table"
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
            return "requires FP16, BF16 or FP8 E4M3FN cache"
        if call.dtype == torch.float8_e4m3fn:
            if call.out_dtype not in (torch.float16, torch.bfloat16):
                return "FP8 requires FP16/BF16 destination"
        elif call.out_dtype != call.dtype:
            return "unquantized gather preserves dtype"
        return super().refusal(call)

    def __init__(self, call):
        reason = self.refusal(call)
        if reason:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self._compiled = _compile(
            call.batch, call.num_pages, call.page_size, call.table_width, call.row_width,
            self.dtype_str, self.dtype_to_str(call.out_dtype), call.has_starts,
            _vector_cores(self.device_index),
        )

    def forward(self, dst, cache, block_table, cu_seq_lens, seq_starts=None, scale=None):
        tensors = dict(dst=dst, cache=cache, block_table=block_table, cu_seq_lens=cu_seq_lens)
        if seq_starts is not None:
            tensors["seq_starts"] = seq_starts
        if scale is not None:
            tensors["scale"] = scale
        self._require_devices(**tensors)
        if any(t.device != dst.device for t in tensors.values()):
            raise ValueError("every gather tensor must be on the same NPU")
        c = self.call
        if (seq_starts is not None) != c.has_starts or (scale is not None) != (c.dtype == torch.float8_e4m3fn):
            raise ValueError("optional inputs do not match gather specialization")
        if (cache.shape[:2] != (c.num_pages, c.page_size)
                or tuple(dst.shape[1:]) != tuple(cache.shape[2:])
                or math.prod(dst.shape[1:]) != c.row_width
                or cache.dtype != c.dtype or dst.dtype != c.out_dtype):
            raise ValueError("cache/destination does not match gather specialization")
        metadata = [(block_table, (c.batch, c.table_width), torch.int32),
                    (cu_seq_lens, (c.batch + 1,), torch.int32)]
        if seq_starts is not None:
            metadata.append((seq_starts, (c.batch,), torch.int32))
        if scale is not None:
            metadata.append((scale, (1,), torch.float32))
        if any(tuple(t.shape) != shape or t.dtype != dtype for t, shape, dtype in metadata):
            raise ValueError("gather metadata shape/dtype does not match specialization")
        if any(not t.is_contiguous() for t in tensors.values()):
            raise ValueError("kernel inputs must be contiguous; use the public Op for views")
        if not dst.numel():
            return None
        flat_cache = cache.view(c.num_pages * c.page_size, c.row_width)
        self._compiled(dst.view(dst.shape[0], c.row_width), flat_cache, block_table,
                       cu_seq_lens, seq_starts if c.has_starts else cu_seq_lens,
                       scale if scale is not None else flat_cache)
        return None


register_kernel_type("PagedKVCacheGatherFwdOp", "ascend", AscendPagedKVCacheGatherKernel)
