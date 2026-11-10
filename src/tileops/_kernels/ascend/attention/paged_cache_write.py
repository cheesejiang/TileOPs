"""Paged K/V scatter for Ascend 950.

Aligned rows use batched GM->UB reads and per-slot UB->GM writes. Small
decode calls use one row per core; larger calls amortize input DMA and VF
entry over several rows. The original column-tiled kernel handles other widths.
"""

import functools
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._kernels.attention.call_spec import PagedKVCacheWriteFwdInterface
from tileops._kernels.kernel_base import Kernel


@T.macro
def _quantize_rows(raw, packed, scale, elements: int):
    """Exact division and saturated E4M3FN, including signed NaN bytes.

    UNPK_B16 puts 64 input halves into the low halves of 64 b32 lanes.
    The FP8 conversion leaves its result in byte zero of each b32 lane;
    PK4_B32 compacts these bytes into 64 consecutive output bytes.
    """
    with T.SimdVF():
        full = T.simd.pset(32)
        divisor = T.simd.vdup(scale[0], "float32")
        lo = T.simd.vdup(T.float32(-448), "float32")
        hi = T.simd.vdup(T.float32(448), "float32")
        sign_mask = T.simd.vdup(T.uint32(128), "uint32")
        nan_code = T.simd.vdup(T.uint32(127), "uint32")
        magnitude_mask = T.simd.vdup(T.uint32(0x7FFFFFFF), "uint32")
        for part in T.serial(elements // 64):
            original = T.simd.vcvt(
                T.simd.vld(raw[part * 64], dist="UNPK_B16"), "float32", part=0
            )
            wide = T.simd.vdiv(original, divisor, full, precision="exact")
            # Ordered comparisons leave NaNs alone. Repair their output bytes
            # explicitly instead of depending on the native cast's NaN mode.
            low = T.simd.vsel(lo, wide, T.simd.vcmp(wide, lo, full, "lt"))
            bounded = T.simd.vsel(hi, low, T.simd.vcmp(low, hi, full, "gt"))
            converted = T.simd.vcvt(
                bounded, "float8_e4m3fn", full, round="ROUND_R", sat=True, part=0
            )
            original_bits = T.reinterpret(original, "uint32x64")
            magnitude = T.simd.vand(original_bits, magnitude_mask, full)
            # Detect NaNs using integer bits: floating x != x can be folded
            # under finite-math assumptions or use ordered compare semantics.
            is_nan = T.simd.vcmps(magnitude, T.uint32(0x7F800000), full, "gt")
            sign = T.simd.vand(
                T.simd.vshrs(original_bits, 24, full),
                sign_mask,
                full,
            )
            bits = T.simd.vsel(
                T.simd.vor(sign, nan_code, full),
                T.reinterpret(converted, "uint32x64"),
                is_nan,
            )
            T.simd.vsts(
                packed[part * 64], T.reinterpret(bits, "uint8x256"), full, "PK4_B32"
            )


def batched_program(width, physical_rows, dtype, quantized, rows=16, cores=64):
    """Build a PrimFunc separately from compilation for inspection and tuning.

    Only full row tiles enter this kernel. The Python launcher routes partial
    tiles to the generic implementation, so no uninitialized UB lane is read.
    16x1024 native16 K/V rows occupy 64 KiB of UB in total.
    """
    if width <= 0 or width % 64 or rows < 1 or rows * width > 16384:
        raise ValueError("batched rows require a 64-aligned width and at most 16384 elements")
    if cores < 1 or cores > 64:
        raise ValueError("Ascend950 vector core count must be in [1, 64]")
    tokens = T.dynamic("tokens")
    elements = rows * width
    cache_dtype = "uint8" if quantized else dtype

    if quantized:

        @T.prim_func
        def main(
            K: T.Tensor((tokens, width), dtype),
            V: T.Tensor((tokens, width), dtype),
            KP: T.Tensor((physical_rows, width), cache_dtype),
            VP: T.Tensor((physical_rows, width), cache_dtype),
            Slots: T.Tensor((tokens,), "int64"),
            KS: T.Tensor((1,), "float32"),
            VS: T.Tensor((1,), "float32"),
        ):
            T.func_attr({"tilelang_callee_allocated_outputs": 0})
            with T.Kernel(cores) as bx:
                rawk = T.alloc_shared((rows, width), dtype)
                flatk = T.view(rawk, (elements,))
                rawv = T.alloc_shared((rows, width), dtype)
                flatv = T.view(rawv, (elements,))
                qk = T.alloc_shared((elements,), "uint8")
                qv = T.alloc_shared((elements,), "uint8")
                slots = T.alloc_shared((rows,), "int64")
                ks = T.alloc_shared((1,), "float32")
                vs = T.alloc_shared((1,), "float32")
                T.copy(KS, ks)
                T.copy(VS, vs)
                for step in T.serial(T.ceildiv(tokens // rows, cores)):
                    start = (step * cores + bx) * rows
                    if start < tokens:
                        T.copy(K[start : start + rows, :], rawk)
                        T.copy(V[start : start + rows, :], rawv)
                        T.copy(Slots[start : start + rows], slots)
                        _quantize_rows(flatk, qk, ks, elements)
                        _quantize_rows(flatv, qv, vs, elements)
                        for r in T.serial(rows):
                            slot = slots[r]
                            if slot >= 0:
                                T.copy(qk[r * width : (r + 1) * width], KP[slot, :])
                                T.copy(qv[r * width : (r + 1) * width], VP[slot, :])
    else:

        @T.prim_func
        def main(
            K: T.Tensor((tokens, width), dtype),
            V: T.Tensor((tokens, width), dtype),
            KP: T.Tensor((physical_rows, width), dtype),
            VP: T.Tensor((physical_rows, width), dtype),
            Slots: T.Tensor((tokens,), "int64"),
        ):
            T.func_attr({"tilelang_callee_allocated_outputs": 0})
            with T.Kernel(cores) as bx:
                rawk = T.alloc_shared((rows, width), dtype)
                rawv = T.alloc_shared((rows, width), dtype)
                slots = T.alloc_shared((rows,), "int64")
                for step in T.serial(T.ceildiv(tokens // rows, cores)):
                    start = (step * cores + bx) * rows
                    if start < tokens:
                        T.copy(K[start : start + rows, :], rawk)
                        T.copy(V[start : start + rows, :], rawv)
                        T.copy(Slots[start : start + rows], slots)
                        for r in T.serial(rows):
                            slot = slots[r]
                            if slot >= 0:
                                T.copy(rawk[r, :], KP[slot, :])
                                T.copy(rawv[r, :], VP[slot, :])
    return main


@functools.lru_cache(maxsize=64)
def build_batched(width, physical_rows, dtype, quantized, rows=16, cores=64):
    return tilelang.compile(
        batched_program(width, physical_rows, dtype, quantized, rows, cores),
        target="ascend",
        out_idx=[],
    )


def launch_config(tokens, width, quantized):
    """Conservative starting points; benchmark row batches on the target machine.

    N stays dynamic inside each program. Selecting a small finite family here
    does not cache a separate executable for every decode batch size.
    """
    if tokens <= 0 or width not in (128, 512, 1024):
        return None
    if tokens <= 64:
        rows = 1
    elif tokens < 1024:
        rows = 4
    else:
        rows = 8 if quantized else 16
    if tokens % rows:
        return None
    return rows, min(64, tokens // rows)


@functools.lru_cache(maxsize=64)
def build(width, physical_rows, dtype, quantized):
    tokens = T.dynamic("tokens")
    tile = min(1024, (width + 63) // 64 * 64)
    columns = (width + tile - 1) // tile
    cache_dtype = "uint8" if quantized else dtype
    if quantized:

        @T.prim_func
        def main(
            K: T.Tensor((tokens, width), dtype),
            V: T.Tensor((tokens, width), dtype),
            KP: T.Tensor((physical_rows, width), cache_dtype),
            VP: T.Tensor((physical_rows, width), cache_dtype),
            Slots: T.Tensor((tokens,), "int64"),
            KS: T.Tensor((1,), "float32"),
            VS: T.Tensor((1,), "float32"),
        ):
            T.func_attr({"tilelang_callee_allocated_outputs": 0})
            with T.Kernel(64) as bx:
                rawk = T.alloc_shared((tile,), dtype)
                rawv = T.alloc_shared((tile,), dtype)
                qk = T.alloc_shared((tile,), "uint8")
                qv = T.alloc_shared((tile,), "uint8")
                ks = T.alloc_shared((1,), "float32")
                vs = T.alloc_shared((1,), "float32")
                T.copy(KS, ks)
                T.copy(VS, vs)
                for step in T.serial(T.ceildiv(tokens * columns, 64)):
                    job = step * 64 + bx
                    if job < tokens * columns:
                        token = job // columns
                        col = job % columns * tile
                        count = T.min(tile, width - col)
                        slot = Slots[token]
                        if slot >= 0:
                            T.copy(K[token, col : col + count], rawk[:count])
                            T.copy(V[token, col : col + count], rawv[:count])
                            with T.SimdVF():
                                full = T.simd.pset(32)
                                sk = T.simd.vdup(ks[0], "float32")
                                sv = T.simd.vdup(vs[0], "float32")
                                lo = T.simd.vdup(T.float32(-448), "float32")
                                hi = T.simd.vdup(T.float32(448), "float32")
                                for part in T.serial(tile // 64):
                                    k = T.simd.vcvt(
                                        T.simd.vld(rawk[part * 64], dist="UNPK_B16"),
                                        "float32",
                                        part=0,
                                    )
                                    v = T.simd.vcvt(
                                        T.simd.vld(rawv[part * 64], dist="UNPK_B16"),
                                        "float32",
                                        part=0,
                                    )
                                    k = T.simd.vdiv(k, sk, full, precision="exact")
                                    v = T.simd.vdiv(v, sv, full, precision="exact")
                                    k = T.simd.vmin(T.simd.vmax(k, lo, full), hi, full)
                                    v = T.simd.vmin(T.simd.vmax(v, lo, full), hi, full)
                                    ek = T.simd.vcvt(
                                        k, "float8_e4m3fn", full, round="ROUND_R", sat=True, part=0
                                    )
                                    ev = T.simd.vcvt(
                                        v, "float8_e4m3fn", full, round="ROUND_R", sat=True, part=0
                                    )
                                    T.simd.vsts(
                                        qk[part * 64],
                                        T.reinterpret(ek, "uint8x256"),
                                        full,
                                        "PK4_B32",
                                    )
                                    T.simd.vsts(
                                        qv[part * 64],
                                        T.reinterpret(ev, "uint8x256"),
                                        full,
                                        "PK4_B32",
                                    )
                            with T.SimtVF(threads=256):
                                for i in T.Parallel(tile):
                                    if i < count:
                                        if T.isnan(T.float32(rawk[i])):
                                            sign = (
                                                T.reinterpret(T.float32(rawk[i]), "uint32") >> 24
                                            ) & T.uint32(128)
                                            qk[i] = T.cast(sign | T.uint32(127), "uint8")
                                        if T.isnan(T.float32(rawv[i])):
                                            sign = (
                                                T.reinterpret(T.float32(rawv[i]), "uint32") >> 24
                                            ) & T.uint32(128)
                                            qv[i] = T.cast(sign | T.uint32(127), "uint8")
                            T.copy(qk[:count], KP[slot, col : col + count])
                            T.copy(qv[:count], VP[slot, col : col + count])
    else:

        @T.prim_func
        def main(
            K: T.Tensor((tokens, width), dtype),
            V: T.Tensor((tokens, width), dtype),
            KP: T.Tensor((physical_rows, width), dtype),
            VP: T.Tensor((physical_rows, width), dtype),
            Slots: T.Tensor((tokens,), "int64"),
        ):
            T.func_attr({"tilelang_callee_allocated_outputs": 0})
            with T.Kernel(64) as bx:
                rawk = T.alloc_shared((tile,), dtype)
                rawv = T.alloc_shared((tile,), dtype)
                for step in T.serial(T.ceildiv(tokens * columns, 64)):
                    job = step * 64 + bx
                    if job < tokens * columns:
                        token = job // columns
                        col = job % columns * tile
                        count = T.min(tile, width - col)
                        slot = Slots[token]
                        if slot >= 0:
                            T.copy(K[token, col : col + count], rawk[:count])
                            T.copy(V[token, col : col + count], rawv[:count])
                            T.copy(rawk[:count], KP[slot, col : col + count])
                            T.copy(rawv[:count], VP[slot, col : col + count])

    return tilelang.compile(main, target="ascend", out_idx=[])


class AscendPagedKVCacheWriteKernel(Kernel, PagedKVCacheWriteFwdInterface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__()
        self.call = call
        self.dtype = str(call.dtype).removeprefix("torch.")

    def forward(self, k, v, k_pages, v_pages, slot_mapping, k_scale=None, v_scale=None):
        n = k.shape[0]
        if n == 0:
            return None
        width = self.call.row_width
        physical = self.call.physical_tokens
        kp = k_pages.view(torch.uint8) if self.call.quantized else k_pages
        vp = v_pages.view(torch.uint8) if self.call.quantized else v_pages
        args = [
            k.reshape(n, width),
            v.reshape(n, width),
            kp.reshape(physical, width),
            vp.reshape(physical, width),
            slot_mapping,
        ]
        if self.call.quantized:
            args.extend([k_scale, v_scale])
        config = launch_config(n, width, self.call.quantized)
        if config is None:
            compiled = build(width, physical, self.dtype, self.call.quantized)
        else:
            compiled = build_batched(width, physical, self.dtype, self.call.quantized, *config)
        compiled(*args)
        return None


from tileops.backend._registry import register_kernel_type  # noqa: E402

register_kernel_type("PagedKVCacheWriteFwdOp", "ascend", AscendPagedKVCacheWriteKernel)
