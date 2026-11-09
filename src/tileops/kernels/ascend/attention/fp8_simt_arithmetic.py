"""Experimental FP8-input/FP8-output SIMT addition and multiplication.

The arithmetic expressions use FP8-typed fragments without an explicit FP32
cast. The compiler may still promote them internally; this experiment does not
establish that the hardware has native FP8 add/multiply instructions.

Inputs must have identical shapes and FP8 formats and be contiguous on one NPU.
There is no broadcasting, scaling, saturation override, or TileOPs registration.
"""

import functools

import torch


@functools.lru_cache(maxsize=32)
def _compile(operation, dtype, block_size, threads):
    # Import the compiler only when an actual device kernel is needed.
    import tilelang
    import tilelang.ascend.language as T

    n = T.dynamic("n")

    @T.prim_func
    def main(
        a: T.Tensor((n,), dtype),
        b: T.Tensor((n,), dtype),
        out: T.Tensor((n,), dtype),
    ):
        with T.Kernel(T.ceildiv(n, block_size)) as bx:
            begin = bx * block_size
            valid = T.min(block_size, n - begin)
            a_ub = T.alloc_shared((block_size,), dtype)
            b_ub = T.alloc_shared((block_size,), dtype)
            out_ub = T.alloc_shared((block_size,), dtype)

            # MTE2: read only valid elements, including non-aligned tails.
            T.copy(a[begin : begin + valid], a_ub[0:valid])
            T.copy(b[begin : begin + valid], b_ub[0:valid])
            T.ascend_set_flag("MTE2_V", 0)
            T.ascend_wait_flag("MTE2_V", 0)

            # Use FP8-typed operands in the SIMT expressions themselves.
            with T.SimtVF(threads=threads):
                a_local = T.alloc_fragment((block_size,), dtype)
                b_local = T.alloc_fragment((block_size,), dtype)
                out_local = T.alloc_fragment((block_size,), dtype)
                for i in T.Parallel(block_size):
                    if i < valid:
                        a_local[i] = a_ub[i]
                        b_local[i] = b_ub[i]
                        if operation == "add":
                            out_local[i] = a_local[i] + b_local[i]
                        else:
                            out_local[i] = a_local[i] * b_local[i]
                        out_ub[i] = out_local[i]

            # MTE3: write FP8 results after the Vector Function finishes.
            T.ascend_set_flag("V_MTE3", 0)
            T.ascend_wait_flag("V_MTE3", 0)
            T.copy(out_ub[0:valid], out[begin : begin + valid], l2_cache_ctrl=0)
            T.ascend_set_flag("MTE3_S", 0)
            T.ascend_wait_flag("MTE3_S", 0)

    return tilelang.compile(
        main,
        target="ascend",
        out_idx=[],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
            tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
    )


def _binary(a, b, operation, *, block_size=1024, threads=128):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("both operands must be torch tensors")
    if a.device.type != "npu" or b.device != a.device:
        raise ValueError("both operands must be on the same NPU")
    if a.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2) or b.dtype != a.dtype:
        raise ValueError("both operands must have the same FP8 E4M3FN or E5M2 dtype")
    if a.shape != b.shape:
        raise ValueError("operand shapes must match; broadcasting is not supported")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("operands must be contiguous")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads not in (128, 256):
        raise ValueError("threads must be 128 or 256")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < threads
        or block_size > 8192
        or block_size % threads
    ):
        raise ValueError("block_size must be a multiple of threads, between threads and 8192")
    out = torch.empty_like(a)
    if a.numel():
        dtype = str(a.dtype).removeprefix("torch.")
        kernel = _compile(operation, dtype, block_size, threads)
        with torch.npu.device(a.device):
            kernel(a.view(-1), b.view(-1), out.view(-1))
    return out


def fp8_add(a, b, *, block_size=1024, threads=128):
    """Return elementwise a + b with the same FP8 dtype and shape as a."""
    return _binary(a, b, "add", block_size=block_size, threads=threads)


def fp8_mul(a, b, *, block_size=1024, threads=128):
    """Return elementwise a * b with the same FP8 dtype and shape as a."""
    return _binary(a, b, "mul", block_size=block_size, threads=threads)
