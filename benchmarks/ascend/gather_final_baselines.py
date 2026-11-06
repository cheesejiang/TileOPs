"""Final gather comparison: ops-nn full composition or vllm-ascend FP8 adapter.

The FP8 route follows the supplied historical adapter, including CPU indices,
full FP32 cache conversion/clone, Triton state copy and Triton scale application.
It is a Triton baseline, not an Ascend C implementation.
"""
import functools
import math
import os
from pathlib import Path

import torch

from benchmarks.api import Implementation


@functools.lru_cache(maxsize=1)
def _vllm_modules():
    from benchmarks.ascend.vendor_source import load_source

    configured = os.environ.get('VLLM_ASCEND_ROOT')
    if not configured:
        raise RuntimeError('Set VLLM_ASCEND_ROOT to the source checkout containing vllm_ascend/')
    checkout = Path(configured).resolve(strict=True)
    vendor = checkout / 'vllm_ascend'
    if not vendor.is_dir():
        raise ValueError('VLLM_ASCEND_ROOT must contain vllm_ascend/')
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError('An Ascend-capable Triton installation is required; full vLLM is not required') from exc
    try:
        target = triton.runtime.driver.active.get_current_target()
    except Exception as exc:
        raise RuntimeError('Cannot activate Ascend Triton backend; check CANN and the Triton wheel') from exc
    if target.backend not in ('ascend', 'npu', 'cann'):
        raise RuntimeError(f'Expected Ascend Triton backend, found {target.backend!r}')
    bindings = {'triton': triton, 'tl': tl, 'HAS_TRITON': True}
    # Load the original hardware utility, not a hardcoded replacement for core count.
    envs, env_evidence = load_source(vendor / 'envs.py', {})
    hardware, hardware_evidence = load_source(
        vendor / 'ops/triton/triton_utils.py', bindings | {'envs': envs})
    hardware.init_device_properties_triton()
    native, native_evidence = load_source(vendor / 'ops/triton/kda_state_copy_kernel.py', bindings)
    scaler, scaler_evidence = load_source(
        vendor / 'ops/triton/muls_add.py',
        bindings | {'get_vectorcore_num': hardware.get_vectorcore_num})
    evidence = {
        'loading': 'source-only-no-vllm-package-import',
        'triton_version': triton.__version__, 'triton_backend': target.backend,
        'modules': [env_evidence, hardware_evidence, native_evidence, scaler_evidence],
    }
    return native, scaler, evidence


class _VllmFP8Run:
    def __init__(self, inputs):
        dst, self.cache, self.table, self.cu, self.starts, self.scale = inputs
        self.dst = torch.empty_like(dst, memory_format=torch.contiguous_format)
        self.native, self.scaler, sources = _vllm_modules()
        self.provenance = {
            'library': 'vllm-ascend', 'implementation_language': 'Triton',
            'source_loading': sources, 'api': '_kda_state_copy_kernel + muls_add_triton',
            'scope': 'historical-fp8-adapter-with-dst-copy',
            'cpu_indices_each_call': True, 'full_cache_fp32_clone_each_call': True,
            'dst_copy_included': True,
        }

    def __call__(self):
        # Recompute from live metadata, as in the handed-over historical adapter.
        bounds = self.cu.tolist()
        beginnings = [0] * (len(bounds) - 1) if self.starts is None else self.starts.tolist()
        pages = self.table.tolist()
        ps = self.cache.shape[1]
        slots = [pages[b][position // ps] * ps + position % ps
                 for b, (begin, end) in enumerate(zip(bounds[:-1], bounds[1:]))
                 for position in range(beginnings[b], beginnings[b] + end - begin)]
        indices = torch.tensor(slots, dtype=torch.int64, device=self.cache.device)
        rows = self.cache.shape[0] * ps
        width = math.prod(self.cache.shape[2:])
        # Preserve original adapter work: full-cache widening AND clone.
        storage = self.cache.to(torch.float32).clone().contiguous().reshape(rows, width)
        packed = torch.empty((indices.numel(), width), dtype=torch.float32, device=self.cache.device)
        block = self.native.DEFAULT_KDA_BLOCK_SIZE
        self.native._kda_state_copy_kernel[(indices.numel(), (width + block - 1) // block)](
            storage, packed, indices, indices, rows, storage.stride(0), width,
            indices.stride(0), 0, TO_CACHE=False, HAS_FLAGS=False, BLOCK_SIZE=block,
        )
        flat = packed.reshape(-1, 1)
        scaled = self.scaler.muls_add_triton(flat, torch.zeros_like(flat), float(self.scale.item()))
        output = scaled.reshape(self.dst.shape).to(self.dst.dtype)
        # Match the public write-only dst contract; keep this device copy timed.
        self.dst.copy_(output)
        return self.dst

    def reset(self):
        self.dst.fill_(float('nan'))


def make_implementations(op_name, case):
    """Return exactly one baseline per case, in addition to runner's tileops."""
    if op_name != 'PagedKVCacheGatherFwdOp':
        return {}
    dtype = case.inputs[1].dtype
    if dtype == torch.float8_e4m3fn:
        run = _VllmFP8Run(case.inputs)
        tag = 'vllm-ascend-fp8-full'
    elif dtype in (torch.float16, torch.bfloat16):
        from benchmarks.ascend.gather_v2_baseline import _Run
        run = _Run(case.inputs, include_index_generation=True)
        tag = 'ops-nn-gatherv2-full'
    else:
        raise ValueError(f'Unsupported baseline cache dtype: {dtype}')
    return {tag: Implementation(run, args=(), reset=run.reset)}


def make_core_implementations(op_name, case):
    """Non-FP8: GatherV2 core only; FP8: the existing vllm full adapter.

    Core precomputes physical indices outside timing and is a partial-work
    comparison. The full baseline factory remains unchanged.
    """
    if op_name != 'PagedKVCacheGatherFwdOp':
        return {}
    if case.inputs[1].dtype in (torch.float16, torch.bfloat16):
        from benchmarks.ascend.gather_v2_baseline import _Run
        run = _Run(case.inputs, include_index_generation=False)
        return {'ops-nn-gatherv2-core': Implementation(run, args=(), reset=run.reset)}
    return make_implementations(op_name, case)
