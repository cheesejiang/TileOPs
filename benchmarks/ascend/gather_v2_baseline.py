"""Non-FP8 GatherV2 baselines; call the selected ACLNN library directly.

No torch.index_select substitute is used for the measured cache gather.
"""

import ctypes as C
import functools
import hashlib
import os
from pathlib import Path

import torch

from benchmarks.api import Implementation


@functools.lru_cache(maxsize=1)
def _library():
    configured = os.environ.get("OPS_NN_OPAPI_LIBRARY")
    if not configured:
        raise RuntimeError("Set OPS_NN_OPAPI_LIBRARY to the absolute GatherV2 op-api .so path")
    path = Path(configured).resolve(strict=True)
    lib = C.CDLL(str(path), mode=C.RTLD_GLOBAL)
    tensor_lib = C.CDLL(os.environ["OPS_NN_TENSOR_LIBRARY"]) if os.environ.get("OPS_NN_TENSOR_LIBRARY") else lib
    i64p = C.POINTER(C.c_int64)
    tensor_lib.aclCreateTensor.argtypes = [i64p, C.c_uint64, C.c_int, i64p, C.c_int64,
                                          C.c_int, i64p, C.c_uint64, C.c_void_p]
    tensor_lib.aclCreateTensor.restype = C.c_void_p
    tensor_lib.aclDestroyTensor.argtypes = [C.c_void_p]
    tensor_lib.aclDestroyTensor.restype = C.c_int
    lib.aclnnGatherV2GetWorkspaceSize.argtypes = [C.c_void_p, C.c_int64, C.c_void_p,
                                                C.c_void_p, C.POINTER(C.c_uint64),
                                                C.POINTER(C.c_void_p)]
    lib.aclnnGatherV2GetWorkspaceSize.restype = C.c_int
    lib.aclnnGatherV2.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]
    lib.aclnnGatherV2.restype = C.c_int
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    provenance = {"api": "aclnnGatherV2", "library": str(path), "library_sha256": digest}
    return lib, tensor_lib, provenance


def physical_slots(table, cu, starts, rows, page_size):
    """Resolve live device metadata; no tolist/item or host metadata staging."""
    # 根据累积长度，找到这行属于哪个请求
    request = torch.searchsorted(cu[1:].contiguous(), rows, right=True)
    # 求该请求内部的逻辑读取位置
    pos = rows - cu[request].to(torch.int64)
    if starts is not None:
        pos = pos + starts[request].to(torch.int64)
    # 查页表，得到物理行号
    page_index = request * table.shape[1] + torch.div(pos, page_size, rounding_mode="floor")
    page = table.reshape(-1)[page_index].to(torch.int64)
    return (page * page_size + pos.remainder(page_size)).contiguous()


class GatherV2:
    """One logical stream, live ACL descriptors, and reusable device workspace."""

    def __init__(self):
        self.lib, self.tensor_lib, self.provenance = _library()
        self.workspace = None

    def _descriptor(self, tensor):
        if not tensor.is_contiguous() or tensor.device.type != "npu":
            raise ValueError("ACL GatherV2 baseline requires contiguous NPU tensors")
        dtype = {torch.float16: 1, torch.bfloat16: 27, torch.int32: 3, torch.int64: 9}[tensor.dtype]
        shape = (C.c_int64 * tensor.ndim)(*tensor.shape)
        strides = (C.c_int64 * tensor.ndim)(*tensor.stride())
        # data_ptr already includes the storage offset; the descriptor offset is zero.
        descriptor = self.tensor_lib.aclCreateTensor(shape, tensor.ndim, dtype, strides,
                                                     0, 2, shape, tensor.ndim, tensor.data_ptr())
        if not descriptor:
            raise RuntimeError("aclCreateTensor returned null")
        return descriptor

    def __call__(self, cache, slots, out):
        descriptors = []
        try:
            for tensor in (cache, slots, out):
                descriptors.append(self._descriptor(tensor))
            size, executor = C.c_uint64(), C.c_void_p()
            rc = self.lib.aclnnGatherV2GetWorkspaceSize(
                descriptors[0], 0, descriptors[1], descriptors[2], C.byref(size), C.byref(executor))
            if rc:
                raise RuntimeError(f"aclnnGatherV2GetWorkspaceSize failed: {rc}")
            if size.value and (self.workspace is None or self.workspace.numel() < size.value):
                self.workspace = torch.empty(size.value, dtype=torch.uint8, device=out.device)
            pointer = self.workspace.data_ptr() if size.value else 0
            stream = torch.npu.current_stream(out.device).npu_stream
            rc = self.lib.aclnnGatherV2(pointer, size.value, executor, stream)
            if rc:
                raise RuntimeError(f"aclnnGatherV2 failed: {rc}")
        finally:
            for descriptor in descriptors:
                self.tensor_lib.aclDestroyTensor(descriptor)
        return out


class _Run:
    def __init__(self, inputs, *, include_index_generation):
        dst, cache, self.table, self.cu, self.starts, _ = inputs
        self.dst = torch.empty_like(dst)
        self.cache = cache.flatten(0, 1)
        self.page_size = cache.shape[1]
        self.rows = torch.arange(dst.shape[0], dtype=torch.int64, device=dst.device)
        self.gather = GatherV2()
        self.full = include_index_generation
        self.slots = self._slots() if not self.full else None
        self.provenance = dict(self.gather.provenance,
                               scope="index-generation-and-gather" if self.full else "gather-only-precomputed-indices")

    def _slots(self):
        return physical_slots(self.table, self.cu, self.starts, self.rows, self.page_size)

    def __call__(self):
        slots = self._slots() if self.full else self.slots
        return self.gather(self.cache, slots, self.dst)

    def reset(self):
        # Output poisoning is outside profiling, avoiding stale-output false positives.
        self.dst.fill_(float("nan"))


def make_implementations(op_name, case):
    """Return full-equivalent and explicitly partial-timing baselines for 5 cases."""
    if op_name != "PagedKVCacheGatherFwdOp":
        return {}
    if case.params["cache"][1] not in ("float16", "bfloat16"):
        return {}
    implementations = {}
    for tag, full in (("ops-nn-gatherv2-full", True), ("ops-nn-gatherv2-core", False)):
        run = _Run(case.inputs, include_index_generation=full)
        implementations[tag] = Implementation(run, args=(), reset=run.reset)
    return implementations
