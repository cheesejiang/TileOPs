# Paged cache gather on Ascend

[中文版](paged_cache_gather_zh.md)

The implementation follows PR #28's backend registration structure:
https://github.com/yyttt6/TileOPs/pull/28
`kernels/ascend/attention/paged_cache_gather.py` implements the existing gather
interface and registers an NPU candidate. The public Op, manifest, CUDA kernel,
write-only destination handling and numerical contract are shared.

All seven current manifest workloads are covered by the candidate's domain.
The domain additionally permits FP16/BF16/FP8 rows of width 576 or 1024, page
sizes 16 or 64, and 1..256 requests. Device metadata must obey the manifest's
index bounds, prefix-offset and paged-range contracts; no CPU metadata read
or additional synchronization is introduced in the operator.

The persistent grid uses the current device's vector-core count. Tasks cover
output tiles, balancing ragged requests without padding each request to the
maximum length. Each core caches request offsets and starts in UB. A binary
search resolves the first row of each task. A tile entirely inside one request
and physical page uses a single rectangular load and store. Tiles crossing
request/page boundaries load rows independently. Output tails only store valid
rows. Tiles contain a whole page without starts, or 16 rows with starts.

FP8 uses a separate output UB buffer, FP32 conversion and multiplication, then
one conversion to the destination dtype. Explicit DMA/vector dependencies
protect buffer reuse. This initial schedule is serialized between tiles;
double buffering and SIMD conversion are future tuning candidates, not measured
improvements. The largest data buffers total 108 KiB for FP8-to-BF16 (64x576),
plus metadata and compiler scratch; actual generated resource usage needs checking.

## Validation

On the supported Ascend 950 toolchain, run the focused tests under the site's
NPU reservation mechanism:

```bash
python -m pytest tests/ops/test_paged_cache_gather_ascend.py --tileops-device=npu
```

The device tests cover every manifest workload with contiguous and strided
destinations. They rerun the same Op after modifying page IDs, starts and scales.
For correctness through the shared guarded launcher:

```bash
python scripts/benchmark_ascend.py --owner gather/verify \
  --run-dir /shared/results/gather-verify-unique \
  --op PagedKVCacheGatherFwdOp --verify-only
```

Formal performance runs must use the clock configuration and protocol described
in [development.md](../../../../../docs/ascend950/development.md). Compare page-sized vs smaller tiles, core counts and a
double-buffered schedule using the same inputs and protocol. Report all seven
cases, particularly `ds-v3-mixed` and the unaligned-start cases.

Implementation-host checks only establish Python syntax and patch hygiene.
This host has no `torch_npu`, CANN or importable `tilelang.ascend`, so Ascend
compilation, FP8 lowering, simulator checks, device correctness and latency
remain unverified. No performance target or measured speedup is claimed.
