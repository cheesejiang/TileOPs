# Ascend 分页 Cache Gather 实现与验证说明

[English](paged_cache_gather.md)

## 接入方式与支持范围

实现参考 [PR #28](https://github.com/yyttt6/TileOPs/pull/28) 的后端注册结构。
同目录的 `paged_cache_gather.py` 实现已有 gather 内核接口，并注册为 NPU 候选实现。
公共 Op、manifest、CUDA 内核、只写输出的处理方式和数值契约保持共用。

候选实现的支持范围涵盖当前 manifest 中的全部 7 个 workload：

| workload | Cache dtype → 输出 dtype | 展平行宽 | 页大小 | 起点 |
| --- | --- | --- | --- | --- |
| `ds-v3-chunk-fp8` | FP8 E4M3FN → BF16 | 576 | 64 | 默认 0 |
| `ds-v3-mixed` | BF16 → BF16 | 576 | 64 | 默认 0 |
| `llama-70b-cp` | BF16 → BF16 | 1024 | 16 | `seq_starts` |
| `mla-small` | FP16 → FP16 | 576 | 64 | 默认 0 |
| `mla-chunk-fp8-fp16` | FP8 E4M3FN → FP16 | 576 | 64 | 非对齐起点 |
| `llama-cp-unaligned` | BF16 → BF16 | 1024 | 16 | 非对齐起点 |
| `mla-b256` | BF16 → BF16 | 576 | 64 | 默认 0 |

候选实现还允许满足以下条件的其他输入：行宽为 576 或 1024，页大小为 16 或 64，
请求数为 1～256；Cache 为 FP16、BF16 或 FP8 E4M3FN。
非量化路径保持 dtype；FP8 路径输出 FP16 或 BF16。
这些额外组合并不意味着已经通过实机验证。

设备端元数据必须满足 manifest 的索引范围、前缀偏移和分页区间契约。
算子不将元数据读回 CPU，也不为读取元数据引入主机同步；内核内部仍有必要的流水同步。

## 执行策略

持久化执行网格按当前设备的 Vector 核数建立，并按输出块分配任务。
这种方式避免按请求分核时，长短请求混合导致部分核工作量过大，也无需将各请求补齐到最大长度。

每个核将请求输出偏移和可选起点缓存到 UB。每个任务先二分查找首行所属请求，
再根据页表计算物理位置。整个块位于同一请求、同一物理页内时，使用矩形批量搬运；
跨请求或跨页时逐行读取。输出尾部只写有效行。
未传起点时一个块包含整页行数，传入起点时一个块包含 16 行。

FP8 路径使用独立输出 UB，先转 FP32 并乘以 `scale[0]`，再转换一次到目标 dtype。
显式 DMA/Vector 依赖保护缓冲区复用。当前初版在块之间串行执行，尚未实现双缓冲。
双缓冲和 SIMD 转换是后续调优方向，不能视为已测得的性能收益。

FP8 → BF16 的最大数据缓冲区合计为 108 KiB（64×576 元素的输入和输出），
此外还有元数据及编译器临时空间；实际生成代码的资源占用仍需检查。

## 正确性验证

以下命令在 `TileOPs` 仓库根目录执行。使用受支持的 Ascend 950 工具链，
并通过站点的 NPU 预约机制运行，避免与其他任务争用设备。

```bash
python -m pytest tests/ops/test_paged_cache_gather_ascend.py --tileops-device=npu
```

设备测试覆盖全部 7 个 workload 的连续输出与非连续输出 view。
测试会修改页表、起点及 scale，并复用同一个 Op 再次执行，以检查元数据仍从设备动态读取。
结果与独立参考实现精确比较，同时检查非连续输出未覆盖的存储位置保持不变。

通过公共受保护启动器执行正确性验证：

```bash
python scripts/benchmark_ascend.py --owner gather/verify \
  --run-dir /shared/results/gather-verify-unique \
  --op PagedKVCacheGatherFwdOp --verify-only
```

`--run-dir` 必须使用新的目录；按本机部署调整路径。
`--verify-only` 只验证正确性，不产生性能验收结果。

## 性能验证与调优

正式性能测量必须遵循
[Ascend 开发说明](../../../../../docs/ascend950/development.md)
规定的时钟配置、设备预约和测量协议。

### 运行性能测试

先通过上述正确性验证，再在 `TileOPs` 仓库根目录执行：

```bash
python scripts/benchmark_ascend.py \
  --owner gather/baseline \
  --run-dir /shared/results/gather-baseline-001 \
  --op PagedKVCacheGatherFwdOp \
  --clock-config /site/ascend950-clocks.json
```

该命令默认测试全部 7 个 workload。参数说明：

- `--owner`：本次运行的负责人或实验名称，可替换为自己的标识。
- `--run-dir`：新的结果目录，替换为本机可写路径，每次运行使用不同目录。
- `--clock-config`：站点提供的有效锁频配置，包含锁频、恢复命令和两个组件的目标频率。
  `/site/ascend950-clocks.json` 是路径示例，需要替换为实际配置；没有有效配置时不能进行正式性能测量。
- 不传 `--verify-only`：执行正确性检查及性能测量。

如需优先测试最初关注的三个规格，在上述命令末尾追加：

```bash
--case-regex 'ds-v3-chunk-fp8|ds-v3-mixed|llama-70b-cp'
```

启动器会在导入 NPU 运行时之前申请设备锁、检查设备占用，并监测运行期间的争用。
当前正式协议为 `ascend950-profiler-locked-v1`：正确性检查和 JIT 设置完成后，
预热 3 次，再采集 15 次独立 profiler 样本，每个样本执行一次算子。
每次采样前执行缓存重置、1 GiB 缓存驱逐及同步。
计时采用原始 CANN profiler 的设备执行区间，重叠区间只计算一次，不包含主机间隙。

### 判断运行是否有效并读取结果

先检查 `receipt.json`：只有 `status` 为 `passed`，且 `result.verify_only` 为 `false`，
才能将此次运行作为性能结果。设备争用、锁频失败、正确性失败或运行期间源码变化都会使结果失效。

| 文件或字段 | 含义 |
| --- | --- |
| `receipt.json` | 整次运行的最终状态和设备保护退出码 |
| `results.jsonl` | 每个 workload 一行，包含正确性和测量结果 |
| `implementations.tileops.measurement.latency_ms` | 15 次设备执行区间耗时的算术平均值，单位 ms，主要比较指标 |
| `implementations.tileops.measurement.samples_ms` | 全部 15 次原始耗时 |
| `implementations.tileops.measurement.stdev_ms` | 样本标准差，用于观察波动 |
| `environment.json` | 软件、设备及测量环境 |
| `clock-readbacks.json` | 时钟读回记录 |
| `profiles/` | 原始 profiler 数据，用于定位瓶颈 |
| `run.log` | 编译、执行与报错日志 |

`median_ms`、`min_ms`、`max_ms` 是辅助统计，正式比较使用 `latency_ms`，
不要只挑选最快样本。保留结果目录及原始 profiler 数据，便于复查。

### 比较优化版本

先保存当前 Ascend 初版的测量结果，再逐项修改和复测。
每次修改后先检查正确性，用新的结果目录运行相同命令，并记录源码版本和改动内容。
测量期间不要修改源码。

在相同输入与协议下比较以下选项：

- 整页块与更小块的搬运效率。
- 不同核数下的负载均衡与带宽利用率。
- 双缓冲是否能够隐藏搬运或转换开销。
- FP8 转换采用当前 SIMT 路径与 SIMD 路径的差异。

报告全部 7 个 workload 的结果，尤其关注 `ds-v3-mixed` 的长短请求混合，
以及非对齐起点触发逐行搬运的情况。

建议按下面的格式记录每个规格：

| workload | 初版平均耗时（ms） | 优化版平均耗时（ms） | 加速比 | 样本标准差（ms） |
| --- | --- | --- | --- | --- |
| `ds-v3-chunk-fp8` | 待测 | 待测 | 待测 | 待测 |
| `ds-v3-mixed` | 待测 | 待测 | 待测 | 待测 |
| `llama-70b-cp` | 待测 | 待测 | 待测 | 待测 |

加速比为“初版平均耗时 / 优化版平均耗时”。初版与优化版应在同一台 Ascend 上，
使用相同工具链、输入、锁频策略和测量协议。
如果需要与其他 Ascend 实现比较，可以通过启动器的 `--implementation-factory module:function`
接入符合仓库 `Implementation` 接口的基线适配器；CUDA 耗时仅作为跨硬件参考。

## 当前验证状态

实现时所在主机仅完成 Python 语法检查和补丁格式检查。
该环境没有 `torch_npu`、CANN，也无法导入 `tilelang.ascend`。
因此 Ascend 编译、FP8 转换代码生成、模拟器检查、设备正确性及延迟均尚未验证。
当前不宣称达到性能目标，也没有实测加速比。
