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

### 多卡服务器上的开发测量（无需锁频或固定核数）

在 `TileOPs` 根目录执行，下面以物理卡 3 为例：

```bash
python scripts/benchmark_ascend.py \
  --device 3 --unlocked \
  --owner gather/dev \
  --run-dir "/shared/results/gather-dev-$(date +%Y%m%d-%H%M%S)" \
  --op PagedKVCacheGatherFwdOp
```

`--device` 是物理卡编号。保护程序只预约和检查该卡，使用独立的 `device-3.lock`，
子进程中该卡映射为逻辑 `npu:0`；其他卡可以继续运行任务。
共享 manager/HDC 句柄不能归属到某张卡，因此多卡模式不据此阻止运行，
而是检查目标卡的设备句柄及 DSMI 驱动进程；尚未建立设备上下文的初始化进程可能稍后才被发现。
运行过程中仍检查目标卡是否出现外部进程。

`--unlocked` 不执行锁频或频率读回，不要求固定 32 个 Cube 核、设备型号名称
或精确的 TileLang 版本。它不替代底层内核与硬件的兼容性要求。
实际设备型号、Cube/Vector 核数和物理卡编号会记录到 `environment.json`。
仍执行精度检查、3 次预热、15 次 profiler 采样和缓存驱逐。
不要同时传 `--clock-config`。

仅验证正确性时，在同一命令中追加 `--verify-only`。
查看目标卡的状态：

```bash
python scripts/npu_guard.py status --device 3 --json
```

非 root 用户需要管理员先创建该卡的共享锁：

```bash
python scripts/npu_guard.py init --device 3
```

只有 `receipt.json` 中 `status` 为 `passed`，且 `result.verify_only` 为 `false` 时，
才是成功的性能运行。每个 case 的耗时位于 `results.jsonl` 的
`implementations.tileops.measurement.latency_ms`，单位 ms，是 15 次采样的算术平均值。
同时保留 `samples_ms`、`stdev_ms` 和 `profiles/`，比较优化前后的波动与耗时。
开发测量标记为 `ascend-profiler-unlocked-v1`、`formal: false`，
应在同卡同环境下比较，不与锁频正式测量混用。

### 原有正式测量

正式性能测量必须遵循
[Ascend 开发说明](../../../../../docs/ascend950/development.md)
规定的时钟配置、设备预约和测量协议。

在相同输入与协议下比较以下选项：

- 整页块与更小块的搬运效率。
- 不同核数下的负载均衡与带宽利用率。
- 双缓冲是否能够隐藏搬运或转换开销。
- FP8 转换采用当前 SIMT 路径与 SIMD 路径的差异。

报告全部 7 个 workload 的结果，尤其关注 `ds-v3-mixed` 的长短请求混合，
以及非对齐起点触发逐行搬运的情况。

## 当前验证状态

实现时所在主机仅完成 Python 语法检查和补丁格式检查。
该环境没有 `torch_npu`、CANN，也无法导入 `tilelang.ascend`。
因此 Ascend 编译、FP8 转换代码生成、模拟器检查、设备正确性及延迟均尚未验证。
当前不宣称达到性能目标，也没有实测加速比。
