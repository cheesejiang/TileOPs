# 分组地址准备与搬运候选

`paged_cache_gather_batch.py` 新增 batch2/batch4，不注册为默认实现，不改变公共 Op。
仅支持同 dtype 的 FP16/BF16，沿用公共 kernel 的主机检查和原地 dst 输出。

每核领取连续的 16 行 tile 区间，首次二分定位请求，之后递增跳过请求边界。
每组准备 2/4 个 tile 的全部片段地址，再连续发出 cache → UB 搬运，
整组 MTE2_MTE3 同步后写出；MTE3_MTE2 等待后才复用 UB。没有组间双缓冲。
完整组输出为一次矩形复制。跨页、跨请求、空请求及尾部用动态片段处理。
物理地址由每次调用的设备元数据计算，没有 CPU 预处理或固定测试值。

BF16 行宽 1024 时 payload UB 为 64/128 KiB，另有元数据和片段描述符。
首版使用标量写读 UB 描述符；是否产生额外标量开销、编译是否保持连续发射，
需检查生成代码和 simulator，不保证比 auto2 快。已安装 CANN 的 GatherV2
与本地源码不一定一致，也不能从图中轨道数推断固定并发度。

从 TileOPs 根目录运行正确性检查：

```bash
python -m pytest tests/ops/test_paged_cache_gather_batch_ascend.py --tileops-device=npu -q
```

同一轮性能对照（按实际空闲卡修改 device；汇总只显示 auto2、batch2 和 batch2 双缓冲）：

```bash
python scripts/benchmark_ascend.py \
  --device 3 --unlocked --owner gather/batch \
  --run-dir "./benchmark-results/batch-$(date +%Y%m%d-%H%M%S)" \
  --op PagedKVCacheGatherFwdOp \
  --implementation-factory benchmarks.ascend.gather_auto:make_implementations \
  --summary-implementations tilelang-auto2 tilelang-batch2 tilelang-batch2-db
```

`summary.md` 保存在上述 `--run-dir` 目录中，终端也打印相同表格。
要显示 batch4，可在 `--summary-implementations` 后添加 `tilelang-batch4`；
要显示默认手工版，可添加 `tileops`；删除整个筛选参数则显示全部方法。

现有 manual/auto 候选保留，五个非 FP8 规格额外产生 batch2、batch4 和 batch2-db。
FP8 规格没有 batch 候选，在上述筛选条件下仅显示 auto2。
需要先验证时追加 `--verify-only`。测试应覆盖修改页表/起点后的重复调用、
跨页、空请求以及不足一组的输出。真实耗时以同卡同轮测试为准。

流水采集脚本新增 `--implementation batch2` 和 `batch4`，其余参数沿用
scripts/profile_gather_pipeline_README.md。传入 `--source-dir` 可导出生成代码。
使用 msopprof simulator 时不要同时开启脚本的 `--profile-dir`。

## batch2 双缓冲候选

新增 `tilelang-batch2-db`，通过同一个 implementation factory 自动加入非 FP8 对照。
原来的 batch2/batch4 单缓冲仍保留。构造参数为
`AscendPagedKVCacheGatherBatchKernel(call, group_tiles=2, double_buffer=True)`。

两组独立 payload UB 交替使用，BF16 行宽 1024 时共 128 KiB，不含元数据。
每块缓冲使用独立的 ready（MTE2_MTE3）和 release（MTE3_MTE2）事件：

1. 准备本组地址，复用缓冲前等待其上次输出完成。
2. 发起本组全部读请求，设置本组 ready。
3. 等待上一组 ready，写出上一组，设置上一组 release。
4. 循环结束后写出最后一组，并消费所有剩余 release 事件。

第一组只读入；从第二组开始读当前组、写上一组；只有最后一组可能不足完整组。
没有任务的核不参与组事件。事件 ID 按缓冲固定为 0/1，不使用动态事件编号。
地址计算仍有标量/页表依赖，因此实际重叠和收益须通过生成代码、流水及实机确认。
batch4 交替区域候选现已提供，见下节；其 256 KiB payload 加元数据需要设备编译验证容量。

原正确性命令已同时覆盖单缓冲和双缓冲。流水采集可使用：

```bash
python scripts/profile_gather_pipeline.py \
  --implementation batch2-db --source-dir ./pipeline-results/batch2-db-source --check
```

在对应预约或 simulator 环境下运行，设备和 SoC 配置沿用流水采集 README。
本地仅完成语法、差异格式及事件配对模型检查，未完成 NPU 编译/执行。

只在终端汇总和 `summary.md` 显示指定对照，可在 benchmark 命令中追加：

```bash
--summary-implementations tilelang-auto2 tilelang-batch2 tilelang-batch2-db
```

不指定时显示全部。该参数只过滤展示，不改变实际执行、正确性验证或原始
`results.jsonl`，receipt 仍表示整次运行的状态。


## batch4 使用两个交替区域

新增 `tilelang-batch4-db`，每组仍为 64 行、每次读入仍为 16 行。
只改变 UB 区域数量和组间同步，不同时改变读入粒度。
同一 UB 分配具有 `[2, 64, row_width]` 布局：组 0/2/4 使用 A，
组 1/3/5 使用 B。发出组 n 读入后，写出组 n-1；复用 A/B 前分别等待
该区域的上次写出完成。组内四次读入之间没有额外的完成等待。
这沿用 batch2-db 的按区域事件协议，不是另一种新的调度算法。

BF16、row_width=1024 时仅 payload 就占 256 KiB，还需要元数据和描述符。
如果超出实际 UB 容量，设备编译会失败；不能删除复用等待来绕过容量限制。
本地没有完成设备编译，不能保证该配置在当前机器可用。

只加载 batch4 两个候选（公共 tileops 仍由 runner 加入），并仅展示两者：

```bash
python scripts/benchmark_ascend.py \
  --device 3 --unlocked --owner gather/batch4 \
  --run-dir "./benchmark-results/batch4-$(date +%Y%m%d-%H%M%S)" \
  --op PagedKVCacheGatherFwdOp \
  --implementation-factory benchmarks.ascend.gather_auto:make_batch4_implementations \
  --summary-implementations tilelang-batch4 tilelang-batch4-db
```

先追加 `--verify-only` 验证，再去掉该参数测性能。FP8 不生成这两个候选。
流水脚本选择 `--implementation batch4-db`；`--source-dir` 可导出代码，
对比事件 0/1 是否分别约束两个 UB 区域。

## 单缓冲：16 次 × 4 行读入

新增 `tilelang-batch16-r4`：`group_tiles=16, tile_rows=4, double_buffer=False`。
每组仍为 64 行，BF16、行宽 1024 时 payload UB 仍为 128 KiB。
与 batch4 的区别只有读入上限从 16 行变为 4 行：对齐完整组由
4 次 × 32 KiB 变成 16 次 × 8 KiB，最后仍统一写出 128 KiB。
跨页/请求时可能进一步拆分，因此不是所有输入都恰好 16 次读入。
每核区间仍按 16 行划分，与原 batch4 保持一致，避免改变任务分配干扰比较。
组间仍串行：写出完成后复用同一段 UB，不采用两个交替区域。

下面命令只运行这两个单缓冲候选的正确性和性能测试，并只汇总两者。
不实例化或执行公共 tileops，不运行其他实验候选。独立的数值参考校验仍保留。

```bash
python scripts/benchmark_ascend.py \
  --device 3 --unlocked --owner gather/load-granularity \
  --run-dir "./benchmark-results/load-granularity-$(date +%Y%m%d-%H%M%S)" \
  --op PagedKVCacheGatherFwdOp \
  --case-regex "^(ds-v3-mixed|llama-70b-cp|mla-small|llama-cp-unaligned|mla-b256)-" \
  --implementation-factory benchmarks.ascend.gather_auto:make_load_granularity_implementations \
  --implementations tilelang-batch4 tilelang-batch16-r4
```

先追加 `--verify-only` 验证；正确性 pytest 也已加入该配置。
流水采集选择 `--implementation batch16-r4`。
本地已做 Python 语法和 diff 检查；NPU 编译、精度和性能尚待设备验证。

`--implementations` 同时筛选执行和汇总；`--summary-implementations` 仅筛选显示。
上述 case 筛选排除不支持的 FP8 规格。指定的方法在某个规格不可用时明确报错。

## SIMT 直接读写 GM 候选

`paged_cache_gather_simt.py` 提供 `tilelang-simt`，不替换默认实现。
使用 `T.SimtVF(threads=256)`，线程按展平元素并行，执行请求二分、页表寻址，
直接读取 GM cache 并写入 GM dst；不分配 payload UB，不使用组级 DMA 事件。
FP8 路径在 SIMT 内转 FP32、乘 scale、写目标 dtype，覆盖原七种规格。
这是一版基线：同一行各元素可能重复计算和读取元数据，不能预先保证更快。
若当前后端不支持该 GM 访问模式或 FP8 转换，设备编译会暴露，尚未实机确认。
流水应关注 SIMT 访存和指令开销，不能再仅用 MTE2/MTE3 的密集程度评价。

先验证：

```bash
python -m pytest tests/ops/test_paged_cache_gather_simt_ascend.py --tileops-device=npu -q
```

只执行 auto2 与 SIMT 两者，不运行其他实验或公共 tileops：

```bash
python scripts/benchmark_ascend.py \
  --device 3 --unlocked --owner gather/simt \
  --run-dir "./benchmark-results/simt-$(date +%Y%m%d-%H%M%S)" \
  --op PagedKVCacheGatherFwdOp \
  --implementation-factory benchmarks.ascend.gather_auto:make_simt_implementations \
  --implementations tilelang-auto2 tilelang-simt
```

可先追加 `--verify-only`；流水脚本新增 `--implementation simt`。
本地仅通过语法和 diff 检查，NPU 编译、数值及性能需要设备验证。

## 固定总 UB 的交叠候选（当前重点）

`tilelang-overlap32-r4`：每区域 32 行、每次读 4 行，两区域共 64 行。
BF16、行宽 1024 时 payload 总计 128 KiB，和 batch16-r4 相同，写出粒度
从 64 行改成 32 行。不是单区域原地覆盖；A/B 交替复用。

本次还修正了交替区域的发射顺序：每轮先等待上一组 ready 并提交其 MTE3
写出，再准备当前组地址、提交当前组 MTE2 读入；只有覆盖同一区域之前才
等待其 release。避免当前组的标量页表访问延迟上一组写出提交。
初始组只读，末尾组单独排空。这一顺序修正也作用于旧的 db 候选。
事件约束的是对应流水线；是否真实重叠仍需生成代码和设备流水确认。

下面只运行所选两种方法，不执行 SIMT、默认 tileops 或其他实验：

```bash
python scripts/benchmark_ascend.py \
  --device 3 --unlocked --owner gather/overlap \
  --run-dir "./benchmark-results/overlap-$(date +%Y%m%d-%H%M%S)" \
  --op PagedKVCacheGatherFwdOp \
  --case-regex "^(ds-v3-mixed|llama-70b-cp|mla-small|llama-cp-unaligned|mla-b256)-" \
  --implementation-factory benchmarks.ascend.gather_auto:make_overlap_implementations \
  --implementations tilelang-batch16-r4 tilelang-overlap32-r4
```

先加 `--verify-only`。只验证新候选的 pytest 命令：

```bash
python -m pytest tests/ops/test_paged_cache_gather_batch_ascend.py \
  --tileops-device=npu -k overlap32-r4 -q
```

流水脚本使用 `--implementation overlap32-r4 --source-dir ./pipeline-results/overlap-source`。
重点查看：上一组 ub2gm 是否在本组地址准备及 gm2ub 之前提交，两个区域的
release 等待是否只发生在复用之前。相邻组地址不重叠不代表硬件一定并行。
本地没有 NPU，仅完成语法、diff 和事件配对模型检查。

## 写出缓存策略 4/0 对照

本地 TileLang 源码核查结果：

- `tilelang/src/ascend/op/copy.cc` 的 `kStoreDefaultL2CacheCtrl` 为 4。
- `tilelang/tilelang/ascend/language/copy_op.py` 中 Store 的 4 对应
  `NOTALLOC_CI`，0 对应 `NORMAL_FV`。
- 本地 GatherV2 `gather_v2_simd.h::CopyOut` 使用 `DataCopyPad`，未显式设置
  L2 策略；仅凭此处不能确认 SDK 的最终默认值或已安装二进制的实际策略。

增加仅改变写出缓存模式的实验。两者均为 overlap32-r4，读入策略、
UB 布局、事件顺序和搬运大小一致。模式 0 不保证更快，不能把结果预先归因于缓存。

```bash
python scripts/benchmark_ascend.py \
  --device 3 --unlocked --owner gather/store-policy \
  --run-dir "./benchmark-results/store-policy-$(date +%Y%m%d-%H%M%S)" \
  --op PagedKVCacheGatherFwdOp \
  --case-regex "^(ds-v3-mixed|llama-70b-cp|mla-small|llama-cp-unaligned|mla-b256)-" \
  --implementation-factory benchmarks.ascend.gather_auto:make_store_policy_implementations \
  --implementations tilelang-overlap32-r4-store4 tilelang-overlap32-r4-store0
```

先追加 `--verify-only`。不运行其他候选。流水脚本用
`--implementation overlap32-r4 --store-l2-cache-ctrl 0`（对照为 4），
导出代码确认所有 `asc_copy_ub2gm_align` 的 mode 为指定值，
同时记录单次 64 KiB 写出的起止周期及对应整组读入跨度。
附件提供了寄存器和源码，没有时长字段；尚不能计算真实搬运耗时。
本地无 NPU，已通过语法/diff 检查，未进行设备编译或性能验证。

## 当前对照：store0 下的单缓冲与双缓冲

| 实现 | UB 区域 | 每次读入 | 每次完整写出 | 写出模式 |
| --- | --- | --- | --- | --- |
| `tilelang-batch16-r4-store0` | 单区域 64 行 | 4 行 | 64 行 | 0 |
| `tilelang-overlap32-r4-store0` | 两区域，各 32 行 | 4 行 | 32 行 | 0 |

两者 payload 总容量相同：BF16、行宽 1024 时为 128 KiB。
该对照同时改变区域轮转和单次写出粒度，用于选择整体调度；不能把全部差值
单独归因为双缓冲。上一轮 store4/store0 实验两者都开了双缓冲。

只执行并汇总这两个候选：

```bash
python scripts/benchmark_ascend.py \
  --device 3 --unlocked --owner gather/store0-overlap \
  --run-dir "./benchmark-results/store0-overlap-$(date +%Y%m%d-%H%M%S)" \
  --op PagedKVCacheGatherFwdOp \
  --case-regex "^(ds-v3-mixed|llama-70b-cp|mla-small|llama-cp-unaligned|mla-b256)-" \
  --implementation-factory benchmarks.ascend.gather_auto:make_store0_overlap_implementations \
  --implementations tilelang-batch16-r4-store0 tilelang-overlap32-r4-store0
```

先加 `--verify-only` 验证；去掉后测性能。不运行公共 tileops、SIMT 或其他实验。
独立数值参考仍用于正确性校验。

流水采集分别使用下面两个选择，均显式设置模式 0：

```bash
python scripts/profile_gather_pipeline.py \
  --implementation batch16-r4 --store-l2-cache-ctrl 0 \
  --source-dir ./pipeline-results/single-store0-source --check

python scripts/profile_gather_pipeline.py \
  --implementation overlap32-r4 --store-l2-cache-ctrl 0 \
  --source-dir ./pipeline-results/overlap-store0-source --check
```

以上为应用命令；simulator 外层命令及预约规则沿用流水 README。
MTE3 显示轨道数量不等同缓冲区数量或硬件引擎数量，应检查实际轨道标签及读写重叠。
