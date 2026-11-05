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
当前不提供 batch4 双缓冲，避免其 256 KiB payload 再加元数据造成容量问题。

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
