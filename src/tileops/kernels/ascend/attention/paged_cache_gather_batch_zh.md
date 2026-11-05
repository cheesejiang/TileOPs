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

同一轮对照（按实际空闲卡修改 device）：

```bash
python scripts/benchmark_ascend.py \
  --device 3 --unlocked --owner gather/batch \
  --run-dir "./benchmark-results/batch-$(date +%Y%m%d-%H%M%S)" \
  --op PagedKVCacheGatherFwdOp \
  --implementation-factory benchmarks.ascend.gather_auto:make_implementations
```

现有 manual/auto 候选保留，五个非 FP8 规格额外产生 tilelang-batch2/4。
需要先验证时追加 `--verify-only`。测试应覆盖修改页表/起点后的重复调用、
跨页、空请求以及不足一组的输出。真实耗时以同卡同轮测试为准。

流水采集脚本新增 `--implementation batch2` 和 `batch4`，其余参数沿用
scripts/profile_gather_pipeline_README.md。传入 `--source-dir` 可导出生成代码。
使用 msopprof simulator 时不要同时开启脚本的 `--profile-dir`。
