# 分页 Gather 自动调度实验版本

新增 32 行双缓冲候选 `tilelang-auto2-r32`，仅用于页大小为 64 的规格。
原有 `tilelang-auto1` / `tilelang-auto2` 均保留 16 行，便于在同次运行中对照。
新候选仍包含非对齐起点和输出尾部保护，需独立通过精度测试。
32 行、576 宽的 FP8 双缓冲输入和输出逻辑数据量合计约 108 KiB，
另有元数据和编译器临时空间，实际 UB 占用以生成代码为准。

设备实现：`paged_cache_gather_auto.py`。
对比适配器：`benchmarks/ascend/gather_auto.py`。
默认注册的手工内核 `paged_cache_gather.py` 保持不变。

## 修改范围

1. 删除全部 `ascend_set_flag / ascend_wait_flag`，不把手写事件混入自动同步。
2. 启用 `TL_ENABLE_AUTO_SCHEDULE`，由编译器分析标量、DMA、Vector 依赖并插入同步。
3. 将 `TL_DISABLE_SHARED_MEMORY_REUSE` 和 `TL_DISABLE_THREAD_STORAGE_SYNC` 设为 false，
   不再显式禁用这些编译器处理；具体 Ascend pass 顺序以实际安装版本为准。
4. 使用 `T.Pipelined` 描述任务循环，提供 stages=1/2 两种配置。
   stages=2 对 src/converted 标注两个缓冲版本，让编译器安排轮换及依赖。
5. stages=1 和 stages=2 每块都固定为 16 行，保持任务划分和搬运粒度一致。
6. 保留原有设备端二分查找、页表映射、逐行慢路径、尾部保护和 FP8 数值逻辑。
   主机 forward 与输入检查继承手工版本，不重复实现。

自动调度不代表自动搜索最优 tile 或保证 DMA/计算重叠。
auto1 与 auto2 现在使用相同的 tile_rows，仅改变 stage 数及缓冲版本配置。
早期版本的 auto1 在无起点且页大小为 64 时使用 64 行；这部分旧数据不能与新版混用。
手工默认版本未修改，比较手工版与自动版时仍需考虑部分规格块大小不同的影响。

## 验证与测量

在 TileOPs 根目录，使用现有 Ascend 环境执行：

```bash
python -m pytest tests/ops/test_paged_cache_gather_auto_ascend.py --tileops-device=npu
```

包含 7 个规格 × 2 种 stage 配置，每项都复用内核并修改元数据重跑。
先运行公共 benchmark 正确性检查：

```bash
python scripts/benchmark_ascend.py \
  --device 3 --unlocked --verify-only \
  --owner gather/auto-verify \
  --run-dir "./benchmark-results/auto-verify-$(date +%Y%m%d-%H%M%S)" \
  --op PagedKVCacheGatherFwdOp \
  --implementation-factory benchmarks.ascend.gather_auto:make_implementations
```

通过后去掉 `--verify-only`，改用新的目录和 owner 进行性能测试。
summary.md 每个规格会输出 `tileops`（手工）、`tilelang-auto1`、`tilelang-auto2`，
页大小为 64 时再增加 `tilelang-auto2-r32`。
所有候选都通过同一参考实现精确检查，不静默回退到手工内核。

本机缺少可导入的 `tilelang.ascend`、torch_npu 与 CANN，尚未实机编译或测量。
动态二分循环、数据依赖地址、条件 DMA 和尾部分支均增加了自动依赖分析难度，
必须确认安装版本支持这些调度模式。若编译失败，需保留完整异常，不能假设开关翻转即已成功。
在完成 7 个规格精度与性能验证前，不将实验版本注册为默认实现。
