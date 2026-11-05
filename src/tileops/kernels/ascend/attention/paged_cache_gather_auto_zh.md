# 分页 Gather 自动调度实验版本

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
5. stages=2 的每块固定为 16 行，控制双缓冲 UB 占用；stages=1 保持原始块大小。
6. 保留原有设备端二分查找、页表映射、逐行慢路径、尾部保护和 FP8 数值逻辑。
   主机 forward 与输入检查继承手工版本，不重复实现。

自动调度不代表自动搜索最优 tile 或保证 DMA/计算重叠。
auto2 同时改变了缓冲版本和部分规格的块大小，不能将其收益全部归因于自动同步。
若要严格隔离因素，应在后续实验中统一 tile_rows 后再比较。

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
summary.md 每个规格会输出 `tileops`（手工）、`tilelang-auto1`、`tilelang-auto2` 三行。
所有候选都通过同一参考实现精确检查，不静默回退到手工内核。

本机缺少可导入的 `tilelang.ascend`、torch_npu 与 CANN，尚未实机编译或测量。
动态二分循环、数据依赖地址、条件 DMA 和尾部分支均增加了自动依赖分析难度，
必须确认安装版本支持这些调度模式。若编译失败，需保留完整异常，不能假设开关翻转即已成功。
在完成 7 个规格精度与性能验证前，不将实验版本注册为默认实现。
