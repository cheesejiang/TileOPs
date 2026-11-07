# 最终 PagedKVCacheGatherFwdOp 验证

从最新主分支恢复的独立验证工具；不改写上游通用 benchmark、guard、锁频协议。
只验证当前已注册的公共算子，不导入历史 sweep/auto/SIMT 或外部基线。
配套文件为 scripts/benchmark_gather_final.py、scripts/npu_guard_gather.py、
benchmarks/ascend_gather_final/。算子核心提交与验证工具可以分别审查。

先在服务器激活原来验证通过的环境，加载对应 CANN set_env.sh，在 TileOPs
根目录安装当前 checkout：

```bash
python -m pip install -e . --no-deps
```

## 七规格正确性及输出视图

独立 guard 按物理卡号预约，子进程将选中卡映射为逻辑 npu:0。
下面 --device 3 按实际空闲卡修改。不要同时启动多个验证任务。

```bash
python scripts/npu_guard_gather.py run \
  --device 3 --owner gather/final-pytest --wait 600 --timeout 7200 -- \
  python -m pytest tests/ops/test_paged_cache_gather_ascend.py \
    --tileops-device=npu -q
```

21个测试：1个manifest覆盖、6个拒绝条件、7规格×连续/非连续dst；
设备用例每个还会修改页表/起点/scale重放，并检查非连续dst视图间隙不被写坏。
这21个测试不包括旧分支的实验候选，不需恢复其他测试文件。

两个必要验证兼容补丁：workloads/attention/paged_cache_gather.py 对NPU FP8
通过uint8视图收集参考数据；workloads/numerics.py 用字节比较检查FP8输入保护。
不改数学参考、精度标准或输入生成，保留最新上游的复数检查逻辑。
对应CPU回归为 tests/test_fp8_input_preservation.py。

## 受保护 launcher 验证

```bash
python scripts/benchmark_gather_final.py \
  --device 3 --unlocked --owner gather/final-verify --timeout 7200 \
  --run-dir "./benchmark-results/gather-final-verify-$(date +%Y%m%d-%H%M%S)" \
  --op PagedKVCacheGatherFwdOp --verify-only
```

## 最终七规格性能

```bash
python scripts/benchmark_gather_final.py \
  --device 3 --unlocked --owner gather/final-perf --timeout 7200 \
  --run-dir "./benchmark-results/gather-final-perf-$(date +%Y%m%d-%H%M%S)" \
  --op PagedKVCacheGatherFwdOp
```

不加工厂即只运行 tileops。新目录保存receipt.json、results.jsonl、environment.json、
summary.md及原始profiles；终端同时打印表格。成功应为receipt passed且七规格均
measured，verify-only时为verified。unlocked为开发测量，不进行锁频，不宣称正式结果；
3次预热、15次采样、1GiB驱逐，统计设备执行区间并集。
BW为逻辑读取cache与写dst字节数/耗时，不是硬件实测HBM带宽。

旧入口 scripts/benchmark_ascend.py 保持上游原样，不支持这套多卡开发参数。
独立入口仍验证设备空闲和预约归属，不绕过其他使用者的锁；waiting reservation
时另一个终端可运行：

```bash
python scripts/npu_guard_gather.py status --device 3
```

CPU工具回归：

```bash
PYTHONPATH=. python -m unittest discover -s benchmarks/tests -p 'test_gather_final_*.py' -q
python -m unittest discover -s tests -p test_fp8_input_preservation.py -q
```

本地9项launcher/多卡检查及2项FP8输入保护通过；本地无Ascend NPU，设备测试
与性能尚待运行。新工具未提供外部baseline适配，也未恢复旧实验。
