# PagedKVCacheGatherFwdOp：遵循原仓协议的950DT验证

本分支只增加最终算子、针对性的正确性测试，以及必要的FP8参考/输入保护兼容修复。
不提供独立runner、guard、未锁频模式或开发版计时器。验证复用原仓：

- [开发和共享benchmark协议](development.md)
- [工具链约束](toolchain.md)及`config/ascend950-toolchain.json`
- `scripts/npu_guard.py`、`scripts/benchmark_ascend.py`
- `benchmarks/ascend/runner.py`、`clocks.py`、`timing.py`

## 1. 代码与环境

在服务器获取私仓的950DT-test分支。以下假设私仓远端名为jyw：

```bash
git fetch jyw
git switch 950DT-test
git pull --ff-only jyw 950DT-test
```

首次没有本地分支时使用`git switch -c 950DT-test --track jyw/950DT-test`。
服务器私仓远端如果叫origin，对应替换。已有结果和本地代码修改先备份，禁止强制清理。

使用站点已准备、符合toolchain lock的环境，不自行替换共享CANN运行库。
原仓canonical profile：Python3.12.15、torch2.13.0+cpu、torch_npu2.13.0rc1、
官方ARM64 TileLang0.1.15、CANN编译器9.2.0 release及独立9.1.0 runtime/OPP、
GCC12.3.0、Ascend950DT_9582物理32Cube/64Vector。以锁文件的指纹检查为准，
不是只匹配版本字符串。36核机器不替代该验收硬件。

激活管理员指定的环境脚本后，在仓库根目录检查工具链（不初始化设备）：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/ci/check_ascend_toolchain.py
```

原仓列出的环境脚本路径是canonical部署位置，不是所有服务器均存在的通用路径。
若不匹配，按报错准备正确环境，不通过修改检查器/锁文件绕过。
当前源码可编辑安装到同一路径时无需重装；未安装时在已准备环境下执行
`python -m pip install -e . --no-deps`。

## 2. 原仓设备保护

默认guard要求设备0的隔离视图和真实host PID/共享锁可见性，所有参与者共用
`/run/lock/ascend-npu/device-0.lock`。不支持旧开发脚本的`--device 3`。

```bash
python scripts/npu_guard.py status
```

物理映射不同的站点必须提供经审计且实现相同锁和进程归属契约的guard；
原仓允许`--guard /绝对路径/npu_guard.py`或`TILEOPS_NPU_GUARD`。
不通过环境变量伪装为独占卡，不删除持有中的锁，不重置设备。
NPU导入、编译及执行都应包含在站点guard保护的整个前台进程内。

## 3. 七规格21项针对性正确性测试

```bash
python scripts/npu_guard.py run \
  --owner gather/950dt-pytest --wait 600 --timeout 7200 -- \
  python -m pytest tests/ops/test_paged_cache_gather_ascend.py \
    --tileops-device=npu -q
```

预期21passed：manifest覆盖、6个拒绝条件、7规格×连续/非连续dst。
设备用例修改页表/起点/scale重放，并检查输出视图间隙不被写坏。
本分支保留两个必要修复：NPU FP8参考通过uint8收集，再按原dtype反量化；
FP8输入未修改检查使用精确字节比较。原数学参考和精度标准不变，
最新上游复数输入保护检查也保留。

## 4. 原仓launcher七规格数值验证

```bash
python scripts/benchmark_ascend.py \
  --owner gather/950dt-verify --timeout 7200 \
  --run-dir "/shared/results/gather-950dt-verify-$(date +%Y%m%d-%H%M%S)" \
  --op PagedKVCacheGatherFwdOp --verify-only
```

正确性模式按原协议无需clock-config，不产生性能验收数据。
receipt.json应为passed，results.jsonl中七规格全部verified。

## 5. 按站点锁频策略采集正式性能

先向管理员取得经过验证的锁频配置文件。原仓约定其包含：

- `lock`：锁频工具的绝对路径argv数组。
- `restore`：恢复原策略的绝对路径argv数组，部分锁频失败也须能恢复。
- `component_mhz`：两个硬件组件的目标MHz，不能超过额定频率。

不能填入未经验证的命令、空操作或猜测频率。原仓会独立读取两个组件频率，
在采样期间校验并在退出时恢复；没有有效站点配置就不能进行正式计时。
`/site/ascend950-clocks.json`是原文示例路径，必须替换为实际文件。

```bash
export ASCEND950_CLOCK_CONFIG=/站点提供的实际路径/ascend950-clocks.json

python scripts/benchmark_ascend.py \
  --owner gather/950dt-perf --timeout 7200 \
  --run-dir "/shared/results/gather-950dt-perf-$(date +%Y%m%d-%H%M%S)" \
  --clock-config "$ASCEND950_CLOCK_CONFIG" \
  --op PagedKVCacheGatherFwdOp
```

不加factory即只测最终公共tileops，覆盖全部七规格。不使用--unlocked、
--device或--implementations，也不修改固定预热/重复次数。
原协议ascend950-profiler-locked-v1：正确性/JIT之后3次预热、15个独立profiler
窗口、每次reset与1GiB缓存驱逐/同步均位于采样外，均值为设备忙区间并集，
重叠只计一次；无event或wall-clock回退。不合并旧unlocked测量。

## 6. 验收结果与CI

保留完整run-dir，包括receipt.json、result.json、results.jsonl、environment.json、
sources.json、clock-readbacks.json、run.log及所有原始profiles。原入口不生成新增开发版summary.md。
必须receipt.json=passed且result.verify_only=false，七规格均measured，才能引用性能。
任何正确性、频率、污染、源代码变化或guard失败都使结果失效，不能只截取先前成功行。
guard退出75/76/78无效。保存所有15次原始样本，不删样、不修剪异常值。

原仓全量CI入口保持不变（这是所有注册算子的检查，不只本算子）：

```bash
python scripts/ci/ascend950.py --phase cpu --output /results/unique-cpu-run
python scripts/ci/ascend950.py --phase device --output /results/unique-device-run
```

CPU可额外运行FP8兼容补丁回归：

```bash
python -m unittest discover -s tests -p test_fp8_input_preservation.py -q
```

本地无NPU和站点锁频配置，尚未执行canonical设备正确性或锁频性能验收。
