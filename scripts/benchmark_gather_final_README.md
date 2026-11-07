# 950DT-test：最终算子的正确性和性能验证

本分支基于核心提交分支 paged-gather，算子及注册文件不变，只补独立验证工具
与FP8参考/输入保护兼容修复。原始实验、参数扫描及外部基线不恢复。
所有数据记录实际型号与核数，不将36核数据标为32核。

## 1. 服务器获取分支

已有私仓时，先确认 jyw/origin 哪个指向自己的仓库；以下假设私仓远端名为 jyw：

```bash
git fetch jyw
git switch --track -c 950DT-test jyw/950DT-test
```

本地已有该分支时使用 `git switch 950DT-test`，再执行
`git pull --ff-only jyw 950DT-test`。已有结果文件先保留，禁止git clean或强制切分支。
若私仓远端实际叫origin，把上面的jyw换为origin。

## 2. 环境与设备准备

激活此前能运行TileLang Ascend的环境，加载**该服务器实际安装的CANN环境脚本**，
不要直接套用另一台机器的/home/CANN路径，也不要把simulator的lib放在实机运行库前。

```bash
conda activate 你的测试环境
source /实际CANN安装目录/set_env.sh
cd /实际位置/TileOPs
python -m pip show tileops tilelang torch torch-npu
```

如果tileops已可编辑安装到这个同一路径，不必重装；否则：

```bash
python -m pip install -e . --no-deps
```

不需要vllm、ops-nn动态库或Triton；本次只测TileOPs最终实现。
环境版本不同仍由unlocked开发模式记录，不声称与旧机器完全相同。
先查询设备及空闲状态（NPU_DEVICE必须设为管理员分配的物理卡）：

```bash
npu-smi info
export NPU_DEVICE=0
python scripts/npu_guard_gather.py status --device "$NPU_DEVICE"
```

多卡环境不要把宿主物理卡号当作子进程逻辑卡号。guard会设置可见性，使
选中的物理卡映射为子进程npu:0；后续代码不要手动改为npu:3等。
不要设置成未经分配的卡。waiting reservation表示预约排队，并未开始编译。

## 3. 七规格21项正确性检查

```bash
python scripts/npu_guard_gather.py run \
  --device "$NPU_DEVICE" --owner gather/950dt-pytest --wait 600 --timeout 7200 -- \
  python -m pytest tests/ops/test_paged_cache_gather_ascend.py \
    --tileops-device=npu -q
```

预期21 passed：manifest覆盖、6个拒绝条件、7规格连续/非连续dst。
设备用例会修改页表/起点/scale重放，非连续输出检查间隙不被写坏。

## 4. 带收据的七规格验证与设备记录

```bash
python scripts/benchmark_gather_final.py \
  --device "$NPU_DEVICE" --unlocked --owner gather/950dt-verify --timeout 7200 \
  --run-dir "./benchmark-results/950dt-verify-$(date +%Y%m%d-%H%M%S)" \
  --op PagedKVCacheGatherFwdOp --verify-only
```

检查receipt.json为passed、七条结果均verified，以及environment.json的
`device`、`physical_device`和`core_counts`。核心不发生主机读取序列长度。
36核机器可以完成此步骤，不可将其当作32核性能验收。

## 5. 32核950DT性能采集

确认管理员分配的是32核950DT后运行：

```bash
python scripts/benchmark_gather_final.py \
  --device "$NPU_DEVICE" --unlocked --owner gather/950dt-perf --timeout 7200 \
  --expected-cube-cores 32 \
  --run-dir "./benchmark-results/950dt-perf-$(date +%Y%m%d-%H%M%S)" \
  --op PagedKVCacheGatherFwdOp
```

`--expected-cube-cores 32`只验证运行时报告的物理Cube核数，不限制核数，
不能把36核设备变成32核；查询不到或不符合会停止并在run.log报错。
算子实际按设备报告的Vector核数启动，environment.json同时记录Cube/Vector核数。
如果只有36核开发机，可以去掉该检查做开发测试，目录命名改成950dt-36core-dev，
报告必须标明实际核数，不作比例缩放后当成32核结果。

模式为unlocked开发测量：不改变频率、不宣称锁频验收。需要正式锁频时，
必须使用管理员提供且核实过的--clock-config，不能与--unlocked一起使用。
每规格3次预热、15次采样、1GiB缓存驱逐；只测公共tileops、不运行外部基线。
首次编译较慢，等编译/七规格执行完成；不要同时运行其他NPU任务。

## 6. 结果确认与重复测量

输出run-dir中应有receipt.json、summary.md、results.jsonl、environment.json、
sources.json、profiles/和run.log。receipt passed，七规格全部measured、数值校验通过
才作为有效运行；失败时查看results.jsonl内error和run.log，不只看summary.md。
BW是逻辑读cache＋写dst字节数/时间，不是实测HBM流量。

建议同一环境重复性能命令3次，每次新目录，分别保存均值/标准差和实际核数。
不要直接合并不同核数、频率策略或软件环境的测量。性能验证前不再修改源码；
runner会检查执行期间的源码指纹变化。
