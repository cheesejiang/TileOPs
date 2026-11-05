# 分页 Gather 的 msopprof simulator 流水分析

配套脚本：`scripts/profile_gather_pipeline.py`。
以下命令均在 TileOPs 仓库根目录执行，使用支持 Ascend 950 的 `msopprof simulator`。

## 环境与采集对象

先激活此前测试通过的 Python、torch_npu、TileLang 和 CANN 环境，并确认：

```bash
which python
msopprof simulator --help
```

按站点要求使用空闲设备及预约机制。仿真库与环境设置遵循当前 msopprof 的安装说明，
不要直接套用其他芯片版本的 simulator 库路径。

| implementation | 对象 | 工作范围 |
| --- | --- | --- |
| `manual` | 默认 TileLang 手工同步版本 | 页表寻址和数据搬运 |
| `auto2` | 自动调度、16 行双缓冲版本 | 页表寻址和数据搬运 |
| `gatherv2-core` | 指定 CANN 库中的 ACLNN GatherV2 | 物理索引预生成，只收集数据 |

脚本固定使用 `llama-70b-cp`：BF16，cache `[4096,16,8,128]`，
dst `[32768,8,128]`，8 个请求各读 4096 行，逻辑起点均为 2048。
它不执行 benchmark 的预热循环、1 GiB 缓存驱逐或嵌套 profiler。
输入在 CPU 生成后传到 NPU，每次进程执行一次目标实现；输入准备仍可能产生设备活动。
仿真采集用于解释流水，不替代实机 benchmark 的延迟结果。

## 先确定实际内核名

TileLang 两种实现可以在正常 NPU 环境运行并导出代码：

```bash
python scripts/profile_gather_pipeline.py \
  --implementation auto2 --source-dir ./pipeline-source --check

python scripts/profile_gather_pipeline.py \
  --implementation manual --source-dir ./pipeline-source --check
```

查看 `pipeline-source/auto2.cpp` 和 `manual.cpp` 的实际设备函数名，
并与实机 profiler 中的 kernel name 核对。
两者可能使用相同名字，但在不同进程、不同输出目录中采集。
`gatherv2-core` 的实际内核名需从此前实机 profiler 的任务记录中确认。

**建议使用 `--kernel-name` 筛选。** 默认 `--launch-count 1` 限定收集次数，
不能保证自动选中目标 Gather；没有筛选时可能采到准备阶段的其他设备内核。
若 ACLNN GatherV2 包含 Gather、ViewCopy 等多个内核，应分别采集，
不能将一个内核的流水当作整个接口。

## 采集 auto2

先将下面变量替换成上一步核对的实际名称：

```bash
GATHER_AUTO_KERNEL=实际的auto2内核名称

msopprof simulator \
  --output "./pipeline-results/auto2-$(date +%Y%m%d-%H%M%S)" \
  --soc-version Ascend950 \
  --aic-metrics PipeUtilization \
  --kernel-name "$GATHER_AUTO_KERNEL" \
  --launch-count 1 \
  --timeout 120 \
  "$(command -v python)" \
  scripts/profile_gather_pipeline.py \
  --implementation auto2 \
  --source-dir ./pipeline-source
```

## 采集手工版本

```bash
GATHER_MANUAL_KERNEL=实际的manual内核名称

msopprof simulator \
  --output "./pipeline-results/manual-$(date +%Y%m%d-%H%M%S)" \
  --soc-version Ascend950 \
  --aic-metrics PipeUtilization \
  --kernel-name "$GATHER_MANUAL_KERNEL" \
  --launch-count 1 \
  --timeout 120 \
  "$(command -v python)" \
  scripts/profile_gather_pipeline.py \
  --implementation manual \
  --source-dir ./pipeline-source
```

## 采集 GatherV2 core

先沿用此前测试成功的动态库设置，以下均为路径示例：

```bash
export OPS_NN_OPAPI_LIBRARY=/实际路径/libopapi_nn.so
export OPS_NN_TENSOR_LIBRARY=/实际路径/libnnopbase.so
GATHER_CANN_KERNEL=实际的GatherV2内核名称

msopprof simulator \
  --output "./pipeline-results/gatherv2-core-$(date +%Y%m%d-%H%M%S)" \
  --soc-version Ascend950 \
  --aic-metrics PipeUtilization \
  --kernel-name "$GATHER_CANN_KERNEL" \
  --launch-count 1 \
  --timeout 120 \
  "$(command -v python)" \
  scripts/profile_gather_pipeline.py \
  --implementation gatherv2-core
```

CANN 版本不提供源码导出，因此不要传 `--source-dir`。
core 不计入物理索引生成，仅作为搬运路径参考；它和 TileLang 完整流程的工作范围不同。

## 参数和结果

- `--soc-version Ascend950`：需匹配本机工具支持的型号字符串；若不接受，按工具说明替换，不能猜测其他型号。
- `--aic-metrics PipeUtilization`：本次采集流水利用率；950 不支持帮助中注明的 PMSampling、Overhead 项。
- `--timeout 120`：工具超时上限，单位为分钟；不是预期执行时间。
- `--check`：脚本可选的 CPU 参考结果检查；先在正常环境检查，流水采集命令默认不加，减少回传干扰。

结果保存在 `--output` 指定目录下的 `OPPROF_<timestamp>_<random>/`。
流水文件的名称和展示方式以本机 msopprof 输出及配套分析工具为准。
已有原始模拟数据可以通过 export 模式解析；不与 application 模式同时指定：

```bash
msopprof simulator --export /实际模拟数据目录 --output ./pipeline-results/exported
```

对照时重点检查：

1. 目标 kernel 是否确实被采集，Block Dim 与实际执行核是否符合预期。
2. auto2 的 MTE2 读取与 MTE3 写出是否重叠，以及双缓冲切换是否引入等待。
3. DMA 是否为预期的多行批量搬运，是否被拆成较多小操作。
4. 标量寻址、同步等待和流水空闲分别出现在哪里。
5. 不同执行核的任务数量是否均衡，不只观察 core 0 就概括全部执行。

采集命令已按用户提供的帮助格式编写；本地主机没有 msopprof/CANN/NPU，尚未实际执行验证。
