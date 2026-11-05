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

推荐直接复用相同输入采集一次实机 profiler，并自动解析。
使用已配置的 GatherV2 动态库环境，在站点支持的按卡保护程序下执行：

```bash
python scripts/npu_guard.py run --device 3 --owner gather/kernel-name -- \
  python scripts/profile_gather_pipeline.py \
    --implementation gatherv2-core \
    --profile-dir "./pipeline-results/kernel-name-$(date +%Y%m%d-%H%M%S)" \
    --check
```

无需修改物理/逻辑卡号：guard 将所选物理卡映射为子进程的逻辑 npu:0。
profile-dir 必须是新目录。脚本在采样外预热一次，再只采集一次目标调用，
启用自动解析并打印原始采样位置、生成的 CSV/JSON/DB 路径及 op_summary 名称列。
如无 op_summary，请保留终端的分析警告和打印出的原始目录，不要默认认为名称为空。
可将 implementation 换成 manual 或 auto2，对它们同样采集实际名称。
`--profile-dir` 仅用于实机名称查询，运行 msopprof simulator 时不要传该参数，避免嵌套 profiler。

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
GATHER_AUTO_KERNEL=main_kernel

msopprof simulator \
  --output "./pipeline-results/auto2-$(date +%Y%m%d-%H%M%S)" \
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
GATHER_MANUAL_KERNEL=main_kernel

msopprof simulator \
  --output "./pipeline-results/manual-$(date +%Y%m%d-%H%M%S)" \
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

先沿用此前测试使用的 CANN 已安装版本动态库设置。
此前提供的完整路径如下，在该服务器上使用前确认文件存在：

```bash
export OPS_NN_OPAPI_LIBRARY=/home/CANN/9.3.0.weekly.20260916/cann-9.3.0/x86_64-linux/lib64/libopapi_nn.so
export OPS_NN_TENSOR_LIBRARY=/home/CANN/9.3.0.weekly.20260916/cann-9.3.0/x86_64-linux/lib64/libnnopbase.so
GATHER_CANN_KERNEL=GatherV2_4c62aa93d3ed2b471b7de5c999eaf660_high_precision_1000000299

msopprof simulator \
  --output "./pipeline-results/gatherv2-core-$(date +%Y%m%d-%H%M%S)" \
  --aic-metrics PipeUtilization \
  --kernel-name "$GATHER_CANN_KERNEL" \
  --launch-count 1 \
  --timeout 120 \
  "$(command -v python)" \
  scripts/profile_gather_pipeline.py \
  --implementation gatherv2-core
```

CANN 版本不提供源码导出，因此不要传 `--source-dir`。
实机报告的上层任务名称如下，它与 simulator 用于筛选的设备符号不同：

```text
Device_id: 3
Name: aclnnGatherV2_GatherV2AiCore_GatherV2
Duration(us): 45.115
Block Num: 56
```

45.115 μs 是单次调用的记录，不替代 benchmark 的 15 次平均耗时。
Block Num 是启动块数，不等于设备物理核总数。
本次 simulator 实际输出：

```text
Kernel GatherV2_4c62aa93d3ed2b471b7de5c999eaf660_high_precision_1000000299 skipped: not selected via --kernel-name
```

因此上面的采集命令已改为该实际设备符号；不要用实机报告中的
`aclnnGatherV2_GatherV2AiCore_GatherV2` 作为本次 simulator 的筛选名称。
如果日志显示所有内核被过滤，不能将“应用执行成功”当作“已采集到流水”。
不同 CANN 版本、二进制或输入配置可能改变哈希及 tiling 后缀；届时以日志中的实际名称为准。
注意：上述 op-api 路径在 `cann-9.3.0/` 下，此前查到的 simulator 路径在
`cann/x86_64-linux/simulator/` 下。不要直接将两者互换；先核对安装关系和版本兼容性。
要确认某次成功测量实际加载的库，以其 `results.jsonl` 中的 `provenance.library` 为准。
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
