# 使用 ops-nn GatherV2 对比非 FP8 分页 Gather

适配器位于 `benchmarks/ascend/gather_v2_baseline.py`，直接通过 ctypes 调用
指定动态库的 `aclnnGatherV2GetWorkspaceSize` 和 `aclnnGatherV2`。
沿用当前 torch_npu 逻辑设备的流，不创建第二个 ACL 上下文，不使用 CPU 参考结果作为输出。
数据输入由同一 manifest workload 提供，精度验证和测量由现有 runner 完成。

## 两种测量范围

| 汇总表中的实现名 | 计时范围 | 能否作为完整分页算子对比 |
| --- | --- | --- |
| `tileops` | 现有 TileLang 分页 gather | 是 |
| `ops-nn-gatherv2-full` | 设备端解析请求/起点/页表、生成物理行索引，再执行 ACLNN GatherV2 | 是，属于组合基线 |
| `ops-nn-gatherv2-core` | 只执行 ACLNN GatherV2；物理行索引提前生成 | 否，只用于分析纯收集性能 |

`full` 的索引生成采用 PyTorch NPU 原语，Gather 数据搬运直接调用指定 ACLNN 库。
因此它是基于 CANN 算子的组合流程，不是单个原生融合分页内核。
参数形状决定的行号 arange 提前准备；依赖设备元数据的请求编号、逻辑位置和物理索引
在 full 每次调用时重新计算，全部计入设备执行时间。
输出由调用者侧预分配，输出填 NaN 的 reset 不计时；不包含 CPU 调度间隙或主机 API 开销。
ACLNN 内部可能产生多个内核，包括 ViewCopy，以 profiler 实际记录为准。

测试范围限定为以下 5 个非 FP8 workload：

- `ds-v3-mixed`：BF16，129024×576 行数据。
- `llama-70b-cp`：BF16，32768×8×128，包含起点。
- `mla-small`：FP16，31×576。
- `llama-cp-unaligned`：BF16，1024×8×128，非对齐起点。
- `mla-b256`：BF16，131072×576，256 个请求。

## 1. 选择测试来源（二选一）

这里有两个可选来源，不是一次测试必须执行两遍：

| 类别 | 测试对象 | 是否需要编译 ops-nn |
| --- | --- | --- |
| A：CANN 已安装版本 | 服务器已有的预编译 GatherV2 | 不需要 |
| B：当前 ops-nn 源码构建版本 | 从指定源码提交构建并安装的 GatherV2 | 需要 |

克隆源码不会自动替换 CANN 中的实现。若目标是测刚克隆的 ops-nn，选择 B；
若只想测试当前服务器提供的 GatherV2，选择 A。
完成任一类别的准备后，执行第 2、3 节共用的验证和测量命令。
full/core 是计时范围，与 A/B 两种库来源是独立的概念：任一来源都会测 full 和 core。

### 共用环境准备

在运行测试的 Ascend 服务器上，复用之前通过 TileLang 测试的环境：

```bash
conda activate jyw_ops
source /实际路径/Ascend环境配置脚本.sh
which python
echo "$ASCEND_HOME_PATH"
echo "$ASCEND_OPP_PATH"
```

将环境名和脚本路径替换为实际配置。如果已有项目专用的 CANN SDK/Runtime 混合配置脚本，
继续加载该脚本，不要随意 source 另一套 CANN 的环境。
A 类需要可运行的 CANN、torch_npu 和现有 TileLang 环境；
B 类还需要满足 ops-nn 构建文档要求的编译器、CMake 和第三方依赖。
两类都无需仅为了该基线重新安装 TileOPs 或覆盖已验证的 Python 依赖。

### A 类：使用 CANN 已安装版本（无需编译）

查找提供 GatherV2 的库（实际路径随 CANN 版本变化）：

```bash
find "$ASCEND_HOME_PATH" \( -name 'libopapi*.so' -o -name 'libnnopbase.so' \) 2>/dev/null
nm -D /实际路径/libopapi_nn.so | rg 'aclnnGatherV2(GetWorkspaceSize)?$'
ldd /实际路径/libopapi_nn.so
export OPS_NN_OPAPI_LIBRARY=/实际路径/libopapi_nn.so
```

确认库提供 `aclnnGatherV2` 和 `aclnnGatherV2GetWorkspaceSize`，依赖中没有 `not found`。
`libopapi_nn.so` 是文件名示例，实际也可能由其他 op-api 库提供接口。
如果 `$ASCEND_HOME_PATH` 是单独的编译 SDK、其中没有运行库，在实际 Runtime 安装目录查找。
这种方式记录为“CANN 已安装版本”，不宣称它与当前克隆的 ops-nn 提交一致。
准备完成后直接进入第 2 节，不需要执行 B 类步骤。

### B 类：编译当前 ops-nn 源码（测试克隆版本）

参考 ops-nn 的 `docs/zh/develop/aicore_develop_guide.md`。
先准备该文档要求的编译依赖，在 ops-nn 根目录记录源码版本并编译指定算子
（下面针对 Ascend 950）：

```bash
cd /实际路径/ops-nn
git rev-parse HEAD
bash build.sh --pkg --soc=ascend950 --vendor_name=gatherbench --ops=gather_v2 -j8
```

根据构建输出给出的实际文件名安装 `build_out/` 下的 `.run` 包；
配置其 vendor 环境及 op-api 库搜索路径，使运行时能够找到配套内核。
安装会写入所选 CANN 环境，使用自己的隔离环境，按站点约定操作。
官方文档中的 vendor 安装位置为 `${ASCEND_HOME_PATH}/opp/vendors/`，
实际目录名和安装选项以生成包为准。

```bash
bash /实际路径/build_out/实际生成的算子包.run --help
```

查看安装选项后按生成包说明完成安装；本步骤不能只完成编译就跳过安装和 vendor 环境配置。
再指定该次安装产生的动态库：

```bash
export OPS_NN_OPAPI_LIBRARY=/实际安装路径/op_api/lib/libcust_opapi.so
export LD_LIBRARY_PATH="$(dirname "$OPS_NN_OPAPI_LIBRARY"):${LD_LIBRARY_PATH:-}"
nm -D "$OPS_NN_OPAPI_LIBRARY" | rg 'aclnnGatherV2(GetWorkspaceSize)?$'
ldd "$OPS_NN_OPAPI_LIBRARY"
```

保留源码提交与构建日志；运行结果中的 `provenance` 记录所选动态库绝对路径、SHA256、
API 名称和计时范围。若要严格确认采用了源码构建的内核，还需核对 vendor 内核安装/搜索路径
及 profiler 内核信息，不能只凭 `.so` 文件名判断。

### 两类来源共用的加载检查

如果加载时提示找不到 `aclCreateTensor`，可以另外指定当前 CANN 配套的 tensor API 库：

```bash
export OPS_NN_TENSOR_LIBRARY=/实际路径/libnnopbase.so
```

动态库缺失、符号缺失或 ACLNN 调用失败会明确报错，不会静默切换为其他 Gather 实现。
若要分别比较 A、B，建议在不同 shell 中加载各自环境，并使用不同结果目录，
避免 B 的 vendor 配置影响本来想测的 A 类运行。普通使用只需选择其中一类。

## 2. 先验证 5 个规格

切换到 TileOPs 根目录。替换物理卡号，沿用本机 NPU 预约与环境配置：

```bash
python scripts/benchmark_ascend.py \
  --device 3 --unlocked --verify-only \
  --owner gather/opsnn-verify \
  --run-dir "./benchmark-results/gatherv2-verify-$(date +%Y%m%d-%H%M%S)" \
  --op PagedKVCacheGatherFwdOp \
  --case-regex 'ds-v3-mixed|llama-70b-cp|mla-small|llama-cp-unaligned|mla-b256' \
  --implementation-factory benchmarks.ascend.gather_v2_baseline:make_implementations
```

预期匹配 5 个 case，每个 case 对三种实现进行精确结果检查。
确认 `receipt.json` 为 `passed`，`result.passed` 为 5，且 `failed` 为 0。
FP8 case 被命令筛选排除；适配器也不会为 FP8 创建基线。

## 3. 测量性能

使用相同命令去掉 `--verify-only`，并改用新的结果目录：

```bash
python scripts/benchmark_ascend.py \
  --device 3 --unlocked \
  --owner gather/opsnn-perf \
  --run-dir "./benchmark-results/gatherv2-perf-$(date +%Y%m%d-%H%M%S)" \
  --op PagedKVCacheGatherFwdOp \
  --case-regex 'ds-v3-mixed|llama-70b-cp|mla-small|llama-cp-unaligned|mla-b256' \
  --implementation-factory benchmarks.ascend.gather_v2_baseline:make_implementations
```

每个实现采用相同的 3 次预热、15 次 profiler 采样及 1 GiB 缓存驱逐。
`summary.md` 中每个 case 有三行，包含平均耗时、标准差和精度误差。
比较完整功能时使用 `ops-nn-gatherv2-full / tileops` 的平均耗时比；
core 数据只能说明预生成索引后的 GatherV2 时间。
仅在最终 receipt 为 passed 且 verify_only 为 false 时采用性能结果。

## 验证限制

本地 CPU 测试覆盖空请求、非对齐起点、动态页表修改及 full/core 的索引生命周期。
本机没有可用的 CANN/torch_npu，无法验证 ctypes 与目标 CANN 的实际 ABI、
ACLNN 执行或给出性能结果。以上两段设备命令需在目标服务器运行后确认。
