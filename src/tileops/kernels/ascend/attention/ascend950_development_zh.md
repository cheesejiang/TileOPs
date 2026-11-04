# Ascend 950 开发说明（中文翻译）

原文：[docs/ascend950/development.md](../../../../../docs/ascend950/development.md)。
本文翻译当前仓库原文，其中算子数量、部署路径和 CI 状态均为原文描述。
工具链的具体版本以 [toolchain.md](../../../../../docs/ascend950/toolchain.md)
及其机器可读锁文件为准。

## 实现接入

`ascend950` 分支保留上游 GPU 实现，通过 TileOPs 的 `tileops.backends`
入口和 `register_kernel_type` 接口添加 Ascend 内核。
公共算子族导入、算子签名、workload 和数值契约共用。
例如，`from tileops.gemm import BmmFwdOp` 对 NPU 张量使用 Ascend 实现，
对 CUDA 张量使用已有 CUDA 实现。

原文记录后端实现了逐元素、Bmm 和 Cumsum 等类别的 41 个算子。
注册时不会编译内核，也不会查询设备属性。
未安装 `torch_npu` 时，Ascend 入口不执行注册。
安装了 NPU 运行环境时，后端导入错误可通过 `tileops.load_failures()` 查看。

## 运行环境

使用标准 [Ascend 950 工具链契约](../../../../../docs/ascend950/toolchain.md)
及 `config/ascend950-toolchain.json` 锁文件。
它区分 CANN 正式版与 beta 构建，除包版本外，还固定本地二进制和头文件指纹。
仅安装笼统的“CANN 9.2”或“TileLang 0.1.15”不能证明环境匹配。
不要在隔离的 NPU 环境中安装 GPU 依赖集合。

核心和开发依赖在所有目标上使用 TileLang 0.1.15。
上游 GPU benchmark 依赖集合有单独限制：vLLM 0.27.1 固定 TileLang 0.1.12。
因此完整 GPU `[bench]` 扩展及镜像需要先迁移基线依赖，才能用于本分支。
保持未变的 `constraints-runner-lock.txt` 记录的是上游镜像，不是已验证的 950 环境。
不要通过 `--no-deps` 绕过这些依赖，再声称 GPU benchmark 环境兼容。

## 共用性能测试入口

新增的多卡开发模式使用 `--device N --unlocked`：按物理卡单独预约，
不锁频、不要求标准 32-Cube 型号或精确 TileLang 版本，记录实际硬件信息。
精度检查和 profiler 采样仍保留。结果标记为非正式协议，不能与下文的正式结果混用。
具体命令见 [gather 中文说明](paged_cache_gather_zh.md#性能验证与调优)。
下面保留原文的正式测量要求。

**所有开发者和基线适配器使用同一源码版本中的以下入口：**

```bash
python scripts/benchmark_ascend.py \
  --owner developer-name/experiment-name \
  --run-dir /shared/results/unique-run-name \
  --clock-config /site/ascend950-clocks.json \
  --op BmmFwdOp
```

启动器在导入运行时之前申请主机级设备预约，并保护整个前台执行过程。
所有用户必须共用 `/run/lock/ascend-npu/device-0.lock`。
保护程序在启动前检查驱动进程、设备句柄、健康状态和持续空闲状态，
运行期间检查外部驱动活动。这是协作锁，其他不遵循协议的程序仍可能忽略它。
不要为了启动任务而删除正在使用的锁文件或重置设备。
保护程序退出码为 75、76 或 78 时，本次结果无效。

物理设备映射不同的站点可以通过 `--guard /absolute/path/to/npu_guard.py`
或环境变量 `TILEOPS_NPU_GUARD` 指定经过站点审核的 Python 保护入口。
该入口必须实现相同的主机锁和进程归属契约；此选项不会关闭预约验证。
容器需要能访问真实主机锁并看到主机 PID；容器私有锁不足以满足要求。

固定正式测量协议为 `ascend950-profiler-locked-v1`：

| 项目 | 要求 |
| --- | --- |
| 正确性 | 使用 workload 参考实现和既有数值策略，检查每个输出，保持共享输入不变 |
| 预热 | 正确性和 JIT 设置完成后调用 3 次 |
| 重复采样 | 15 个独立 profiler 会话，每个会话调用算子一次 |
| 缓存 | 重置、执行 1 GiB int32 缓存驱逐内核、同步，然后采样 |
| 时钟 | 执行站点验证过的锁频命令，每次采样前后独立读取两个组件的频率，退出时恢复 |
| 计时器 | `torch_npu.profiler.profile`，使用原始 CANN `ffts_profile` 内核耗时 |
| 评分 | 每次调用的设备执行区间并集长度取算术平均值；排除主机间隙，重叠内核只计一次；耗时总和保留作诊断 |
| 证据 | 全部 15 个样本和原始轨迹、环境信息、时钟读回、代码和契约哈希、最终保护回执 |
| 无效运行 | 正确性、profiler、时钟、外部干扰或源码变化检查失败时，不予验收 |
| Event 计时 | Ascend 计时器不回退到 event 或主机墙钟计时 |

结果目录必须是新目录。不允许静默续跑、删减样本，或让各开发者自行覆盖预热与重复次数。
只有当 `receipt.json` 的状态为 `passed`，且其中 `result.verify_only` 为 `false` 时，
才能采用结果。如果先写出某行结果，随后设备保护检查失败，该行结果同样无效。
原始 profiler 目录可能较大，应与结果一起归档，不要在审查前删除。
不同协议 ID、TileLang 版本或锁频策略的结果不能混用，包括早期开发测量结果。

锁频配置包含绝对路径命令参数数组 `lock`、`restore`，以及
`component_mhz: [component0_mhz, component1_mhz]`。
站点必须使用受支持且已验证的频率控制工具；即使锁频部分失败，
`restore` 也必须能够恢复之前的策略。
benchmark 检查 DSMI 读回值，并拒绝超过额定频率的配置。
**没有经过验证的锁频命令，就不能进行正式测量。**
采样时的频率读回不能证明两个采样点之间频率从未变化。

只验证正确性时，添加 `--verify-only` 并省略 `--clock-config`。
这种运行不产生延迟数据，也不构成性能验收。正确性 CI 不需要基线。

要接入其他实现，传入 `--implementation-factory package.module:function`。
该函数接收 `(op_name, case)`，返回名称到已有 `benchmarks.api.Implementation`
对象的映射。各实现使用与 TileOPs 相同的参考结果、输入、重置顺序和计时器。
有状态实现使用私有参数和 `reset`。
共用可调用接口是 `benchmarks.ascend.runner.benchmark_case`。
计时器的短搜索模式使用独立协议 ID，不能产生正式结果。

## CI

本地与 GitHub 使用相同命令：

```bash
python scripts/ci/ascend950.py --phase cpu --output /results/unique-cpu-run
python scripts/ci/ascend950.py --phase device --output /results/unique-device-run
```

CPU 检查验证 manifest、benchmark 契约、后端分发和打包覆盖。
设备检查验证所有已注册 Ascend 算子的每个 manifest case。
没有匹配 case、通过测试数为零、导入错误或源码变化都会导致运行失败。
在基线采用完全相同的测量协议和工具链之前，性能验收仍未完成。
正确性 CI 通过不代表达到速度目标。

GitHub 的 `Ascend 950` 工作流针对 `ascend950` 分支的推送和 PR 运行。
NPU 执行在预检查后由可信推送或手动触发。
runner 需要 `ascend-950` 标签，以及指向已准备运行环境的 `ASCEND_ENV_SCRIPT`。
测试入口使用实际检出的源码，而非另一份 editable checkout。
已有 GPU 工作流和面向 910 的 `main` 分支保持各自的硬件范围。

公开网站使用 `yyttt6/TileOPs.github.io` 中独立的
`Official Ascend 950 nightly` 工作流，在远端生产 runner 上于每日 UTC 00:00 执行。
本分支的本地正确性 CI 不发布网站性能数据。
远端工作流在检查前固定源码提交，并保留共用协议、原始样本、源码哈希和时钟读回。
没有受支持的主机时钟控制或设备预约时，远端设备阶段仍无法运行。

多 Linux 用户环境下，仓库、Python 环境和结果根目录应位于各用户都可访问的位置。
私有 `/root` 目录下的 checkout 不能作为共享安装。
每位开发者应使用独立 worktree 和结果目录，同时使用相同 benchmark 版本和锁频策略。
