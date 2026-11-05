# 中文逐行讲解副本：paged_cache_gather.py
# 原文件负责实际注册与执行；本文件只用于阅读，不由 attention/__init__.py 导入。
# 唯一有意的代码差异是末尾注册语句已注释，避免导入副本时重复注册实现。
# 阅读顺序：主机 forward → _compile/main → 快速/逐行加载 → 转换 → 写出。
# 符号：B=batch，NP=pages，PS=page_size，R=width，T=total。
# 对请求序号 b 的第 j 行：p=start[b]+j；slot=table[b,p//PS]*PS+p%PS；
# dst[cu[b]+j,:]=cache[slot,:]；FP8 时先转 FP32 乘 scale，再转输出类型。
# 数据值契约由调用者保证：cu 从 0 非递减到 total，页号合法，起点+长度不超页表容量。
# 本文件的 shape/dtype 检查不等于逐项验证上述设备端数值。
# MTE2=GM→UB 搬运，MTE3=UB→GM 搬运，S=标量流水，V=Vector 流水。
# persistent 指固定数量块循环领取自己的任务，不代表跨多次调用常驻的服务内核。


"""Persistent, page-aware cache gathering on Ascend 950.

Request lengths, page IDs, starts and scale remain device data. Output tiles
provide balanced work even when one request contains most of the tokens.
"""

# Python 主机侧缓存工具；用于复用已编译的内核。
import functools
# 计算输出尾部维度的乘积，例如 8×128=1024。
import math
# 标明 devices 是类级属性，而非单次调用的状态。
from typing import ClassVar

# TileLang 编译入口；这里只导入，不启动内核。
import tilelang
# Ascend 专用 DSL；后续 T.* 描述设备程序，不是普通 Python 张量运算。
import tilelang.ascend.language as T
# 使用 PyTorch dtype、设备和张量元信息。
import torch

# 将实现接入 TileOPs 的后端分发注册表。
from tileops.backend import register_kernel_type
# 复用公共调用描述和接口，因此不需要更改上层 Op 的调用签名。
from tileops.kernels.attention.paged_cache_gather import (
    # CallSpec 保存 batch、页大小、行宽、dtype 等特化信息，不保存设备张量内容。
    PagedKVCacheGatherCall,
    # 约定 forward(dst, cache, table, cu, starts, scale) 的参数次序和行为。
    PagedKVCacheGatherFwdInterface,
)
# 复用设备检查、dtype 字符串转换及内核管理基类。
from tileops.kernels.kernel_base import Kernel


# 主机函数：查询本次逻辑设备的可用 Vector/AIV 核数；不是 Cube 核数。
def _vector_cores(device_index):
    # 延迟导入查询工具，在构建内核时才读取设备信息。
    from tilelang.carver.arch.driver import get_num_vector_cores

    # 将调用对应的设备索引传给查询工具，避免写死物理卡 0。
    count = get_num_vector_cores(device_id=device_index)
    # 设备属性缺失或查询到 0 时拒绝编译，避免创建非法网格。
    if not count:
        # 明确报错，不猜测某个固定型号的核数。
        raise RuntimeError("cannot determine Ascend vector core count")
    # 统一为 Python int，用作编译时网格大小。
    return int(count)


# 主机缓存：参数元组相同就复用编译产物；最多保留 64 项。设备元数据的值不在此缓存键中。
@functools.lru_cache(maxsize=64)
# 内核工厂：这些参数是编译期常量；starts 是“是否传起点”的布尔值，不是起点张量。
def _compile(batch, pages, page_size, table_width, width, dtype, out_dtype, starts, blocks):
    # Whole-page DMA for the large MLA cases; smaller tiles limit the cost
    # of unaligned starts while retaining a full page for 16-row Llama pages.
    # 每任务处理的行数：有起点时为 16；无起点时为一页。小块降低非对齐逐行路径的成本。
    tile_rows = 16 if starts else page_size # att_1
    # 编译期选择 FP8 反量化分支；非 FP8 程序不执行转换。
    quantized = dtype == "float8_e4m3fn"
    # 输出总行数由运行时 dst.shape[0] 绑定；换总行数不必生成新的 Python 工厂缓存项。
    total = T.dynamic("total")

    # 把下面函数解析为设备端 Tensor IR。
    @T.prim_func
    # 设备函数：六个形参都是 GM（全局内存）张量视图。
    def main(
        # 输出展平为 [总行数, 行宽]，写入调用者提供的内存。
        dst: T.Tensor((total, width), out_dtype),
        # 物理页池展平为 [物理行数, 行宽]；物理行号 = 页号×页大小+页内偏移。
        cache: T.Tensor((pages * page_size, width), dtype),
        # 页表 [请求数, 每请求逻辑页数上限]，内容是物理页号。
        table: T.Tensor((batch, table_width), "int32"),
        # 输出前缀边界：请求 b 占 dst[cu[b]:cu[b+1]]；长度为两者之差。
        cu: T.Tensor((batch + 1,), "int32"),
        # 有起点时传 [batch]；无起点时复用 cu 作不被读取的占位参数。
        start: T.Tensor((batch,) if starts else (batch + 1,), "int32"),
        # FP8 接受单个 FP32 乘数；非 FP8 复用 cache 作不被读取的占位参数。
        scale: T.Tensor((1,) if quantized else (pages * page_size, width),
                        # 占位参数的 dtype 也必须与真实传入的张量匹配。
                        "float32" if quantized else dtype),
    ):
        # 启动 blocks 个设备任务，bx 是块编号；任务轮转覆盖输出，不是一个请求绑定一个核。
        with T.Kernel(blocks) as bx:
            # 输入块缓冲区；在 Ascend 这里是片上 UB，不是 CUDA shared memory。
            src = T.alloc_shared((tile_rows, width), dtype)
            # FP8 转换后的输出块缓冲区；非量化分支不使用它，最终是否消除由编译器决定。
            converted = T.alloc_shared((tile_rows, width), out_dtype)
            # Metadata is small; cache it once per core, outside the tile loop.
            # 每个块缓存全部前缀边界，供标量二分查找重复使用。
            offsets = T.alloc_shared((batch + 1,), "int32")
            # 缓存每请求的逻辑读取起点，只在 starts=True 时加载与读取。
            beginnings = T.alloc_shared((batch,), "int32")
            # 缓存 FP8 反量化乘数，避免逐元素从 GM 读 scale。
            scale_ub = T.alloc_shared((1,), "float32")
            # MTE2 将边界从 GM 搬入 UB；块内后续多个任务复用。
            T.copy(cu, offsets)
            # 编译期分支：只有调用提供 seq_starts 才生成相关逻辑。
            if starts:
                # 把起点从 GM 搬入 UB。
                T.copy(start, beginnings)
            # 编译期分支：只为 FP8 cache 执行相关加载、计算或输出操作。
            if quantized:
                # 把一个 FP32 scale 从 GM 搬到 UB。
                T.copy(scale, scale_ub)
            # 发布 MTE2→标量流水的事件 0，标记此前元数据加载完成。
            T.ascend_set_flag("MTE2_S", 0)
            # 标量侧等待元数据就绪，随后才能读取 offsets/beginnings。set/wait 是流水依赖，不是主机同步。
            T.ascend_wait_flag("MTE2_S", 0)
            # 总块数向上取整后均摊到网格；每个 bx 串行处理自己的若干轮任务。
            for task in T.serial(T.ceildiv(T.ceildiv(total, tile_rows), blocks)):
                # 轮转分配：本块第 task 轮负责的全局输出起始行。
                row = (task * blocks + bx) * tile_rows
                # 屏蔽最后一轮多出来的任务；不读取无效输出行对应的元数据。
                if row < total:
                    # 设备端可变标量，保存当前请求编号；不是主机 int。
                    request = T.alloc_var("int32", init=0)
                    # At most 256 requests in the manifest; binary search skips
                    # repeated offsets belonging to empty requests.
                    # 二分区间左端，初始为第 0 个请求。
                    lo = T.alloc_var("int32", init=0)
                    # 二分区间右端，采用半开区间 [lo, hi)。
                    hi = T.alloc_var("int32", init=batch)
                    # 查找第一个满足 cu[b+1] > row 的请求 b。
                    while lo < hi:
                        # 取中间请求编号。
                        mid = (lo + hi) // 2
                        # 该请求已经在当前行之前结束；等号也右移，因此能跳过空请求。
                        if offsets[mid + 1] <= row:
                            # 继续搜索右半边。
                            lo = mid + 1
                        # 进入对应条件不成立的另一条路径；具体含义由上方 if 决定。
                        else:
                            # 继续搜索左半边，保留 mid 作为候选。
                            hi = mid
                    # 二分收敛后得到 row 所属请求。
                    request = lo
                    # 把 packed 输出行转成当前请求内部的相对位置。
                    pos = row - offsets[request]
                    # 编译期分支：只有调用提供 seq_starts 才生成相关逻辑。
                    if starts:
                        # 加起点后得到该请求 cache 中的绝对逻辑 token 位置。
                        pos = pos + beginnings[request]
                    # 快速路径条件：整个块既不跨请求，也不跨逻辑页；合法输入下同页对应连续物理行。
                    if row + tile_rows <= offsets[request + 1] and pos % page_size + tile_rows <= page_size:
                        # 逻辑位置除页大小得到页表列；页号转 int64 后计算物理行，避免大池乘法的 int32 溢出。
                        physical = T.cast(table[request, pos // page_size], "int64") * page_size + pos % page_size
                        # 整块连续矩形 DMA：一次表达多行 GM→UB 加载，减少逐行操作。
                        T.copy(cache[physical:physical + tile_rows, :], src)
                    # 进入对应条件不成立的另一条路径；具体含义由上方 if 决定。
                    else:
                        # Unaligned starts and request/page tails are uncommon;
                        # load each valid row, leaving all full tiles coalesced.
                        # 设备标量循环处理块内每一行；用于加载慢路径或输出尾部。
                        for r in T.serial(tile_rows):
                            # 只处理真实存在的行，避免尾块越界或读取无效缓冲内容。
                            if row + r < total:
                                # 当块跨越请求边界时推进请求编号；while 同时跳过零长度请求。
                                while offsets[request + 1] <= row + r:
                                    # 移到下一个请求；合法 cu 必须以 total 结尾，保证有效行不会越过 batch。
                                    request = request + 1
                                # 重新计算当前这行在所属请求内的相对位置。
                                p = row + r - offsets[request]
                                # 编译期分支：只有调用提供 seq_starts 才生成相关逻辑。
                                if starts:
                                    # 加上该行所属请求自己的逻辑起点。
                                    p = p + beginnings[request]
                                # 逐行查页表并求物理位置，支持非对齐起点以及跨页。
                                physical = T.cast(table[request, p // page_size], "int64") * page_size + p % page_size
                                # 只搬运一整行；行内连续，但相邻行可能来自不同物理页。
                                T.copy(cache[physical, :], src[r, :])
                    # 编译期分支：只为 FP8 cache 执行相关加载、计算或输出操作。
                    if quantized:
                        # 发布输入块加载完成事件。
                        T.ascend_set_flag("MTE2_V", 0)
                        # Vector 流水等待 src/scale 就绪，防止读取未完成的 DMA。
                        T.ascend_wait_flag("MTE2_V", 0)
                        # 在 Vector Function 中用 SIMT 方式表达逐元素计算；256 是逻辑线程配置，不是 256 个物理核。
                        with T.SimtVF(threads=256):
                            # 块内二维元素并行，r 为行、c 为列；具体映射由 Ascend 编译器降低。
                            for r, c in T.Parallel(tile_rows, width):
                                # 只处理真实存在的行，避免尾块越界或读取无效缓冲内容。
                                if row + r < total:
                                    # FP8→FP32 后乘 scale，再由目标缓冲 dtype 转成 FP16/BF16；不改变 src 或 cache。
                                    converted[r, c] = T.cast(src[r, c], "float32") * scale_ub[0]
                        # 发布 Vector 转换完成事件。
                        T.ascend_set_flag("V_MTE3", 0)
                        # MTE3 等待转换结果就绪后才能写出。
                        T.ascend_wait_flag("V_MTE3", 0)
                    # 进入对应条件不成立的另一条路径；具体含义由上方 if 决定。
                    else:
                        # 非 FP8 无需计算，建立输入 DMA→输出 DMA 的依赖。
                        T.ascend_set_flag("MTE2_MTE3", 0)
                        # 等待 src 加载完成，随后可以直接从 UB 写回 GM。
                        T.ascend_wait_flag("MTE2_MTE3", 0)
                    # 完整输出块：可以采用矩形 DMA，不必逐行保护。
                    if row + tile_rows <= total:
                        # 编译期分支：只为 FP8 cache 执行相关加载、计算或输出操作。
                        if quantized:
                            # 量化路径写转换后的完整块，写入 dst 对应连续行。
                            T.copy(converted, dst[row:row + tile_rows, :])
                        # 进入对应条件不成立的另一条路径；具体含义由上方 if 决定。
                        else:
                            # 非量化路径直接写原 dtype 数据，不做额外数值变换。
                            T.copy(src, dst[row:row + tile_rows, :])
                    # 进入对应条件不成立的另一条路径；具体含义由上方 if 决定。
                    else:
                        # 设备标量循环处理块内每一行；用于加载慢路径或输出尾部。
                        for r in T.serial(tile_rows):
                            # 只处理真实存在的行，避免尾块越界或读取无效缓冲内容。
                            if row + r < total:
                                # 编译期分支：只为 FP8 cache 执行相关加载、计算或输出操作。
                                if quantized:
                                    # 输出尾块的有效行：写反量化结果。
                                    T.copy(converted[r, :], dst[row + r, :])
                                # 进入对应条件不成立的另一条路径；具体含义由上方 if 决定。
                                else:
                                    # 输出尾块的有效行：直接写原始值。
                                    T.copy(src[r, :], dst[row + r, :])
                    # A subsequent tile must not overwrite an in-flight DMA.
                    # 发布本轮写出进度，建立输出→下轮输入的缓冲复用依赖。
                    T.ascend_set_flag("MTE3_MTE2", 0)
                    # 等写出完成才能复用 src；当前为串行块调度，尚未做双缓冲重叠。
                    T.ascend_wait_flag("MTE3_MTE2", 0)
                    # 编译期分支：只为 FP8 cache 执行相关加载、计算或输出操作。
                    if quantized:
                        # 量化分支还要保护 converted 缓冲区免于提前覆盖。
                        T.ascend_set_flag("MTE3_V", 0)
                        # 下一轮 Vector 转换前，确保 MTE3 不再读上一轮 converted。
                        T.ascend_wait_flag("MTE3_V", 0)
            # 结束前发布输出 DMA 完成事件。
            T.ascend_set_flag("MTE3_S", 0)
            # 标量侧等待输出结束，避免内核结束时写出仍未完成。
            T.ascend_wait_flag("MTE3_S", 0)

    # 主机调用 Ascend 编译器；out_idx=[] 表示所有缓冲由调用者提供，不自动分配返回张量。
    return tilelang.compile(main, target="ascend", out_idx=[], pass_configs={
        # 禁用自动调度，本内核显式安排流水依赖。
        tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
        # 禁用编译器对 UB/shared 缓冲的自动存储复用，保留各缓冲的独立存储安排。
        tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
        # 禁用该自动线程存储同步插入，由显式事件维护本实现的数据依赖。
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    })


# 主机包装类：实现公共 gather 接口，并继承 TileOPs 内核工具。
class AscendPagedKVCacheGatherKernel(Kernel, PagedKVCacheGatherFwdInterface):
    """Gather supported row geometries without copying metadata to the host."""

    # 分发只允许 NPU；与默认 CUDA 内核并存，不让此实现接收 CUDA 张量。
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    # 类方法，可以在构造/编译内核之前检查适用范围。
    @classmethod
    # 返回拒绝原因字符串；None 表示该调用元信息可以使用本实现。
    def refusal(cls, call: PagedKVCacheGatherCall):
        # 支持的页大小和展平行宽；这是实现范围，不是所有理论 gather 的范围。
        if call.page_size not in (16, 64) or call.row_width not in (576, 1024):
            return "requires page size 16/64 and row width 576/1024"
        # 限制请求数并要求页池/页表非空；这里只看 shape，不读取张量内容。
        if not 0 < call.batch <= 256 or min(call.num_pages, call.table_width) <= 0:
            return "requires 1..256 requests and nonempty page pool/table"
        # 只支持这三种 cache 数据类型。
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
            return "requires FP16, BF16 or FP8 E4M3FN cache"
        # FP8 调用要求反量化为 16 位类型。
        if call.dtype == torch.float8_e4m3fn:
            # 拒绝不在目标范围的输出类型。
            if call.out_dtype not in (torch.float16, torch.bfloat16):
                return "FP8 requires FP16/BF16 destination"
        # 非量化路径要求原样搬运，因此输入输出 dtype 必须相同。
        elif call.out_dtype != call.dtype:
            return "unquantized gather preserves dtype"
        # 继续遵守基类的适用范围约定；通常返回 None。
        return super().refusal(call)

    # 根据一个 CallSpec 构建内核实例。
    def __init__(self, call):
        # 再次检查构造参数，保护直接构造的调用方式。
        reason = self.refusal(call)
        # 存在拒绝原因则阻止继续编译。
        if reason:
            # 向调用者报告具体不支持的配置。
            raise ValueError(reason)
        # 初始化基类的设备索引；有设备则使用其逻辑 index，否则保留默认设备语义。
        super().__init__(device_index=call.device.index if call.device is not None else None)
        # 保存特化元信息，供每次 forward 校验与展平使用。
        self.call = call
        # 基类 dtype_str 依赖这个属性。
        self.dtype = call.dtype
        # 创建或命中编译缓存；保留可调用的设备程序。
        self._compiled = _compile(
            # 固定编译所需几何参数；cache 总页数和页表宽度也包含在工厂缓存键中。
            call.batch, call.num_pages, call.page_size, call.table_width, call.row_width,
            # 将 torch dtype 转成 DSL dtype 字符串，并特化可选起点分支。
            self.dtype_str, self.dtype_to_str(call.out_dtype), call.has_starts,
            # 读取 Vector 核数作为 blocks；这里没有暴露手动核数调优参数。
            _vector_cores(self.device_index),
        )

    # 一次主机调用：校验元信息、创建二维 view，然后启动设备内核；返回 None。
    def forward(self, dst, cache, block_table, cu_seq_lens, seq_starts=None, scale=None):
        # 收集必选张量，统一执行设备和连续性检查。
        tensors = dict(dst=dst, cache=cache, block_table=block_table, cu_seq_lens=cu_seq_lens)
        # Python 侧判断实际可选参数是否存在。
        if seq_starts is not None:
            # 有起点时将它加入检查集合。
            tensors["seq_starts"] = seq_starts
        # Python 侧判断是否传入反量化乘数。
        if scale is not None:
            # 将 scale 加入检查集合。
            tensors["scale"] = scale
        # 基类检查所有张量的设备类型必须是 npu。
        self._require_devices(**tensors)
        # 还要保证具体 NPU 索引一致，避免不同卡的指针混用。
        if any(t.device != dst.device for t in tensors.values()):
            # 校验失败时在主机侧立即报错，不启动设备程序。
            raise ValueError("every gather tensor must be on the same NPU")
        # 局部别名，只简化元信息访问，不复制设备数据。
        c = self.call
        # 可选参数必须匹配已编译分支：FP8 必须有 scale，非 FP8 必须没有。
        if (seq_starts is not None) != c.has_starts or (scale is not None) != (c.dtype == torch.float8_e4m3fn):
            # 校验失败时在主机侧立即报错，不启动设备程序。
            raise ValueError("optional inputs do not match gather specialization")
        # 核对页池几何参数及输出布局，防止复用错误的编译特化。
        if (cache.shape[:2] != (c.num_pages, c.page_size)
                # 输出与 cache 的每行尾部形状必须完全一致。
                or tuple(dst.shape[1:]) != tuple(cache.shape[2:])
                # 每行元素数必须等于编译行宽。
                or math.prod(dst.shape[1:]) != c.row_width
                # 数据类型必须与编译参数匹配。
                or cache.dtype != c.dtype or dst.dtype != c.out_dtype):
            # 校验失败时在主机侧立即报错，不启动设备程序。
            raise ValueError("cache/destination does not match gather specialization")
        # 声明页表的预期 shape 和 dtype。
        metadata = [(block_table, (c.batch, c.table_width), torch.int32),
                    # 声明前缀边界的预期 shape 和 dtype。
                    (cu_seq_lens, (c.batch + 1,), torch.int32)]
        # Python 侧判断实际可选参数是否存在。
        if seq_starts is not None:
            # 起点为每请求一个 int32。
            metadata.append((seq_starts, (c.batch,), torch.int32))
        # Python 侧判断是否传入反量化乘数。
        if scale is not None:
            # 反量化采用全局单个 FP32 scale，不是逐页/逐 head 的 scale。
            metadata.append((scale, (1,), torch.float32))
        # 只检查张量元信息，不执行 item/tolist，因此不为检查内容而同步到主机。
        if any(tuple(t.shape) != shape or t.dtype != dtype for t, shape, dtype in metadata):
            # 校验失败时在主机侧立即报错，不启动设备程序。
            raise ValueError("gather metadata shape/dtype does not match specialization")
        # 底层 kernel 要求连续；非连续输出由公共 Op 分配连续 staging 并复制回。
        if any(not t.is_contiguous() for t in tensors.values()):
            # 校验失败时在主机侧立即报错，不启动设备程序。
            raise ValueError("kernel inputs must be contiguous; use the public Op for views")
        # 没有输出元素时跳过启动，不需要创建零大小网格。
        if not dst.numel():
            # 结果通过 dst 原地写回，不返回新张量；空输出分支同样返回 None。
            return None
        # 只创建二维 view，不复制内存；前面已保证连续。
        flat_cache = cache.view(c.num_pages * c.page_size, c.row_width)
        # 启动设备函数；输出也展平为 [total, width]，底层仍是原 dst 存储。
        self._compiled(dst.view(dst.shape[0], c.row_width), flat_cache, block_table,
                       # 无起点分支把 cu 复用作 Starts 形参占位，不读取它作为起点。
                       cu_seq_lens, seq_starts if c.has_starts else cu_seq_lens,
                       # 非 FP8 分支把 cache 复用作 Scale 占位，避免为缺省参数额外分配张量。
                       scale if scale is not None else flat_cache)
        # 结果通过 dst 原地写回，不返回新张量；空输出分支同样返回 None。
        return None


# 正式文件在这里注册 NPU 候选；讲解副本禁用这一副作用。
# register_kernel_type("PagedKVCacheGatherFwdOp", "ascend", AscendPagedKVCacheGatherKernel)
