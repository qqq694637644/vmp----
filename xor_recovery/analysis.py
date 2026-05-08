"""第一遍：动态污点传播和依赖切片。

这一层不负责恢复公式，只做三件事：
1. 把入口参数当作污点源，观察污点如何流向返回值。
2. 记录关键指令、关键寄存器、关键内存和 VM 上下文命中。
3. 统计缺失的状态，告诉后续回放还缺哪一块快照。
"""

from __future__ import annotations

import re
from pathlib import Path

from triton import CALLBACK, REG, TritonContext

from .models import DependencyNode, MemoryRegion, RecoveryConfig, TaintAnalysisResult, TraceStep
from .trace_io import parse_trace
from .triton_runtime import initialize_context, replay_trace


REF_RE = re.compile(r"ref!(\d+)")
VM_CONTEXT_PREFIX = "vm_context+"


def describe_address(address: int, regions: tuple[MemoryRegion, ...]) -> str:
    """把原始地址翻译成可读区域标签，便于人看日志。"""
    for region in regions:
        if region.contains(address):
            return region.describe(address)
    return f"0x{address:016X}"


def describe_origin(origin, regions: tuple[MemoryRegion, ...]) -> str:
    """把 Triton 表达式的来源统一成字符串，方便打印依赖节点。"""
    if origin is None:
        return "unknown"
    if hasattr(origin, "getAddress"):
        return describe_address(origin.getAddress(), regions)
    if hasattr(origin, "getName"):
        return f"reg:{origin.getName()}"
    return str(origin)


def is_vm_context_label(label: str) -> bool:
    # 只把 VM 上下文区域当作关键上下文偏移，避免把栈地址误报成上下文命中。
    return label.startswith(VM_CONTEXT_PREFIX)


def extract_reference_ids(ast_text: str) -> tuple[int, ...]:
    """从 AST 文本里提取 ref!N 引用，用来构建依赖图。"""
    return tuple(sorted({int(ref_id) for ref_id in REF_RE.findall(ast_text)}))


def build_seeded_register_names() -> set[str]:
    """列出已经被入口快照覆盖的寄存器名。

    这里的集合不是为了“放行”，而是为了把真正缺失、但又会被 tracer 读到的寄存器筛出来。
    """
    # 这里只把我们显式写入的入口状态视为已覆盖，便于精确找出还缺哪些寄存器状态。
    seeded = {
        "rax",
        "rbx",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "rbp",
        "rsp",
        "r8",
        "r9",
        "r10",
        "r11",
        "r12",
        "r13",
        "r14",
        "r15",
        "rip",
        "eflags",
        "cs",
        "ds",
        "es",
        "fs",
        "gs",
        "ss",
        "mxcsr",
        "mxcsr_mask",
    }
    seeded.update(
        {
            "al",
            "ah",
            "ax",
            "eax",
            "bl",
            "bh",
            "bx",
            "ebx",
            "cl",
            "ch",
            "cx",
            "ecx",
            "dl",
            "dh",
            "dx",
            "edx",
            "spl",
            "sp",
            "esp",
            "bpl",
            "bp",
            "ebp",
            "sil",
            "si",
            "esi",
            "dil",
            "di",
            "edi",
            "ip",
            "eip",
        }
    )
    seeded.update({f"r{index}{suffix}" for index in range(8, 16) for suffix in ("", "b", "w", "d")})
    seeded.update({"ac", "af", "cf", "df", "id", "if", "nt", "of", "pf", "sf", "tf", "vm", "vip", "vif", "rf", "zf"})
    seeded.update({f"xmm{index}" for index in range(32)})
    seeded.update({f"ymm{index}" for index in range(32)})
    return seeded


def collapse_memory_ranges(accesses: set[tuple[int, int]]) -> tuple[MemoryRegion, ...]:
    """把零散的缺失内存访问折叠成连续区间，方便 CLI 输出。"""
    if not accesses:
        return ()

    intervals = sorted((address, address + max(size, 1)) for address, size in accesses)
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
            continue
        if end > merged[-1][1]:
            merged[-1][1] = end

    return tuple(
        MemoryRegion("missing_memory", base, end - base)
        for base, end in merged
    )


def run_taint_analysis(trace_path: Path, config: RecoveryConfig) -> tuple[int, int, TaintAnalysisResult]:
    """执行第一遍污点分析。

    这一步会重放整条 trace，但关注点不是“算出结果”，而是“结果到底依赖了哪些输入”。
    """
    trace = parse_trace(trace_path)
    entry_address, function_size, steps = trace
    ctx = initialize_context(config)
    tracked_regions = config.tracked_regions()
    seeded_register_names = build_seeded_register_names()
    watch_memory_addresses = tuple(dict.fromkeys(config.watch_memory_addresses))

    # 下面这些容器只负责记录分析结论，不参与回放本身。
    tainted_steps: list[TraceStep] = []
    dependency_nodes: dict[int, DependencyNode] = {}
    dependency_graph: dict[int, tuple[int, ...]] = {}
    expr_step_map: dict[int, int] = {}
    context_hits: set[str] = set()
    missing_memory_accesses: set[tuple[int, int]] = set()
    missing_registers: set[str] = set()
    watched_memory_writes: list[str] = []
    watched_memory_seen: set[int] = set()

    def memory_probe(probe_ctx: TritonContext, memory_access) -> None:
        # 只要 Triton 真的去读某个内存，而我们又没在入口快照里提供它，就把缺口记下来。
        address = memory_access.getAddress()
        size = memory_access.getSize()
        if probe_ctx.isConcreteMemoryValueDefined(memory_access, size) == False:
            missing_memory_accesses.add((address, size))

    def register_probe(_probe_ctx: TritonContext, register) -> None:
        # 同理，遇到 Triton 读了我们没显式覆盖的寄存器，就把它记成状态缺口。
        register_name = register.getName().lower()
        if register_name not in seeded_register_names:
            missing_registers.add(register_name)

    # 两个 callback 分别统计缺失的内存和寄存器状态，不修改执行结果。
    ctx.addCallback(CALLBACK.GET_CONCRETE_MEMORY_VALUE, memory_probe)
    ctx.addCallback(CALLBACK.GET_CONCRETE_REGISTER_VALUE, register_probe)

    def observer(step: TraceStep, instruction, _context: TritonContext) -> None:
        # observer 不参与求解，只收集分析日志：污点指令、关键内存写、表达式来源。
        if instruction.isTainted():
            tainted_steps.append(step)

        for memory_access, _ in instruction.getLoadAccess():
            label = describe_address(memory_access.getAddress(), tracked_regions)
            if is_vm_context_label(label):
                context_hits.add(label)
        for memory_access, _ in instruction.getStoreAccess():
            label = describe_address(memory_access.getAddress(), tracked_regions)
            if is_vm_context_label(label):
                context_hits.add(label)
            store_address = memory_access.getAddress()
            store_size = max(1, memory_access.getSize())
            for watched_address in watch_memory_addresses:
                if watched_address in watched_memory_seen:
                    continue
                if not (store_address <= watched_address < store_address + store_size):
                    continue
                concrete_bytes = _context.getConcreteMemoryAreaValue(store_address, store_size, False)
                watched_memory_writes.append(
                    f"step={step.index} rip={step.address:#x} watched={describe_address(watched_address, tracked_regions)} "
                    f"store={store_address:#x} size={store_size} bytes={concrete_bytes.hex(' ')}"
                )
                watched_memory_seen.add(watched_address)

        for symbolic_expression in instruction.getSymbolicExpressions():
            # 每个符号表达式都记录下来，后面第二遍会基于这些 ID 追依赖链。
            expr_id = symbolic_expression.getId()
            expr_step_map[expr_id] = step.index
            dependency_nodes[expr_id] = DependencyNode(
                expr_id=expr_id,
                step_index=step.index,
                origin=describe_origin(symbolic_expression.getOrigin(), tracked_regions),
                ast=str(symbolic_expression.getAst()),
            )

    # 第一遍不做逐步寄存器全比对。我们的目标是把真正在意的依赖链和汇点先跑通。
    replay_trace(ctx, steps, observer)

    # 返回寄存器 RAX 是这个样本的最终汇点。
    result_register = ctx.getRegister(REG.X86_64.RAX)
    result_expression = ctx.getSymbolicRegister(result_register)
    if result_expression is None:
        raise RuntimeError("返回寄存器 RAX 没有符号表达式，无法还原算法")

    result_name = "reg:rax"
    # sliceExpressions 会给出从当前返回值倒推出来的最小依赖切片。
    slice_expressions = ctx.sliceExpressions(result_expression)
    result_roots = {result_name: result_expression.getId()}
    result_slices = {result_name: tuple(sorted(slice_expressions.keys()))}
    result_sizes = {result_name: config.result_size}
    replayed_result_value = ctx.getConcreteRegisterValue(result_register)
    expected_result_value = trace.result_value if trace.result_value is not None else 0

    # 把切片里出现过的表达式补进依赖节点表，避免第二遍看不到它们的来源。
    for expr_id, expr in slice_expressions.items():
        if expr_id not in dependency_nodes:
            dependency_nodes[expr_id] = DependencyNode(
                expr_id=expr_id,
                step_index=expr_step_map.get(expr_id),
                origin=describe_origin(expr.getOrigin(), tracked_regions),
                ast=str(expr.getAst()),
            )
        dependency_graph[expr_id] = extract_reference_ids(str(expr.getAst()))

    # 这些结果只用于 CLI 展示和最终校验，不影响回放本身。
    tainted_memory = tuple(
        sorted(
            describe_address(address, tracked_regions)
            for address in ctx.getTaintedMemory()
        )
    )
    tainted_registers = tuple(sorted(register.getName() for register in ctx.getTaintedRegisters()))
    missing_memory_regions = collapse_memory_ranges(missing_memory_accesses)

    result = TaintAnalysisResult(
        tainted_steps=tuple(tainted_steps),
        dependency_nodes=dependency_nodes,
        dependency_graph=dependency_graph,
        result_roots=result_roots,
        result_slices=result_slices,
        result_sizes=result_sizes,
        result_value=expected_result_value,
        replayed_result_value=replayed_result_value,
        result_bytes=trace.result_bytes if trace.result_bytes is not None else b"",
        sink_reached=replayed_result_value == expected_result_value,
        sink_tainted=result_expression.isTainted(),
        missing_memory_regions=missing_memory_regions,
        missing_registers=tuple(sorted(missing_registers)),
        tainted_memory=tainted_memory,
        tainted_registers=tainted_registers,
        context_hits=tuple(sorted(context_hits)),
        watched_memory_writes=tuple(watched_memory_writes),
    )
    return entry_address, function_size, result
