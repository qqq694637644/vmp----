from __future__ import annotations

import re

from triton import CALLBACK, REG, TritonContext

from .models import DependencyNode, MemoryRegion, RecoveryConfig, TaintAnalysisResult, TraceStep
from .trace_io import parse_trace
from .triton_runtime import initialize_context, replay_trace


REF_RE = re.compile(r"ref!(\d+)")
VM_CONTEXT_PREFIX = "vm_context+"


def describe_address(address: int, regions) -> str:
    for region in regions:
        if region.contains(address):
            return region.describe(address)
    return f"0x{address:016X}"


def describe_origin(origin, regions) -> str:
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
    return tuple(sorted({int(ref_id) for ref_id in REF_RE.findall(ast_text)}))


def build_seeded_register_names(config: RecoveryConfig) -> set[str]:
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


def run_taint_analysis(trace_path, config: RecoveryConfig) -> tuple[int, int, TaintAnalysisResult]:
    trace = parse_trace(trace_path)
    entry_address, function_size, steps = trace
    ctx = initialize_context(config)
    tracked_regions = config.tracked_regions()
    seeded_register_names = build_seeded_register_names(config)
    watch_memory_addresses = tuple(dict.fromkeys(config.watch_memory_addresses))

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
        address = memory_access.getAddress()
        size = memory_access.getSize()
        if probe_ctx.isConcreteMemoryValueDefined(memory_access, size) == False:
            missing_memory_accesses.add((address, size))

    def register_probe(_probe_ctx: TritonContext, register) -> None:
        register_name = register.getName().lower()
        if register_name not in seeded_register_names:
            missing_registers.add(register_name)

    ctx.addCallback(CALLBACK.GET_CONCRETE_MEMORY_VALUE, memory_probe)
    ctx.addCallback(CALLBACK.GET_CONCRETE_REGISTER_VALUE, register_probe)

    def observer(step: TraceStep, instruction, _context: TritonContext) -> None:
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
            expr_id = symbolic_expression.getId()
            expr_step_map[expr_id] = step.index
            dependency_nodes[expr_id] = DependencyNode(
                expr_id=expr_id,
                step_index=step.index,
                origin=describe_origin(symbolic_expression.getOrigin(), tracked_regions),
                ast=str(symbolic_expression.getAst()),
            )

    # 主流程只依赖控制流重放和最终汇点校验；逐步寄存器全比对保留给单独调试场景。
    replay_trace(ctx, steps, observer)

    result_register = ctx.getRegister(REG.X86_64.RAX)
    result_expression = ctx.getSymbolicRegister(result_register)
    if result_expression is None:
        raise RuntimeError("返回寄存器 RAX 没有符号表达式，无法还原算法")

    result_name = "reg:rax"
    slice_expressions = ctx.sliceExpressions(result_expression)
    result_roots = {result_name: result_expression.getId()}
    result_slices = {result_name: tuple(sorted(slice_expressions.keys()))}
    result_sizes = {result_name: config.result_size}
    replayed_result_value = ctx.getConcreteRegisterValue(result_register)
    expected_result_value = trace.result_value if trace.result_value is not None else 0

    for expr_id, expr in slice_expressions.items():
        if expr_id not in dependency_nodes:
            dependency_nodes[expr_id] = DependencyNode(
                expr_id=expr_id,
                step_index=expr_step_map.get(expr_id),
                origin=describe_origin(expr.getOrigin(), tracked_regions),
                ast=str(expr.getAst()),
            )
        dependency_graph[expr_id] = extract_reference_ids(str(expr.getAst()))

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
