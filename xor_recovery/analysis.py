from __future__ import annotations

import re

from triton import REG, TritonContext

from .models import DependencyNode, RecoveryConfig, TaintAnalysisResult, TraceStep
from .trace_io import parse_trace
from .triton_runtime import initialize_context, replay_trace


REF_RE = re.compile(r"ref!(\d+)")


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


def extract_reference_ids(ast_text: str) -> tuple[int, ...]:
    return tuple(sorted({int(ref_id) for ref_id in REF_RE.findall(ast_text)}))


def run_taint_analysis(trace_path, config: RecoveryConfig) -> tuple[int, int, TaintAnalysisResult]:
    trace = parse_trace(trace_path)
    entry_address, function_size, steps = trace
    ctx = initialize_context(config)
    tracked_regions = config.tracked_regions()

    tainted_steps: list[TraceStep] = []
    dependency_nodes: dict[int, DependencyNode] = {}
    dependency_graph: dict[int, tuple[int, ...]] = {}
    expr_step_map: dict[int, int] = {}
    context_hits: set[str] = set()

    def observer(step: TraceStep, instruction, _context: TritonContext) -> None:
        if instruction.isTainted():
            tainted_steps.append(step)

        for memory_access, _ in instruction.getLoadAccess():
            label = describe_address(memory_access.getAddress(), tracked_regions)
            if label.startswith("context+"):
                context_hits.add(label)
        for memory_access, _ in instruction.getStoreAccess():
            label = describe_address(memory_access.getAddress(), tracked_regions)
            if label.startswith("context+"):
                context_hits.add(label)

        for symbolic_expression in instruction.getSymbolicExpressions():
            expr_id = symbolic_expression.getId()
            expr_step_map[expr_id] = step.index
            dependency_nodes[expr_id] = DependencyNode(
                expr_id=expr_id,
                step_index=step.index,
                origin=describe_origin(symbolic_expression.getOrigin(), tracked_regions),
                ast=str(symbolic_expression.getAst()),
            )

    replay_trace(ctx, steps, observer)

    result_register = ctx.getRegister(REG.X86_64.RAX)
    result_expression = ctx.getSymbolicRegister(result_register)
    if result_expression is None:
        raise RuntimeError("返回寄存器 RAX 没有符号表达式，无法还原算法")
    if not result_expression.isTainted():
        raise RuntimeError("返回寄存器 RAX 没有受输入污点影响")

    result_name = "reg:rax"
    slice_expressions = ctx.sliceExpressions(result_expression)
    result_roots = {result_name: result_expression.getId()}
    result_slices = {result_name: tuple(sorted(slice_expressions.keys()))}
    result_sizes = {result_name: config.result_size}

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

    result = TaintAnalysisResult(
        tainted_steps=tuple(tainted_steps),
        dependency_nodes=dependency_nodes,
        dependency_graph=dependency_graph,
        result_roots=result_roots,
        result_slices=result_slices,
        result_sizes=result_sizes,
        result_value=trace.result_value if trace.result_value is not None else 0,
        result_bytes=trace.result_bytes if trace.result_bytes is not None else b"",
        tainted_memory=tainted_memory,
        tainted_registers=tainted_registers,
        context_hits=tuple(sorted(context_hits)),
    )
    return entry_address, function_size, result
