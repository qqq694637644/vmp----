from __future__ import annotations

import re

from triton import TritonContext

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


def count_reference_ids(ast_text: str) -> int:
    return len(REF_RE.findall(ast_text))


def is_input_origin_label(label: str) -> bool:
    return label.startswith("plaintext+") or label.startswith("key+")


def run_taint_analysis(trace_path, config: RecoveryConfig) -> tuple[int, int, TaintAnalysisResult]:
    entry_address, function_size, steps = parse_trace(trace_path)
    ctx = initialize_context(config)
    tracked_regions = config.tracked_regions()

    tainted_steps: list[TraceStep] = []
    dependency_nodes: dict[int, DependencyNode] = {}
    dependency_graph: dict[int, tuple[int, ...]] = {}
    expr_step_map: dict[int, int] = {}
    context_hits: set[str] = set()
    sink_candidates: list[tuple[int, int, int, object, bool, int]] = []

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

            origin = symbolic_expression.getOrigin()
            if hasattr(origin, "getAddress") and symbolic_expression.isTainted():
                origin_label = describe_origin(origin, tracked_regions)
                if not is_input_origin_label(origin_label):
                    size = origin.getSize() if hasattr(origin, "getSize") else 1
                    if size == config.output_size:
                        ast_text = str(symbolic_expression.getAst())
                        sink_candidates.append(
                            (
                                step.index,
                                origin.getAddress(),
                                size,
                                symbolic_expression,
                                symbolic_expression.getAst().isSymbolized(),
                                count_reference_ids(ast_text),
                            )
                        )

    replay_trace(ctx, steps, observer)

    output_roots: dict[int, int] = {}
    output_slices: dict[int, tuple[int, ...]] = {}
    output_sizes: dict[int, int] = {}

    selected_candidates: list[tuple[int, int, int, int, object]] = []
    for step_index, address, size, symbolic_expression, ast_symbolized, reference_count in sink_candidates:
        if not ast_symbolized:
            continue
        selected_candidates.append((reference_count, step_index, address, size, symbolic_expression))

    if not selected_candidates:
        raise RuntimeError("未在轨迹中找到包含符号输入的 tainted store sink")

    # 先按引用层数挑最深的候选，再用更晚的写点打破平局。
    # 这里保留最终组合出的结果，而不是更早的中间临时值。
    selected_candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    _, step_index, output_address, size, symbolic_expression = selected_candidates[0]
    slice_expressions = ctx.sliceExpressions(symbolic_expression)
    output_roots[output_address] = symbolic_expression.getId()
    output_slices[output_address] = tuple(sorted(slice_expressions.keys()))
    output_sizes[output_address] = size

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
        output_roots=output_roots,
        output_slices=output_slices,
        output_sizes=output_sizes,
        tainted_memory=tainted_memory,
        tainted_registers=tainted_registers,
        context_hits=tuple(sorted(context_hits)),
    )
    return entry_address, function_size, result
