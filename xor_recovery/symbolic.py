from __future__ import annotations

from triton import TritonContext

from .models import FormulaResult, RecoveryConfig, TaintAnalysisResult
from .trace_io import parse_trace
from .triton_runtime import initialize_context, replay_trace


def render_formula(ctx: TritonContext, ast_node) -> object:
    # 这里只做 Triton 的内置简化，不再自己写 AST 化简器。
    return ctx.simplify(ast_node, False, False)


def recover_formulas(trace_path, config: RecoveryConfig, taint_report: TaintAnalysisResult) -> tuple[int, int, tuple[FormulaResult, ...]]:
    entry_address, function_size, steps = parse_trace(trace_path)
    ctx = initialize_context(config)
    ast_ctx = ctx.getAstContext()

    sink_specs: dict[int, tuple[int, int]] = {}
    for output_address, expr_id in taint_report.output_roots.items():
        dependency_node = taint_report.dependency_nodes.get(expr_id)
        if dependency_node is None or dependency_node.step_index is None:
            raise RuntimeError(f"无法定位 sink 依赖节点: addr={hex(output_address)} expr_id={expr_id}")
        sink_size = taint_report.output_sizes.get(output_address)
        if sink_size is None:
            raise RuntimeError(f"无法获取 sink 尺寸: addr={hex(output_address)}")
        sink_specs[output_address] = (dependency_node.step_index, sink_size)

    captured_sinks: dict[int, tuple[object, bytes]] = {}

    def observer(step, instruction, context: TritonContext) -> None:
        for output_address, (sink_step, sink_size) in sink_specs.items():
            if step.index != sink_step:
                continue

            for symbolic_expression in instruction.getSymbolicExpressions():
                origin = symbolic_expression.getOrigin()
                if not hasattr(origin, "getAddress"):
                    continue
                if origin.getAddress() != output_address:
                    continue
                if hasattr(origin, "getSize") and origin.getSize() != sink_size:
                    continue
                if output_address not in captured_sinks:
                    concrete_bytes = bytes(context.getConcreteMemoryAreaValue(output_address, sink_size))
                    captured_sinks[output_address] = (symbolic_expression, concrete_bytes)
                return

    replay_trace(ctx, steps, observer)

    formulas: list[FormulaResult] = []
    for output_address, expr_id in sorted(taint_report.output_roots.items()):
        sink_size = taint_report.output_sizes[output_address]
        expected_slice = set(taint_report.output_slices[output_address])
        captured = captured_sinks.get(output_address)
        if captured is None:
            raise RuntimeError(f"第二遍没有捕获到 sink: addr={hex(output_address)} expr_id={expr_id}")

        symbolic_expression, concrete_bytes = captured
        actual_slice = ctx.sliceExpressions(symbolic_expression)
        actual_slice_ids = set(actual_slice.keys())
        if expected_slice != actual_slice_ids:
            raise RuntimeError(
                f"第二遍切片不一致: addr={hex(output_address)} expected={sorted(expected_slice)} actual={sorted(actual_slice_ids)}"
            )

        for offset in range(sink_size):
            byte_low = offset * 8
            byte_high = byte_low + 7
            byte_formula = render_formula(ctx, ast_ctx.extract(byte_high, byte_low, symbolic_expression.getAst()))
            evaluated_value = ctx.evaluateAstViaSolver(byte_formula)
            concrete_value = concrete_bytes[offset]
            if evaluated_value != concrete_value:
                raise RuntimeError(
                    f"公式校验失败: addr={hex(output_address + offset)} 公式值={evaluated_value:#x} 实际值={concrete_value:#x}"
                )

            formulas.append(
                FormulaResult(
                    output_address=output_address + offset,
                    expr_id=symbolic_expression.getId(),
                    slice_size=len(actual_slice_ids),
                    formula_text=str(byte_formula),
                    evaluated_value=evaluated_value,
                    concrete_value=concrete_value,
                )
            )

    return entry_address, function_size, tuple(formulas)
