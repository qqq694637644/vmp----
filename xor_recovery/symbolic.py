from __future__ import annotations

from triton import REG, TritonContext

from .models import FormulaResult, RecoveryConfig, TaintAnalysisResult
from .trace_io import parse_trace
from .triton_runtime import initialize_context, replay_trace


def render_formula(ctx: TritonContext, ast_node) -> object:
    # 这里只做 Triton 的内置简化，不再自己写 AST 化简器。
    return ctx.simplify(ast_node, False, False)


def recover_formulas(trace_path, config: RecoveryConfig, taint_report: TaintAnalysisResult) -> tuple[int, int, tuple[FormulaResult, ...]]:
    trace = parse_trace(trace_path)
    entry_address, function_size, steps = trace
    ctx = initialize_context(config)
    ast_ctx = ctx.getAstContext()
    replay_trace(ctx, steps)

    result_register = ctx.getRegister(REG.X86_64.RAX)
    result_expression = ctx.getSymbolicRegister(result_register)
    if result_expression is None:
        raise RuntimeError("返回寄存器 RAX 没有符号表达式，无法恢复公式")

    result_name = "reg:rax"
    root_expr_id = taint_report.result_roots.get(result_name)
    if root_expr_id is None:
        raise RuntimeError("第一遍没有记录返回根 reg:rax")
    if result_expression.getId() != root_expr_id:
        raise RuntimeError(
            f"第二遍返回根不一致: expected={root_expr_id} actual={result_expression.getId()}"
        )

    expected_slice = set(taint_report.result_slices[result_name])
    actual_slice = ctx.sliceExpressions(result_expression)
    actual_slice_ids = set(actual_slice.keys())
    if expected_slice != actual_slice_ids:
        raise RuntimeError(
            f"第二遍切片不一致: root={result_name} expected={sorted(expected_slice)} actual={sorted(actual_slice_ids)}"
        )

    result_bytes = taint_report.result_bytes
    if len(result_bytes) != taint_report.result_sizes[result_name]:
        raise RuntimeError("返回值字节长度与第一遍记录不一致")

    formulas: list[FormulaResult] = []
    for offset in range(taint_report.result_sizes[result_name]):
        byte_low = offset * 8
        byte_high = byte_low + 7
        byte_formula = render_formula(ctx, ast_ctx.extract(byte_high, byte_low, result_expression.getAst()))
        evaluated_value = ctx.evaluateAstViaSolver(byte_formula)
        concrete_value = result_bytes[offset]
        if evaluated_value != concrete_value:
            raise RuntimeError(
                f"公式校验失败: root={result_name}[{offset}] 公式值={evaluated_value:#x} 实际值={concrete_value:#x}"
            )

        formulas.append(
            FormulaResult(
                result_name=result_name,
                byte_offset=offset,
                expr_id=result_expression.getId(),
                slice_size=len(actual_slice_ids),
                formula_text=str(byte_formula),
                evaluated_value=evaluated_value,
                concrete_value=concrete_value,
            )
        )

    return entry_address, function_size, tuple(formulas)
