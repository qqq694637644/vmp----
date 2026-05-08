"""第二遍：符号执行和公式恢复。

这一层只处理已经由第一遍筛出来的返回值切片，不再重新做全量污点分析。
目标是把返回值对应的 AST 交给 Triton 自己简化，再逐字节验证它和真实结果一致。
"""

from __future__ import annotations

from pathlib import Path

from triton import REG, TritonContext

from .export import build_recovered_algorithm
from .models import FormulaResult, RecoveryConfig, RecoveredAlgorithm, TaintAnalysisResult
from .trace_io import parse_trace
from .triton_runtime import initialize_context, replay_trace


def render_formula(ctx: TritonContext, ast_node) -> object:
    """把 AST 交给 Triton 的内置简化器处理。

    这里故意不再自己写一层 AST 递归简化，避免重复实现 Triton 已经提供的能力。
    """
    return ctx.simplify(ast_node, True, True)


def recover_formulas(
    trace_path: Path,
    config: RecoveryConfig,
    taint_report: TaintAnalysisResult,
) -> tuple[int, int, RecoveredAlgorithm, tuple[FormulaResult, ...]]:
    """执行第二遍符号恢复。

    前提是第一遍已经找到了返回根和对应切片，因此这里的工作重心是：
    1. 重放同一条 trace。
    2. 确认 RAX 还是同一个返回根。
    3. 把每个返回字节的 AST 简化并和真实结果逐字节对拍。
    """
    trace = parse_trace(trace_path)
    entry_address, function_size, steps = trace
    ctx = initialize_context(config)
    ast_ctx = ctx.getAstContext()
    # 第二遍不再记录污点，只做符号回放和公式提取。
    replay_trace(ctx, steps)

    result_register = ctx.getRegister(REG.X86_64.RAX)
    result_expression = ctx.getSymbolicRegister(result_register)
    if result_expression is None:
        raise RuntimeError("返回寄存器 RAX 没有符号表达式，无法恢复公式")

    result_name = "reg:rax"
    # 第一遍记录的返回根和第二遍实际回放出来的返回根必须一致。
    root_expr_id = taint_report.result_roots.get(result_name)
    if root_expr_id is None:
        raise RuntimeError("第一遍没有记录返回根 reg:rax")
    if result_expression.getId() != root_expr_id:
        raise RuntimeError(
            f"第二遍返回根不一致: expected={root_expr_id} actual={result_expression.getId()}"
        )
    if not taint_report.sink_reached:
        raise RuntimeError(
            f"未跑通到最终汇点: 期望 RAX={taint_report.result_value:#x}，重放得到 RAX={taint_report.replayed_result_value:#x}"
        )

    algorithm = build_recovered_algorithm(ctx, result_name, result_expression.getAst())

    # 第一遍算出的切片和第二遍实际回放出来的切片也必须完全一致。
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
    # 返回值按字节恢复，这样和程序输出、trace 结果、二进制结果都能逐字节对齐。
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

    return entry_address, function_size, algorithm, tuple(formulas)
