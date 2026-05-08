"""第二遍：符号执行和公式恢复。

这一层只处理已经由第一遍筛出来的返回值切片，不再重新做全量污点分析。
目标是把返回值对应的 AST 交给 Triton 自己简化，再逐字节验证它和真实结果一致。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from triton import REG, TritonContext

from .export import build_recovered_algorithm
from .models import FormulaResult, RecoveryConfig, RecoveredAlgorithm, TaintAnalysisResult
from .trace_io import parse_trace
from .triton_runtime import initialize_context, replay_trace


REF_RE = re.compile(r"ref!(\d+)")


def render_formula(ctx: TritonContext, ast_node) -> object:
    """把 AST 交给 Triton 的内置简化器处理。

    这里故意不再自己写一层 AST 递归简化，避免重复实现 Triton 已经提供的能力。
    """
    return ctx.simplify(ast_node, True, True)


def extract_reference_ids(ast_text: str) -> tuple[int, ...]:
    """从 AST 文本中提取引用编号。

    这一步只用于挑选“更像算法本体”的表达式根，不参与符号推导。
    """
    return tuple(sorted({int(ref_id) for ref_id in REF_RE.findall(ast_text)}))


def select_algorithm_expression(slice_expressions) -> object:
    """从切片里挑出最像算法本体的表达式。

    这里不能直接拿 `reg:rax` 作为导出根，因为那个节点通常只是返回值打包层。
    我们优先选那些：
    - 被最终输出字节直接引用
    - 自身含有旋转、异或、加法、乘法等高层操作
    - 不是纯 `concat / extract` 的打包壳

    这样导出的 AST / LLVM IR / 人类可读算法才会更接近真实算法核心。
    """
    texts: dict[int, str] = {}
    bit_widths: dict[int, int] = {}
    byte_parent_counts: Counter[int] = Counter()

    for expr_id, symbolic_expression in slice_expressions.items():
        ast = symbolic_expression.getAst()
        ast_text = str(ast)
        texts[expr_id] = ast_text
        bit_widths[expr_id] = ast.getBitvectorSize()

        # 只统计输出字节节点对上游表达式的直接引用。
        if bit_widths[expr_id] == 8:
            for ref_id in extract_reference_ids(ast_text):
                byte_parent_counts[ref_id] += 1

    # 纯移位不算算法核心，必须至少出现一次真正的语义操作。
    core_tokens = (
        "bvxor",
        "bvadd",
        "bvmul",
        "bvrol",
        "bvror",
        "bswap",
        "rotate_left",
        "rotate_right",
    )
    scoring_tokens = core_tokens + (
        "bvshl",
        "bvlshr",
        "bvashr",
    )

    ranked_candidates: list[tuple[int, int, object]] = []
    for expr_id, symbolic_expression in slice_expressions.items():
        bit_width = bit_widths[expr_id]
        if bit_width not in (32, 64):
            continue

        ast_text = texts[expr_id].lower()
        if not any(token in ast_text for token in core_tokens):
            continue

        # 纯打包壳会出现很多 extract / concat，但并不代表算法本体。
        if ast_text.count("concat") > 4 and ast_text.count("extract") > 4 and not any(
            token in ast_text for token in ("bvxor", "bvadd", "bvmul", "rotate_left", "rotate_right", "bswap")
        ):
            continue

        op_hits = sum(ast_text.count(token) for token in scoring_tokens)
        penalty = 3 * ast_text.count("concat") + 2 * ast_text.count("extract") + ast_text.count("ref!")
        score = byte_parent_counts[expr_id] * 100 + op_hits * 30 - penalty
        ranked_candidates.append((score, expr_id, symbolic_expression))

    if not ranked_candidates:
        # 没找到明显的核心表达式时，就退回切片根，至少不会静默失败。
        return None

    ranked_candidates.sort(reverse=True)
    return ranked_candidates[0][2]


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

    algorithm_expression = select_algorithm_expression(actual_slice)
    if algorithm_expression is None:
        raise RuntimeError("没有找到可导出的算法核心表达式")

    algorithm = build_recovered_algorithm(ctx, result_name, algorithm_expression.getAst())

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
