"""命令行入口。

这个文件只负责把整条恢复链串起来，不承载具体算法：
1. 读取 trace 和入口快照。
2. 第一遍做污点分析，找出返回值依赖。
3. 第二遍做符号执行，恢复公式。
4. 用真实二进制输出做最终一致性校验。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import run_taint_analysis
from .models import FormulaResult, MemoryRegion, RecoveredAlgorithm, TaintAnalysisResult, TraceMetadata, TraceStep
from .pipeline import build_config_from_trace
from .snapshot import get_minimal_snapshot_items
from .symbolic import recover_formulas
from .triton_runtime import ReplayStateMismatch
from .trace_io import parse_trace
from .verification import BinaryConsistencyReport, verify_binary_consistency


def format_hex(value: int) -> str:
    return f"0x{value:016X}"


def format_preview(values: tuple[str, ...], limit: int = 12) -> str:
    if not values:
        return "无"
    if len(values) <= limit:
        return ", ".join(values)
    preview = ", ".join(values[:limit])
    return f"{preview} ... (+{len(values) - limit})"


def format_step_preview(step: TraceStep) -> str:
    """把单条 trace 步骤压成一行，方便 CLI 预览前几条关键指令。"""
    opcode_text = " ".join(f"{byte:02X}" for byte in step.opcode)
    line_text = f" | 行号={step.line_number}" if step.line_number is not None else ""
    return f"#{step.index:06d} RIP={format_hex(step.address)} | 字节={opcode_text}{line_text}"


def format_region(region: MemoryRegion) -> str:
    end_address = region.base + region.size - 1
    return f"{format_hex(region.base)}-{format_hex(end_address)} size={region.size}"


def print_trace_summary(trace_path: Path, trace_metadata: TraceMetadata) -> None:
    """输出 trace 的基础信息和最小快照清单。"""
    print(f"已读取轨迹: {trace_path}")
    print(f"函数入口: {format_hex(trace_metadata.entry_address)}")
    print(f"函数大小: {trace_metadata.function_size}")

    # 入口全量快照是回放的基础输入，先把它的规模打印出来，方便排查是否漏抓页。
    entry_snapshot_total_bytes = sum(snapshot.size for snapshot in trace_metadata.entry_memory_snapshots)
    print(f"入口全量内存快照: {len(trace_metadata.entry_memory_snapshots)} 页，总字节={entry_snapshot_total_bytes}")

    print("最小快照清单")
    for item in get_minimal_snapshot_items():
        print(f"  - {item}")


def print_taint_report(taint_report: TaintAnalysisResult) -> None:
    """输出第一遍污点分析的结果。

    这一步只关心：哪些步骤沾到了输入污点、哪些寄存器/内存被污染、
    以及最终返回值是否真的沿着污点链走到了汇点。
    """
    print("第一遍：动态污点分析")
    print(f"  污点步骤数: {len(taint_report.tainted_steps)}")
    print("  关键指令:")
    for step in taint_report.tainted_steps[:12]:
        print(f"    {format_step_preview(step)}")
    if len(taint_report.tainted_steps) > 12:
        print(f"    ... 省略 {len(taint_report.tainted_steps) - 12} 项")
    print(f"  关键寄存器: {format_preview(taint_report.tainted_registers)}")
    print(f"  关键内存: {format_preview(taint_report.tainted_memory)}")
    print(f"  关键上下文偏移: {format_preview(taint_report.context_hits)}")
    if taint_report.watched_memory_writes:
        print("  监视写入:")
        for write in taint_report.watched_memory_writes[:12]:
            print(f"    {write}")
        if len(taint_report.watched_memory_writes) > 12:
            print(f"    ... 省略 {len(taint_report.watched_memory_writes) - 12} 项")
    print("  补状态缺口:")
    if taint_report.missing_memory_regions:
        for region in taint_report.missing_memory_regions[:12]:
            print(f"    内存: {format_region(region)}")
        if len(taint_report.missing_memory_regions) > 12:
            print(f"    ... 省略 {len(taint_report.missing_memory_regions) - 12} 项")
    else:
        print("    内存: 无")
    if taint_report.missing_registers:
        print(f"    寄存器: {format_preview(taint_report.missing_registers)}")
    else:
        print("    寄存器: 无")
    print(
        f"  最终汇点: 期望 RAX={taint_report.result_value:#010x}，"
        f"重放 RAX={taint_report.replayed_result_value:#010x}，"
        f"污点命中={'是' if taint_report.sink_tainted else '否'}，"
        f"汇点到达={'是' if taint_report.sink_reached else '否'}"
    )
    print("  返回根:")
    for result_name, slice_ids in sorted(taint_report.result_slices.items()):
        root_id = taint_report.result_roots[result_name]
        result_size = taint_report.result_sizes[result_name]
        print(f"    {result_name} size={result_size} -> root={root_id} slice_size={len(slice_ids)}")

    print(f"  依赖节点数: {len(taint_report.dependency_graph)}")
    for expr_id, references in list(sorted(taint_report.dependency_graph.items()))[:10]:
        refs_text = ", ".join(str(ref_id) for ref_id in references) if references else "无"
        node = taint_report.dependency_nodes.get(expr_id)
        origin = node.origin if node is not None else "unknown"
        print(f"    ref!{expr_id}: {origin} -> {refs_text}")
    if len(taint_report.dependency_graph) > 10:
        print(f"    ... 省略 {len(taint_report.dependency_graph) - 10} 项")


def print_algorithm_report(algorithm: RecoveredAlgorithm) -> None:
    """输出整体算法的三种导出结果。

    这里不把巨长的 AST / LLVM IR 全部直接铺到终端，只打印长度和人类可读算法本体。
    完整文本已经放在 `RecoveryResult.algorithm` 里，便于后续保存或二次处理。
    """
    print("整体算法导出")
    print(f"  结果根: {algorithm.result_name}")
    print(f"  简化 AST: 已生成，字符数={len(algorithm.simplified_ast_text)}")
    print(f"  LLVM IR : 已生成，字符数={len(algorithm.llvm_ir)}")
    print("  人类可读算法:")
    for line in algorithm.human_readable_text.splitlines():
        print(f"    {line}")


def print_formula_report(formulas: tuple[FormulaResult, ...]) -> None:
    """输出第二遍符号执行恢复出的公式。"""
    print("第二遍：符号执行")
    for formula in formulas:
        print(
            f"  {formula.result_name}[{formula.byte_offset}]: {formula.formula_text} "
            f"=> {formula.evaluated_value:#04x} (slice={formula.slice_size})"
        )


def print_verification_report(verification: BinaryConsistencyReport) -> None:
    """输出最终一致性校验结果。"""
    print("最终一致性校验")
    print(f"  未保护程序: {verification.unprotected_binary.name} -> {format_hex(verification.unprotected_result)}")
    print(f"  受保护程序: {verification.protected_binary.name} -> {format_hex(verification.protected_result)}")
    print(f"  公式结果  : {format_hex(verification.symbolic_result)}")
    print(f"  轨迹结果  : {format_hex(verification.trace_result)}")
    print("  一致性    : 是")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 VMP trace 中做两遍分析并还原算法公式。")
    parser.add_argument("trace_file", help="trace_xor.exe 的输出文件")
    parser.add_argument(
        "--watch-memory",
        action="append",
        type=lambda value: int(value, 0),
        default=[],
        help="要监视写入的内存地址，可重复传入十六进制或十进制地址",
    )
    return parser


def configure_utf8_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def main() -> int:
    configure_utf8_console()
    args = build_parser().parse_args()
    trace_path = Path(args.trace_file)

    trace_metadata = parse_trace(trace_path)
    watch_memory_addresses = tuple(dict.fromkeys(args.watch_memory))
    config = build_config_from_trace(trace_metadata, watch_memory_addresses=watch_memory_addresses)

    print_trace_summary(trace_path, trace_metadata)

    try:
        _entry_address, _function_size, taint_report = run_taint_analysis(trace_path, config)
    except ReplayStateMismatch as exc:
        print(f"第一处状态分歧: {exc}")
        return 1
    print_taint_report(taint_report)

    if not taint_report.sink_reached:
        return 1

    _, _, algorithm, formulas = recover_formulas(trace_path, config, taint_report)
    print_algorithm_report(algorithm)
    print_formula_report(formulas)

    verification = verify_binary_consistency(trace_path.parent, taint_report.result_value, formulas)
    print_verification_report(verification)

    return 0
