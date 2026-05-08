from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import build_config_from_trace, recover


def format_hex(value: int) -> str:
    return f"0x{value:016X}"


def format_preview(values: tuple[str, ...], limit: int = 12) -> str:
    if not values:
        return "无"
    if len(values) <= limit:
        return ", ".join(values)
    preview = ", ".join(values[:limit])
    return f"{preview} ... (+{len(values) - limit})"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 VMP trace 中做两遍分析并还原算法公式。")
    parser.add_argument("trace_file", help="trace_xor.exe 的输出文件")
    parser.add_argument("--context-base", default=None)
    parser.add_argument("--context-size", default=None)
    return parser


def parse_hex(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value, 16)


def configure_utf8_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def main() -> int:
    configure_utf8_console()
    args = build_parser().parse_args()
    trace_path = Path(args.trace_file)
    from .trace_io import parse_trace

    trace_metadata = parse_trace(trace_path)
    config = build_config_from_trace(
        trace_metadata,
        context_base=parse_hex(args.context_base),
        context_size=parse_hex(args.context_size),
    )

    result = recover(trace_path, config)

    print(f"已读取轨迹: {result.trace_path}")
    print(f"函数入口: {format_hex(result.entry_address)}")
    print(f"函数大小: {result.function_size}")
    print("第一遍：动态污点分析")
    print(f"  污点步骤数: {len(result.taint.tainted_steps)}")
    print(f"  关键寄存器: {format_preview(result.taint.tainted_registers)}")
    print(f"  关键内存: {format_preview(result.taint.tainted_memory)}")
    print(f"  关键上下文偏移: {format_preview(result.taint.context_hits)}")
    print("  返回根:")
    for result_name, slice_ids in sorted(result.taint.result_slices.items()):
        root_id = result.taint.result_roots[result_name]
        result_size = result.taint.result_sizes[result_name]
        print(f"    {result_name} size={result_size} -> root={root_id} slice_size={len(slice_ids)}")

    print(f"  依赖节点数: {len(result.taint.dependency_graph)}")
    for expr_id, references in list(sorted(result.taint.dependency_graph.items()))[:10]:
        refs_text = ", ".join(str(ref_id) for ref_id in references) if references else "无"
        node = result.taint.dependency_nodes.get(expr_id)
        origin = node.origin if node is not None else "unknown"
        print(f"    ref!{expr_id}: {origin} -> {refs_text}")
    if len(result.taint.dependency_graph) > 10:
        print(f"    ... 省略 {len(result.taint.dependency_graph) - 10} 项")

    print("第二遍：符号执行")
    for formula in result.formulas:
        print(
            f"  {formula.result_name}[{formula.byte_offset}]: {formula.formula_text} "
            f"=> {formula.evaluated_value:#04x} (slice={formula.slice_size})"
        )

    return 0
