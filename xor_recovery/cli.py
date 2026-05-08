from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import run_taint_analysis
from .pipeline import build_config_from_trace
from .snapshot import get_minimal_snapshot_items
from .symbolic import recover_formulas
from .triton_runtime import ReplayStateMismatch


def format_hex(value: int) -> str:
    return f"0x{value:016X}"


def format_preview(values: tuple[str, ...], limit: int = 12) -> str:
    if not values:
        return "无"
    if len(values) <= limit:
        return ", ".join(values)
    preview = ", ".join(values[:limit])
    return f"{preview} ... (+{len(values) - limit})"


def format_step_preview(step) -> str:
    opcode_text = " ".join(f"{byte:02X}" for byte in step.opcode)
    line_text = f" | 行号={step.line_number}" if step.line_number is not None else ""
    return f"#{step.index:06d} RIP={format_hex(step.address)} | 字节={opcode_text}{line_text}"


def format_region(region) -> str:
    end_address = region.base + region.size - 1
    return f"{format_hex(region.base)}-{format_hex(end_address)} size={region.size}"


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
    from .trace_io import parse_trace

    trace_metadata = parse_trace(trace_path)
    watch_memory_addresses = tuple(dict.fromkeys(args.watch_memory))
    config = build_config_from_trace(trace_metadata, watch_memory_addresses=watch_memory_addresses)

    print(f"已读取轨迹: {trace_path}")
    print(f"函数入口: {format_hex(trace_metadata.entry_address)}")
    print(f"函数大小: {trace_metadata.function_size}")
    entry_snapshot_total_bytes = sum(snapshot.size for snapshot in trace_metadata.entry_memory_snapshots)
    print(f"入口全量内存快照: {len(trace_metadata.entry_memory_snapshots)} 页，总字节={entry_snapshot_total_bytes}")
    print("最小快照清单")
    for item in get_minimal_snapshot_items():
        print(f"  - {item}")

    try:
        _entry_address, _function_size, taint_report = run_taint_analysis(trace_path, config)
    except ReplayStateMismatch as exc:
        print(f"第一处状态分歧: {exc}")
        return 1
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

    if not taint_report.sink_reached:
        return 1

    print("第二遍：符号执行")
    _, _, formulas = recover_formulas(trace_path, config, taint_report)
    for formula in formulas:
        print(
            f"  {formula.result_name}[{formula.byte_offset}]: {formula.formula_text} "
            f"=> {formula.evaluated_value:#04x} (slice={formula.slice_size})"
        )

    return 0
