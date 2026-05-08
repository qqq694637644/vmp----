"""恢复流程的编排层。

这个模块只做“把各个阶段串起来”的工作，不包含具体的污点传播或符号推导。
把配置组装、第一遍分析、第二遍恢复分开写，是为了让主流程更容易读，也便于单独测试每一段。
"""

from __future__ import annotations

from pathlib import Path

from .analysis import run_taint_analysis
from .models import MemoryRegion, MemorySnapshot, RecoveryConfig, RecoveryResult, TraceMetadata
from .symbolic import recover_formulas


def build_config(
    plaintext_value: int,
    key_value: int,
    entry_address: int,
    stack_base: int,
    return_address: int,
    operand_size: int = 4,
    entry_registers=None,
    entry_vector_state=None,
    stack_bytes=None,
    stack_size: int = 0x2000,
    vm_context_base: int | None = None,
    vm_context_size: int | None = None,
    vm_context_bytes=None,
    entry_memory_snapshots: tuple[MemorySnapshot, ...] = (),
    extra_memory_snapshots: tuple[MemorySnapshot, ...] = (),
    watch_memory_addresses: tuple[int, ...] = (),
) -> RecoveryConfig:
    """根据显式参数组装恢复配置。

    这个入口主要给手工调试和单元测试使用：调用方已经知道入口地址、
    栈布局、VM 上下文和可选的额外快照，因此这里不做任何推断，只负责把
    数据整理成 Triton 回放所需的结构。
    """
    vm_context_region = None
    if vm_context_base is not None:
        if vm_context_size is None:
            raise ValueError("已指定 vm_context_base，但没有指定 vm_context_size")
        vm_context_region = MemoryRegion("vm_context", vm_context_base, vm_context_size)
    elif vm_context_size is not None:
        raise ValueError("已指定 vm_context_size，但没有指定 vm_context_base")

    return RecoveryConfig(
        plaintext_value=plaintext_value,
        key_value=key_value,
        entry_address=entry_address,
        stack_base=stack_base,
        return_address=return_address,
        operand_size=operand_size,
        entry_registers=entry_registers,
        entry_vector_state=entry_vector_state,
        stack_bytes=stack_bytes,
        vm_context_region=vm_context_region,
        vm_context_bytes=vm_context_bytes,
        entry_memory_snapshots=entry_memory_snapshots,
        extra_memory_snapshots=extra_memory_snapshots,
        watch_memory_addresses=watch_memory_addresses,
        stack_size=stack_size,
    )


def build_config_from_trace(
    trace: TraceMetadata,
    stack_size: int = 0x2000,
    watch_memory_addresses: tuple[int, ...] = (),
) -> RecoveryConfig:
    """从 trace 元数据构建恢复配置。

    这一步的核心是把 tracer 已经采到的真实状态转成 Triton 可以直接回放的
    入口快照。这里一旦缺字段就直接报错，因为缺状态会让后续符号执行产生假结果。
    """
    if trace.entry_arguments is None:
        raise ValueError("轨迹里没有入口参数，无法构建恢复配置")
    if trace.stack_pointer is None:
        raise ValueError("轨迹里没有栈指针，无法构建恢复配置")
    if trace.return_address is None:
        raise ValueError("轨迹里没有返回地址，无法构建恢复配置")
    if trace.vm_context_base is None:
        raise ValueError("轨迹里没有 VM 上下文基址，无法构建恢复配置")
    if trace.vm_context_bytes is None:
        raise ValueError("轨迹里没有上下文快照，无法构建恢复配置")

    arguments = trace.entry_arguments

    # 让恢复时的栈布局直接对齐真实 RSP，而不是用固定常量猜测。
    # 这里保留 0x20 的 home space，是为了和 Windows x64 调用约定的栈形状对齐。
    stack_base = trace.stack_pointer - stack_size + 0x20

    return build_config(
        plaintext_value=arguments.plaintext_value,
        key_value=arguments.key_value,
        entry_address=trace.entry_address,
        stack_base=stack_base,
        return_address=trace.return_address,
        operand_size=len(arguments.plaintext),
        entry_registers=trace.entry_registers,
        entry_vector_state=trace.entry_vector_state,
        stack_bytes=trace.stack_bytes,
        stack_size=stack_size,
        vm_context_base=trace.vm_context_base,
        vm_context_size=len(trace.vm_context_bytes),
        vm_context_bytes=trace.vm_context_bytes,
        entry_memory_snapshots=trace.entry_memory_snapshots,
        extra_memory_snapshots=trace.extra_memory_snapshots,
        watch_memory_addresses=watch_memory_addresses,
    )


def recover(trace_path: Path, config: RecoveryConfig) -> RecoveryResult:
    """两遍恢复主流程。

    第一遍只做动态污点分析，目标是找出返回值相关的依赖切片。
    第二遍只针对这个切片做符号执行和公式恢复，避免把整条 VM 轨迹都当成同等重要。
    """
    entry_address, function_size, taint_report = run_taint_analysis(trace_path, config)
    _, _, formulas = recover_formulas(trace_path, config, taint_report)
    return RecoveryResult(
        trace_path=trace_path,
        entry_address=entry_address,
        function_size=function_size,
        taint=taint_report,
        formulas=formulas,
    )
