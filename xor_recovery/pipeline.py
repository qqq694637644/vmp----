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
    extra_memory_snapshots: tuple[MemorySnapshot, ...] = (),
) -> RecoveryConfig:
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
        extra_memory_snapshots=extra_memory_snapshots,
        stack_size=stack_size,
    )


def build_config_from_trace(
    trace: TraceMetadata,
    stack_size: int = 0x2000,
) -> RecoveryConfig:
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
        extra_memory_snapshots=trace.extra_memory_snapshots,
    )


def recover(trace_path: Path, config: RecoveryConfig) -> RecoveryResult:
    entry_address, function_size, taint_report = run_taint_analysis(trace_path, config)
    _, _, formulas = recover_formulas(trace_path, config, taint_report)
    return RecoveryResult(
        trace_path=trace_path,
        entry_address=entry_address,
        function_size=function_size,
        taint=taint_report,
        formulas=formulas,
    )
