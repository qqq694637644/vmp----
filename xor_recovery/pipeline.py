from __future__ import annotations

from pathlib import Path

from .analysis import run_taint_analysis
from .models import MemoryRegion, RecoveryConfig, RecoveryResult, TraceMetadata
from .symbolic import recover_formulas


def build_config(
    plaintext: bytes,
    key: bytes,
    entry_address: int,
    stack_base: int,
    plaintext_base: int,
    key_base: int,
    output_base: int,
    return_address: int,
    stack_size: int = 0x2000,
    context_base: int | None = None,
    context_size: int | None = None,
) -> RecoveryConfig:
    context_region = None
    if context_base is not None:
        if context_size is None:
            raise ValueError("已指定 context_base，但没有指定 context_size")
        context_region = MemoryRegion("context", context_base, context_size)
    elif context_size is not None:
        raise ValueError("已指定 context_size，但没有指定 context_base")

    return RecoveryConfig(
        plaintext=plaintext,
        key=key,
        entry_address=entry_address,
        stack_base=stack_base,
        plaintext_base=plaintext_base,
        key_base=key_base,
        output_base=output_base,
        return_address=return_address,
        context_region=context_region,
        stack_size=stack_size,
    )


def build_config_from_trace(
    trace: TraceMetadata,
    context_base: int | None = None,
    context_size: int | None = None,
    stack_size: int = 0x2000,
) -> RecoveryConfig:
    if trace.entry_arguments is None:
        raise ValueError("轨迹里没有入口参数，无法构建恢复配置")
    if trace.stack_pointer is None:
        raise ValueError("轨迹里没有栈指针，无法构建恢复配置")
    if trace.return_address is None:
        raise ValueError("轨迹里没有返回地址，无法构建恢复配置")

    arguments = trace.entry_arguments
    if len(arguments.plaintext) != len(arguments.key):
        raise ValueError("轨迹里的 plaintext 和 key 长度不一致")
    if len(arguments.plaintext) != arguments.length:
        raise ValueError("轨迹里的长度与 plaintext/key 长度不一致")

    # 让恢复时的栈布局直接对齐真实 RSP，而不是用固定常量猜测。
    stack_base = trace.stack_pointer - stack_size + 0x20

    return build_config(
        plaintext=arguments.plaintext,
        key=arguments.key,
        entry_address=trace.entry_address,
        stack_base=stack_base,
        plaintext_base=arguments.plaintext_base,
        key_base=arguments.key_base,
        output_base=arguments.output_base,
        return_address=trace.return_address,
        stack_size=stack_size,
        context_base=context_base,
        context_size=context_size,
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
