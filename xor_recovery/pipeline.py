from __future__ import annotations

from pathlib import Path

from .analysis import run_taint_analysis
from .models import MemoryRegion, RecoveryConfig, RecoveryResult
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
