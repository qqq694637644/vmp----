"""基于 Triton 结果的多向量对拍。

这个模块不包含手写算法真值。它做的事情是：
1. 用 `trace_xor.exe` 对目标二进制抓取 trace。
2. 把 trace 交给现有 Triton 恢复管线。
3. 用恢复出来的结果和未保护 / 受保护二进制对拍。

因此这里的“参考值”来源于 Triton 恢复结果本身，而不是人工实现的同逻辑公式。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from xor_recovery.models import RecoveryResult

from .pipeline import build_config_from_trace, recover
from .trace_io import parse_trace
from .verification import BinaryConsistencyReport, verify_binary_consistency
from .vectors import TestVector, build_test_vectors, format_u32_hex


@dataclass(frozen=True)
class TritonVectorCase:
    """单个向量的 Triton 对拍结果。"""

    vector: TestVector
    trace_path: Path
    verification: BinaryConsistencyReport

    @property
    def all_match(self) -> bool:
        return self.verification.all_match


@dataclass(frozen=True)
class TritonVectorReport:
    """整批向量的 Triton 对拍结果。"""

    binary_dir: Path
    cases: tuple[TritonVectorCase, ...]

    @property
    def all_match(self) -> bool:
        return all(case.all_match for case in self.cases)


def get_tracer_path(binary_dir: Path) -> Path:
    tracer_path = binary_dir / "trace_xor.exe"
    if not tracer_path.is_file():
        raise FileNotFoundError(f"找不到 tracer: {tracer_path}")
    return tracer_path


def run_trace_capture(
    tracer_path: Path,
    symbol_path: Path,
    target_path: Path,
    plaintext_value: int,
    key_value: int,
    log_path: Path,
) -> None:
    """调用 tracer 抓取一条新的 trace。"""
    command = [
        str(tracer_path),
        "--symbols",
        str(symbol_path),
        str(target_path),
        format_u32_hex(plaintext_value),
        format_u32_hex(key_value),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_file:
        subprocess.run(
            command,
            cwd=str(target_path.parent),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def compare_vector_against_binaries(
    binary_dir: Path,
    vector: TestVector,
    tracer_path: Path | None = None,
    trace_target: str = "protected",
) -> TritonVectorCase:
    """对单个向量先抓 trace，再做 Triton 恢复与二进制对拍。"""
    binary_dir = binary_dir.resolve()
    tracer = tracer_path or get_tracer_path(binary_dir)
    unprotected_binary = binary_dir / "encrypt_demo.exe"
    protected_binary = binary_dir / "encrypt_demo.protected.exe"
    symbol_path = unprotected_binary

    if trace_target == "protected":
        target_binary = protected_binary
    elif trace_target == "unprotected":
        target_binary = unprotected_binary
    else:
        raise ValueError(f"未知 trace_target: {trace_target}")

    trace_dir = binary_dir / "vector_traces"
    trace_path = trace_dir / f"{vector.name}.{trace_target}.log"
    run_trace_capture(
        tracer,
        symbol_path,
        target_binary,
        vector.plaintext,
        vector.key,
        trace_path,
    )

    trace_metadata = parse_trace(trace_path)
    if trace_metadata.entry_arguments is None:
        raise RuntimeError(f"trace 缺少入口参数: {trace_path}")

    config = build_config_from_trace(trace_metadata)
    recovery: RecoveryResult = recover(trace_path, config)
    verification = verify_binary_consistency(
        binary_dir,
        vector.plaintext,
        vector.key,
        recovery.taint.result_value,
        recovery.formulas,
    )

    return TritonVectorCase(
        vector=vector,
        trace_path=trace_path,
        verification=verification,
    )


def compare_all_vectors(
    binary_dir: Path,
    vectors: tuple[TestVector, ...] | None = None,
    tracer_path: Path | None = None,
    trace_target: str = "protected",
) -> TritonVectorReport:
    """对一组测试向量批量对拍。"""
    binary_dir = binary_dir.resolve()
    if vectors is None:
        vectors = build_test_vectors()

    cases = tuple(
        compare_vector_against_binaries(
            binary_dir,
            vector,
            tracer_path=tracer_path,
            trace_target=trace_target,
        )
        for vector in vectors
    )
    report = TritonVectorReport(binary_dir=binary_dir, cases=cases)
    if not report.all_match:
        failed_case = next(case for case in report.cases if not case.all_match)
        raise RuntimeError(
            "测试向量对拍失败: "
            f"{failed_case.vector.name} "
            f"trace={failed_case.trace_path} "
            f"plain={format_u32_hex(failed_case.vector.plaintext)} "
            f"key={format_u32_hex(failed_case.vector.key)} "
            f"trace={format_u32_hex(failed_case.verification.trace_result)} "
            f"symbolic={format_u32_hex(failed_case.verification.symbolic_result)} "
            f"plain_bin={format_u32_hex(failed_case.verification.unprotected_result)} "
            f"protected={format_u32_hex(failed_case.verification.protected_result)}"
        )

    return report
