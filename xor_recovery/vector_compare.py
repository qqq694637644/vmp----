"""测试向量对拍。

这个模块负责把恢复出的纯函数参考实现，和未保护 / 受保护二进制在多组输入上做对拍。
它不参与 Triton 恢复，也不参与 trace 回放，只做最终一致性验证。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .reference import TestVector, build_test_vectors, format_u32_hex, recovered_transform
from .verification import parse_program_result


@dataclass(frozen=True)
class VectorComparisonCase:
    """单个测试向量的对拍结果。"""

    vector: TestVector
    reference_result: int
    unprotected_result: int
    protected_result: int

    @property
    def all_match(self) -> bool:
        return (
            self.reference_result == self.unprotected_result
            and self.reference_result == self.protected_result
        )


@dataclass(frozen=True)
class VectorComparisonReport:
    """整批测试向量的对拍结果。"""

    binary_dir: Path
    cases: tuple[VectorComparisonCase, ...]

    @property
    def all_match(self) -> bool:
        return all(case.all_match for case in self.cases)


def run_program_result(binary_path: Path, plaintext_value: int, key_value: int) -> int:
    """运行目标二进制并提取返回值。"""
    if not binary_path.is_file():
        raise FileNotFoundError(f"找不到待验证程序: {binary_path}")

    completed = subprocess.run(
        [str(binary_path), format_u32_hex(plaintext_value), format_u32_hex(key_value)],
        cwd=str(binary_path.parent),
        capture_output=True,
        check=True,
    )
    return parse_program_result(completed.stdout)


def compare_vector_against_binaries(
    binary_dir: Path,
    vector: TestVector,
) -> VectorComparisonCase:
    """对单个向量做三方对拍。"""
    unprotected_binary = binary_dir / "encrypt_demo.exe"
    protected_binary = binary_dir / "encrypt_demo.protected.exe"
    reference_result = recovered_transform(vector.plaintext, vector.key)
    unprotected_result = run_program_result(unprotected_binary, vector.plaintext, vector.key)
    protected_result = run_program_result(protected_binary, vector.plaintext, vector.key)
    return VectorComparisonCase(
        vector=vector,
        reference_result=reference_result,
        unprotected_result=unprotected_result,
        protected_result=protected_result,
    )


def compare_all_vectors(
    binary_dir: Path,
    vectors: tuple[TestVector, ...] | None = None,
) -> VectorComparisonReport:
    """对一组测试向量批量对拍。"""
    if vectors is None:
        vectors = build_test_vectors()

    cases = tuple(compare_vector_against_binaries(binary_dir, vector) for vector in vectors)
    report = VectorComparisonReport(binary_dir=binary_dir, cases=cases)
    if not report.all_match:
        failed_case = next(case for case in report.cases if not case.all_match)
        raise RuntimeError(
            "测试向量对拍失败: "
            f"{failed_case.vector.name} "
            f"plain={format_u32_hex(failed_case.vector.plaintext)} "
            f"key={format_u32_hex(failed_case.vector.key)} "
            f"reference={format_u32_hex(failed_case.reference_result)} "
            f"plain_bin={format_u32_hex(failed_case.unprotected_result)} "
            f"protected={format_u32_hex(failed_case.protected_result)}"
        )

    return report
