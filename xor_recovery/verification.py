from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import FormulaResult


PROGRAM_RESULT_RE = re.compile(r"^result\s*:\s*0x([0-9A-Fa-f]+)$")


@dataclass(frozen=True)
class BinaryConsistencyReport:
    binary_dir: Path
    unprotected_binary: Path
    protected_binary: Path
    trace_result: int
    symbolic_result: int
    unprotected_result: int
    protected_result: int

    @property
    def all_match(self) -> bool:
        return (
            self.trace_result == self.symbolic_result
            and self.trace_result == self.unprotected_result
            and self.trace_result == self.protected_result
        )


def parse_program_result(stdout: bytes) -> int:
    text = stdout.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        match = PROGRAM_RESULT_RE.match(line.strip())
        if match is not None:
            result_bytes = bytes.fromhex(match.group(1))
            return int.from_bytes(result_bytes, byteorder="little")
    raise ValueError("程序输出里没有找到 result 行")


def run_program_result(binary_path: Path) -> int:
    if not binary_path.is_file():
        raise FileNotFoundError(f"找不到待验证程序: {binary_path}")

    completed = subprocess.run(
        [str(binary_path)],
        cwd=str(binary_path.parent),
        capture_output=True,
        check=True,
    )
    return parse_program_result(completed.stdout)


def assemble_symbolic_result(formulas: tuple[FormulaResult, ...]) -> int:
    if not formulas:
        raise ValueError("没有可汇总的符号公式")

    ordered_formulas = sorted(formulas, key=lambda item: item.byte_offset)
    result_value = 0
    expected_offset = 0
    for formula in ordered_formulas:
        if formula.byte_offset != expected_offset:
            raise ValueError("公式字节偏移不连续，无法汇总最终返回值")
        result_value |= (formula.evaluated_value & 0xFF) << (formula.byte_offset * 8)
        expected_offset += 1

    return result_value


def verify_binary_consistency(
    binary_dir: Path,
    trace_result: int,
    formulas: tuple[FormulaResult, ...],
) -> BinaryConsistencyReport:
    unprotected_binary = binary_dir / "encrypt_demo.exe"
    protected_binary = binary_dir / "encrypt_demo.protected.exe"

    symbolic_result = assemble_symbolic_result(formulas)
    unprotected_result = run_program_result(unprotected_binary)
    protected_result = run_program_result(protected_binary)

    report = BinaryConsistencyReport(
        binary_dir=binary_dir,
        unprotected_binary=unprotected_binary,
        protected_binary=protected_binary,
        trace_result=trace_result,
        symbolic_result=symbolic_result,
        unprotected_result=unprotected_result,
        protected_result=protected_result,
    )
    if not report.all_match:
        raise RuntimeError(
            "最终校验失败: "
            f"trace={trace_result:#010x} "
            f"symbolic={symbolic_result:#010x} "
            f"plain={unprotected_result:#010x} "
            f"protected={protected_result:#010x}"
        )

    return report
