"""最终一致性校验。

这里不做符号推导，只负责把三种结果放到同一条验收线上：
1. trace 里记录的真实返回值
2. Triton 符号执行恢复出的公式值
3. 未保护 / 受保护二进制自己跑出来的返回值

只要有一项不一致，就说明恢复链没有真正闭环。
"""

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
    """解析样本程序打印出来的 result 行。

    程序输出的十六进制字符串是按字节打印的，因此这里要先把字节串解析出来，
    再用 little-endian 还原成整数，避免把打印顺序误认为数值顺序。
    """
    text = stdout.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        match = PROGRAM_RESULT_RE.match(line.strip())
        if match is not None:
            result_bytes = bytes.fromhex(match.group(1))
            return int.from_bytes(result_bytes, byteorder="little")
    raise ValueError("程序输出里没有找到 result 行")


def run_program_result(binary_path: Path) -> int:
    """运行目标二进制并提取它打印的结果值。"""
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
    """把按字节恢复出来的公式结果重新拼成完整整数。

    这里必须按 byte_offset 排序，否则字节顺序一乱，最终值就会被拼错。
    """
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
    """最终验收入口。

    先把公式结果拼成整数，再分别运行未保护和受保护二进制；
    三者任何一个对不上，都直接抛异常暴露问题。
    """
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
