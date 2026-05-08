"""测试向量集合。

这里只负责生成稳定的输入数据，不包含任何算法真值。
算法真值必须来自 Triton 恢复流程，而不是这里手写的公式。
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random


MASK32 = 0xFFFFFFFF


@dataclass(frozen=True)
class TestVector:
    """一组用于对拍的 32 位输入。"""

    name: str
    plaintext: int
    key: int


def format_u32_hex(value: int) -> str:
    """把 32 位整数格式化成命令行可直接读取的十六进制参数。"""
    return f"0x{value & MASK32:08X}"


def build_test_vectors() -> tuple[TestVector, ...]:
    """生成一组稳定的测试向量。

    向量分三类：
    - 边界值，覆盖旋转、加法、乘法、字节重排的极端情况。
    - 固定样本，覆盖当前 trace 的已知输入。
    - 伪随机样本，用固定种子保证每次都可重复。
    """
    vectors: list[TestVector] = [
        TestVector("zero_zero", 0x00000000, 0x00000000),
        TestVector("one_zero", 0x00000001, 0x00000000),
        TestVector("zero_one", 0x00000000, 0x00000001),
        TestVector("all_ones", 0xFFFFFFFF, 0xFFFFFFFF),
        TestVector("high_bit", 0x80000000, 0x80000000),
        TestVector("alternating_a", 0xAAAAAAAA, 0x55555555),
        TestVector("alternating_b", 0x55555555, 0xAAAAAAAA),
        TestVector("trace_sample", 0x34333231, 0x2179656B),
    ]

    rng = Random(0x20260508)
    for index in range(8):
        vectors.append(
            TestVector(
                name=f"random_{index:02d}",
                plaintext=rng.getrandbits(32),
                key=rng.getrandbits(32),
            )
        )

    return tuple(vectors)
