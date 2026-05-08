"""恢复出的算法真值与测试向量。

这个模块只放纯函数和可重复的测试向量生成逻辑，不依赖 Triton，也不依赖
任何本地二进制。它的用途有两个：
1. 作为恢复结果的独立参考实现。
2. 为对拍测试生成一组稳定的输入向量。
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


def rotl32(value: int, shift: int) -> int:
    """32 位循环左移。"""
    shift &= 31
    value &= MASK32
    return ((value << shift) | (value >> ((32 - shift) & 31))) & MASK32


def bswap32(value: int) -> int:
    """32 位字节重排。"""
    value &= MASK32
    return int.from_bytes(value.to_bytes(4, byteorder="little"), byteorder="big")


def recovered_transform(plaintext: int, key: int) -> int:
    """恢复出的算法真值。

    这里必须和 `encrypt_demo.cpp` 里的 `XorTransform` 保持严格一致。
    """
    step1 = (plaintext + 0x13579BDF) & MASK32
    step2 = rotl32(key ^ 0x2468ACE0, 7)
    step3 = bswap32(step1 ^ step2)
    step4 = rotl32(plaintext ^ 0x11223344, 11)
    step5 = (step3 + step4) & MASK32
    step6 = (step5 * 0x9E3779B1) & MASK32
    step7 = step6 ^ ((key + 0x0F1E2D3C) & MASK32)
    return rotl32(step7, 3) ^ 0xA5A5A5A5


def format_u32_hex(value: int) -> str:
    """把 32 位整数格式化成固定宽度十六进制。"""
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
