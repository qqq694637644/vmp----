"""多向量对拍回归。

这个测试只验证一件事：Triton 恢复出的结果，和未保护 / 受保护二进制在多组输入上必须一致。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from xor_recovery.triton_compare import compare_all_vectors
from xor_recovery.vectors import build_test_vectors


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"


class VectorComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        for required_path in (
            BUILD_DIR / "encrypt_demo.exe",
            BUILD_DIR / "encrypt_demo.protected.exe",
        ):
            if not required_path.exists():
                raise FileNotFoundError(f"缺少回归所需文件: {required_path}")

    def test_binary_comparison(self) -> None:
        vectors = tuple(
            vector
            for vector in build_test_vectors()
            if vector.name in {"trace_sample", "all_ones", "alternating_a"}
        )
        report = compare_all_vectors(BUILD_DIR, vectors)
        self.assertTrue(report.all_match)
        self.assertGreater(len(report.cases), 0)
        for case in report.cases:
            with self.subTest(vector=case.vector.name):
                self.assertEqual(case.verification.trace_result, case.verification.symbolic_result)
                self.assertEqual(case.verification.trace_result, case.verification.unprotected_result)
                self.assertEqual(case.verification.trace_result, case.verification.protected_result)


if __name__ == "__main__":
    unittest.main()
