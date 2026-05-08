"""多向量对拍回归。

这个测试只验证一件事：恢复出的纯函数参考实现，和未保护 / 受保护二进制在多组输入上必须一致。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from xor_recovery.reference import build_test_vectors, recovered_transform
from xor_recovery.vector_compare import compare_all_vectors


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

    def test_reference_transform_matches_vectors(self) -> None:
        sample = next(vector for vector in build_test_vectors() if vector.name == "trace_sample")
        self.assertEqual(recovered_transform(sample.plaintext, sample.key), 0x43FBB6A0)

    def test_binary_comparison(self) -> None:
        report = compare_all_vectors(BUILD_DIR, build_test_vectors())
        self.assertTrue(report.all_match)
        self.assertGreater(len(report.cases), 0)
        for case in report.cases:
            with self.subTest(vector=case.vector.name):
                self.assertEqual(case.reference_result, case.unprotected_result)
                self.assertEqual(case.reference_result, case.protected_result)


if __name__ == "__main__":
    unittest.main()
