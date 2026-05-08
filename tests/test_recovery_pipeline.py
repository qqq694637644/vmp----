"""恢复链的端到端回归测试。

这个测试不是在重复实现算法，而是在验证两件事：
1. 第一遍污点分析和第二遍符号执行能否在受保护样本上跑通。
2. 恢复出来的公式是否能和未保护 / 受保护二进制的真实输出一致。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from xor_recovery.pipeline import build_config_from_trace, recover
from xor_recovery.reference import recovered_transform
from xor_recovery.trace_io import parse_trace
from xor_recovery.verification import verify_binary_consistency


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"
class RecoveryPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 回归测试依赖已经生成好的二进制和 trace，缺一个就直接失败，避免伪通过。
        for required_path in (
            BUILD_DIR / "trace_protected_full.log",
            BUILD_DIR / "encrypt_demo.exe",
            BUILD_DIR / "encrypt_demo.protected.exe",
        ):
            if not required_path.exists():
                raise FileNotFoundError(f"缺少回归所需文件: {required_path}")

    def test_two_pass_recovery(self) -> None:
        trace_path = BUILD_DIR / "trace_protected_full.log"
        trace_metadata = parse_trace(trace_path)
        entry_address, function_size, steps = trace_metadata
        self.assertGreater(entry_address, 0)
        self.assertGreater(function_size, 0)
        self.assertGreater(len(steps), 0)
        self.assertIsNotNone(trace_metadata.entry_arguments)
        self.assertIsNotNone(trace_metadata.entry_vector_state)
        self.assertIsNotNone(trace_metadata.vm_context_base)
        self.assertIsNotNone(trace_metadata.vm_context_bytes)
        self.assertIsNotNone(trace_metadata.stack_pointer)
        self.assertIsNotNone(trace_metadata.return_address)
        self.assertIsNotNone(trace_metadata.result_bytes)
        self.assertIsNotNone(trace_metadata.result_value)
        self.assertEqual(len(trace_metadata.entry_vector_state.xmm_registers), 16)
        self.assertEqual(len(trace_metadata.entry_vector_state.ymm_high_registers), 16)
        self.assertEqual(len(trace_metadata.vm_context_bytes), 0x400)

        config = build_config_from_trace(
            trace_metadata,
        )
        self.assertIsNotNone(config.entry_vector_state)
        self.assertIsNotNone(config.vm_context_region)
        result = recover(trace_path, config)

        self.assertEqual(result.algorithm.result_name, "reg:rax")
        self.assertGreater(len(result.algorithm.simplified_ast_text), 0)
        self.assertGreater(len(result.algorithm.llvm_ir), 0)
        self.assertGreater(len(result.algorithm.human_readable_text), 0)
        self.assertIn("plaintext", result.algorithm.human_readable_text)
        self.assertIn("key", result.algorithm.human_readable_text)
        self.assertIn("rotl32", result.algorithm.human_readable_text)
        self.assertIn("define i64", result.algorithm.llvm_ir)

        self.assertEqual(len(result.formulas), 4)
        self.assertGreater(len(result.taint.tainted_steps), 0)
        self.assertEqual(list(result.taint.result_roots.keys()), ["reg:rax"])
        self.assertEqual(result.taint.result_sizes["reg:rax"], 4)
        self.assertTrue(result.taint.sink_reached)
        self.assertTrue(result.taint.sink_tainted)
        self.assertEqual(result.taint.replayed_result_value, result.taint.result_value)
        self.assertEqual(
            [formula.byte_offset for formula in result.formulas],
            [0, 1, 2, 3],
        )
        expected_value = reference_transform(
            trace_metadata.entry_arguments.plaintext_value,
            trace_metadata.entry_arguments.key_value,
        )
        expected_bytes = expected_value.to_bytes(4, byteorder="little")
        self.assertEqual(result.taint.result_value, trace_metadata.result_value)
        self.assertEqual(result.taint.result_bytes, trace_metadata.result_bytes)
        self.assertEqual(result.taint.result_value, expected_value)
        self.assertEqual([formula.evaluated_value for formula in result.formulas], list(expected_bytes))

        verification = verify_binary_consistency(
            BUILD_DIR,
            trace_metadata.entry_arguments.plaintext_value,
            trace_metadata.entry_arguments.key_value,
            result.taint.result_value,
            result.formulas,
        )
        self.assertTrue(verification.all_match)
        self.assertEqual(verification.trace_result, expected_value)
        self.assertEqual(verification.symbolic_result, expected_value)
        self.assertEqual(verification.unprotected_result, expected_value)
        self.assertEqual(verification.protected_result, expected_value)


if __name__ == "__main__":
    unittest.main()
