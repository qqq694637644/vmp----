from __future__ import annotations

import unittest
from pathlib import Path

from xor_recovery.pipeline import build_config_from_trace, recover
from xor_recovery.trace_io import parse_trace
from xor_recovery.verification import verify_binary_consistency


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"


def rotl32(value: int, shift: int) -> int:
    shift &= 31
    value &= 0xFFFFFFFF
    return ((value << shift) | (value >> ((32 - shift) & 31))) & 0xFFFFFFFF


def bswap32(value: int) -> int:
    value &= 0xFFFFFFFF
    return int.from_bytes(value.to_bytes(4, byteorder="little"), byteorder="big")


def reference_transform(plaintext: int, key: int) -> int:
    step1 = (plaintext + 0x13579BDF) & 0xFFFFFFFF
    step2 = rotl32(key ^ 0x2468ACE0, 7)
    step3 = bswap32(step1 ^ step2)
    step4 = rotl32(plaintext ^ 0x11223344, 11)
    step5 = (step3 + step4) & 0xFFFFFFFF
    step6 = (step5 * 0x9E3779B1) & 0xFFFFFFFF
    step7 = step6 ^ ((key + 0x0F1E2D3C) & 0xFFFFFFFF)
    return rotl32(step7, 3) ^ 0xA5A5A5A5


class RecoveryPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
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

        verification = verify_binary_consistency(BUILD_DIR, result.taint.result_value, result.formulas)
        self.assertTrue(verification.all_match)
        self.assertEqual(verification.trace_result, expected_value)
        self.assertEqual(verification.symbolic_result, expected_value)
        self.assertEqual(verification.unprotected_result, expected_value)
        self.assertEqual(verification.protected_result, expected_value)


if __name__ == "__main__":
    unittest.main()
