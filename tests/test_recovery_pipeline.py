from __future__ import annotations

import unittest
from pathlib import Path

from xor_recovery.pipeline import build_config_from_trace, recover
from xor_recovery.trace_io import parse_trace


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"


class RecoveryPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        trace_log = BUILD_DIR / "trace.log"
        if not trace_log.exists():
            raise FileNotFoundError(f"缺少 VMP 轨迹日志: {trace_log}")

    def test_two_pass_recovery(self) -> None:
        trace_path = BUILD_DIR / "trace.log"
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
        for formula in result.formulas:
            self.assertIn("bvxor", formula.formula_text)
            self.assertIn("extract", formula.formula_text)
        self.assertEqual(
            [formula.byte_offset for formula in result.formulas],
            [0, 1, 2, 3],
        )
        expected_value = trace_metadata.entry_arguments.plaintext_value ^ trace_metadata.entry_arguments.key_value
        expected_bytes = expected_value.to_bytes(4, byteorder="little")
        self.assertEqual(result.taint.result_value, trace_metadata.result_value)
        self.assertEqual(result.taint.result_bytes, trace_metadata.result_bytes)
        self.assertEqual([formula.evaluated_value for formula in result.formulas], list(expected_bytes))


if __name__ == "__main__":
    unittest.main()
