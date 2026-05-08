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
        self.assertIsNotNone(trace_metadata.stack_pointer)
        self.assertIsNotNone(trace_metadata.return_address)

        config = build_config_from_trace(
            trace_metadata,
        )
        result = recover(trace_path, config)

        self.assertEqual(len(result.formulas), 4)
        self.assertGreater(len(result.taint.tainted_steps), 0)
        self.assertEqual(len(result.taint.output_roots), 1)
        sink_address = next(iter(result.taint.output_roots))
        self.assertEqual(result.taint.output_sizes[sink_address], 4)
        for formula in result.formulas:
            self.assertIn("bvxor", formula.formula_text)
            self.assertIn("extract", formula.formula_text)
        self.assertEqual(
            [formula.output_address for formula in result.formulas],
            [sink_address, sink_address + 1, sink_address + 2, sink_address + 3],
        )
        expected_ciphertext = [p ^ k for p, k in zip(trace_metadata.entry_arguments.plaintext, trace_metadata.entry_arguments.key)]
        self.assertEqual([formula.evaluated_value for formula in result.formulas], expected_ciphertext)


if __name__ == "__main__":
    unittest.main()
