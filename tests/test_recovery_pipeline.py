from __future__ import annotations

import unittest
from pathlib import Path

from xor_recovery.pipeline import build_config, recover
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
        entry_address, function_size, steps = parse_trace(trace_path)
        self.assertGreater(entry_address, 0)
        self.assertGreater(function_size, 0)
        self.assertGreater(len(steps), 0)

        config = build_config(
            plaintext=b"1234",
            key=b"key!",
            entry_address=entry_address,
            stack_base=0x70000000,
            plaintext_base=0x10000000,
            key_base=0x10001000,
            output_base=0x10002000,
            return_address=entry_address + function_size + 0x1000,
        )
        result = recover(trace_path, config)

        self.assertEqual(len(result.formulas), 4)
        self.assertGreater(len(result.taint.tainted_steps), 0)
        self.assertEqual(list(result.taint.output_roots.keys()), [0x70001CE8])
        self.assertEqual(result.taint.output_sizes[0x70001CE8], 4)
        for formula in result.formulas:
            self.assertIn("bvxor", formula.formula_text)
            self.assertIn("extract", formula.formula_text)
        self.assertEqual(
            [formula.output_address for formula in result.formulas],
            [0x70001CE8, 0x70001CE9, 0x70001CEA, 0x70001CEB],
        )


if __name__ == "__main__":
    unittest.main()
