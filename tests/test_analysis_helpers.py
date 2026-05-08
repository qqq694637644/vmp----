from __future__ import annotations

import unittest

from xor_recovery.analysis import is_vm_context_label
from xor_recovery.snapshot import get_minimal_snapshot_items


class AnalysisHelperTest(unittest.TestCase):
    def test_vm_context_label_filter(self) -> None:
        self.assertTrue(is_vm_context_label("vm_context+0x10"))
        self.assertFalse(is_vm_context_label("stack+0x10"))
        self.assertFalse(is_vm_context_label("0x0000000000000010"))

    def test_minimal_snapshot_items(self) -> None:
        items = get_minimal_snapshot_items()
        self.assertIn("入口整数寄存器：RAX-R15、RSP、RIP、EFLAGS", items)
        self.assertIn("VM 上下文：函数内部虚拟机状态块、状态槽、表驱动区", items)
        self.assertIn("返回值：RAX，作为函数最终输出锚点", items)


if __name__ == "__main__":
    unittest.main()
