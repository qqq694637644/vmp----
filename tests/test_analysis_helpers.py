from __future__ import annotations

import struct
import unittest
from pathlib import Path

from xor_recovery.analysis import collapse_memory_ranges, is_vm_context_label
from xor_recovery.snapshot import get_minimal_snapshot_items
from xor_recovery.trace_io import parse_trace


class AnalysisHelperTest(unittest.TestCase):
    def test_vm_context_label_filter(self) -> None:
        self.assertTrue(is_vm_context_label("vm_context+0x10"))
        self.assertFalse(is_vm_context_label("stack+0x10"))
        self.assertFalse(is_vm_context_label("0x0000000000000010"))

    def test_collapse_memory_ranges(self) -> None:
        regions = collapse_memory_ranges({(0x1000, 1), (0x1001, 3), (0x2000, 2)})
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0].base, 0x1000)
        self.assertEqual(regions[0].size, 4)
        self.assertEqual(regions[1].base, 0x2000)
        self.assertEqual(regions[1].size, 2)

    def test_minimal_snapshot_items(self) -> None:
        items = get_minimal_snapshot_items()
        self.assertIn("入口整数寄存器：RAX-R15、RSP、RIP、EFLAGS", items)
        self.assertIn("入口全量可读内存快照：所有可读提交页，单独写入快照文件用于 Triton 回放", items)
        self.assertIn("VM 上下文：函数内部虚拟机状态块、状态槽、表驱动区", items)
        self.assertIn("返回值：RAX，作为函数最终输出锚点", items)

    def test_parse_entry_memory_snapshot_file(self) -> None:
        snapshot_path = Path(__file__).resolve().parents[1] / "build" / "_test_entry_memory.snap"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_payload = [
            struct.pack("<4sI", b"VMSN", 1),
            struct.pack("<QQ", 0x3000, 4),
            bytes.fromhex("11 22 33 44"),
            struct.pack("<QQ", 0x4000, 2),
            bytes.fromhex("55 66"),
        ]
        snapshot_path.write_bytes(b"".join(snapshot_payload))

        trace_text = "\n".join(
            [
                "已定位 XorTransform，地址=0x000000014000FE80，RVA=0x000000000000FE80，大小=56",
                "已进入 XorTransform，返回地址=0x000000014000FEF8",
                "寄存器：RAX=0x0000000000000001，RBX=0x0000000000000002，RCX=0x0000000000000003，RDX=0x0000000000000004，RSI=0x0000000000000005，RDI=0x0000000000000006，RBP=0x0000000000000007，RSP=0x0000000000000008，R8=0x0000000000000009，R9=0x000000000000000A，R10=0x000000000000000B，R11=0x000000000000000C，R12=0x000000000000000D，R13=0x000000000000000E，R14=0x000000000000000F，R15=0x0000000000000010，EFLAGS=0x0000000000000002，CS=0x0000000000000033，DS=0x000000000000002B，ES=0x000000000000002B，FS=0x0000000000000053，GS=0x000000000000002B，SS=0x000000000000002B",
                "浮点状态：MXCSR=0x00001F80，MXCSR_MASK=0x0000FFFF",
                "XMM寄存器：XMM0=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，XMM1=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，XMM2=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，XMM3=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，XMM4=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，XMM5=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，XMM6=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，XMM7=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，XMM8=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，XMM9=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，XMM10=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，XMM11=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，XMM12=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，XMM13=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，XMM14=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，XMM15=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00",
                "YMM高位：YMM0=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，YMM1=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，YMM2=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，YMM3=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，YMM4=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，YMM5=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，YMM6=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，YMM7=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，YMM8=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，YMM9=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，YMM10=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，YMM11=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，YMM12=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，YMM13=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，YMM14=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00，YMM15=00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00",
                "栈快照=00 11 22 33 44 55 66 77",
                "VM上下文基址=0x0000000000002000",
                "VM上下文快照=AA BB CC DD",
                f"入口全量快照文件={snapshot_path.resolve()}，区域数=2，总字节=6",
                "附加快照[0]：基址=0x0000000000003000，大小=4，快照=11 22 33 44",
                "参数：RSP=0x0000000000000008，RCX=0x0000000000000003，RDX=0x0000000000000004，plaintext=31 32 33 34，key=61 62 63 64",
                "返回值：RAX=0x0000000000000007，bytes=07 00 00 00",
                "步骤 000001 | RIP=0x000000014000FE80 | 字节=90",
                "已离开 XorTransform，步骤数=1",
            ]
        )

        trace_path = Path(__file__).resolve().parents[1] / "build" / "_test_parse_extra_snapshot.log"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(trace_text, encoding="utf-8")
        try:
            trace = parse_trace(trace_path)
        finally:
            trace_path.unlink(missing_ok=True)
            snapshot_path.unlink(missing_ok=True)

        self.assertEqual(len(trace.entry_memory_snapshots), 2)
        self.assertEqual(trace.entry_memory_snapshots[0].base, 0x3000)
        self.assertEqual(trace.entry_memory_snapshots[0].bytes, bytes.fromhex("11 22 33 44"))
        self.assertEqual(trace.entry_memory_snapshots[1].base, 0x4000)
        self.assertEqual(trace.entry_memory_snapshots[1].bytes, bytes.fromhex("55 66"))
        self.assertEqual(len(trace.extra_memory_snapshots), 1)
        self.assertEqual(trace.extra_memory_snapshots[0].base, 0x3000)
        self.assertEqual(trace.extra_memory_snapshots[0].bytes, bytes.fromhex("11 22 33 44"))


if __name__ == "__main__":
    unittest.main()
