from __future__ import annotations

import re
from pathlib import Path

from .models import TraceStep


ENTRY_RE = re.compile(r"已定位 XorTransform，地址=(0x[0-9A-Fa-f]+)(?:，RVA=(0x[0-9A-Fa-f]+))?，大小=(\d+)")
STEP_RE = re.compile(
    r"^步骤\s+(\d+)\s+\|\s+RIP=(0x[0-9A-Fa-f]+)\s+\|\s+字节=([0-9A-Fa-f ]+)(?:\s+\|\s+行号=(\d+))?$"
)
EXIT_RE = re.compile(r"已离开 XorTransform，步骤数=(\d+)")


def parse_hex_bytes(text: str) -> bytes:
    return bytes.fromhex(text.replace(" ", ""))


def parse_trace(trace_path: Path) -> tuple[int, int, tuple[TraceStep, ...]]:
    entry_address = 0
    function_size = 0
    steps: list[TraceStep] = []

    for raw_line in trace_path.read_text(encoding="utf-8").splitlines():
        entry_match = ENTRY_RE.search(raw_line)
        if entry_match is not None:
            entry_address = int(entry_match.group(1), 16)
            function_size = int(entry_match.group(3))
            continue

        step_match = STEP_RE.match(raw_line)
        if step_match is not None:
            steps.append(
                TraceStep(
                    index=int(step_match.group(1)),
                    address=int(step_match.group(2), 16),
                    opcode=parse_hex_bytes(step_match.group(3)),
                    line_number=int(step_match.group(4)) if step_match.group(4) is not None else None,
                )
            )
            continue

        if EXIT_RE.search(raw_line) is not None:
            break

    if entry_address == 0 or function_size == 0:
        raise ValueError("轨迹里没有找到 XorTransform 入口信息")
    if not steps:
        raise ValueError("轨迹里没有找到可重放的步骤")

    return entry_address, function_size, tuple(steps)
