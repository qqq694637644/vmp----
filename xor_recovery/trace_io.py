from __future__ import annotations

import re
from pathlib import Path

from .models import EntryArguments, EntryRegisters, TraceMetadata, TraceStep


ENTRY_RE = re.compile(r"已定位 XorTransform，地址=(0x[0-9A-Fa-f]+)(?:，RVA=(0x[0-9A-Fa-f]+))?，大小=(\d+)")
ENTER_RE = re.compile(r"^已进入 XorTransform，返回地址=(0x[0-9A-Fa-f]+)$")
REGS_RE = re.compile(
    r"^寄存器：RAX=(0x[0-9A-Fa-f]+)，RBX=(0x[0-9A-Fa-f]+)，RCX=(0x[0-9A-Fa-f]+)，RDX=(0x[0-9A-Fa-f]+)，RSI=(0x[0-9A-Fa-f]+)，RDI=(0x[0-9A-Fa-f]+)，RBP=(0x[0-9A-Fa-f]+)，RSP=(0x[0-9A-Fa-f]+)，R8=(0x[0-9A-Fa-f]+)，R9=(0x[0-9A-Fa-f]+)，R10=(0x[0-9A-Fa-f]+)，R11=(0x[0-9A-Fa-f]+)，R12=(0x[0-9A-Fa-f]+)，R13=(0x[0-9A-Fa-f]+)，R14=(0x[0-9A-Fa-f]+)，R15=(0x[0-9A-Fa-f]+)，EFLAGS=(0x[0-9A-Fa-f]+)$"
)
STACK_RE = re.compile(r"^栈快照=([0-9A-Fa-f ]+)$")
ARGS_RE = re.compile(
    r"^参数：RSP=(0x[0-9A-Fa-f]+)，RCX=(0x[0-9A-Fa-f]+)，RDX=(0x[0-9A-Fa-f]+)，plaintext=([0-9A-Fa-f ]+)，key=([0-9A-Fa-f ]+)$"
)
RESULT_RE = re.compile(r"^返回值：RAX=(0x[0-9A-Fa-f]+)，bytes=([0-9A-Fa-f ]+)$")
STEP_RE = re.compile(
    r"^步骤\s+(\d+)\s+\|\s+RIP=(0x[0-9A-Fa-f]+)\s+\|\s+字节=([0-9A-Fa-f ]+)(?:\s+\|\s+行号=(\d+))?$"
)
EXIT_RE = re.compile(r"已离开 XorTransform，步骤数=(\d+)")


def parse_hex_bytes(text: str) -> bytes:
    return bytes.fromhex(text.replace(" ", ""))


def parse_trace(trace_path: Path) -> TraceMetadata:
    entry_address = 0
    function_size = 0
    steps: list[TraceStep] = []
    entry_arguments: EntryArguments | None = None
    entry_registers: EntryRegisters | None = None
    stack_pointer: int | None = None
    return_address: int | None = None
    result_value: int | None = None
    result_bytes: bytes | None = None
    stack_bytes: bytes | None = None

    for raw_line in trace_path.read_text(encoding="utf-8").splitlines():
        entry_match = ENTRY_RE.search(raw_line)
        if entry_match is not None:
            entry_address = int(entry_match.group(1), 16)
            function_size = int(entry_match.group(3))
            continue

        enter_match = ENTER_RE.match(raw_line)
        if enter_match is not None:
            return_address = int(enter_match.group(1), 16)
            continue

        regs_match = REGS_RE.match(raw_line)
        if regs_match is not None:
            entry_registers = EntryRegisters(
                rax=int(regs_match.group(1), 16),
                rbx=int(regs_match.group(2), 16),
                rcx=int(regs_match.group(3), 16),
                rdx=int(regs_match.group(4), 16),
                rsi=int(regs_match.group(5), 16),
                rdi=int(regs_match.group(6), 16),
                rbp=int(regs_match.group(7), 16),
                rsp=int(regs_match.group(8), 16),
                r8=int(regs_match.group(9), 16),
                r9=int(regs_match.group(10), 16),
                r10=int(regs_match.group(11), 16),
                r11=int(regs_match.group(12), 16),
                r12=int(regs_match.group(13), 16),
                r13=int(regs_match.group(14), 16),
                r14=int(regs_match.group(15), 16),
                r15=int(regs_match.group(16), 16),
                eflags=int(regs_match.group(17), 16),
            )
            continue

        stack_match = STACK_RE.match(raw_line)
        if stack_match is not None:
            stack_bytes = parse_hex_bytes(stack_match.group(1))
            continue

        args_match = ARGS_RE.match(raw_line)
        if args_match is not None:
            entry_arguments = EntryArguments(
                plaintext_value=int(args_match.group(2), 16),
                key_value=int(args_match.group(3), 16),
                plaintext=parse_hex_bytes(args_match.group(4)),
                key=parse_hex_bytes(args_match.group(5)),
            )
            stack_pointer = int(args_match.group(1), 16)
            continue

        result_match = RESULT_RE.match(raw_line)
        if result_match is not None:
            result_value = int(result_match.group(1), 16)
            result_bytes = parse_hex_bytes(result_match.group(2))
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
    if entry_arguments is None:
        raise ValueError("轨迹里没有找到 XorTransform 入口参数")
    if entry_registers is None:
        raise ValueError("轨迹里没有找到 XorTransform 入口寄存器")
    if stack_pointer is None:
        raise ValueError("轨迹里没有找到 XorTransform 栈指针")
    if return_address is None:
        raise ValueError("轨迹里没有找到 XorTransform 返回地址")
    if result_value is None or result_bytes is None:
        raise ValueError("轨迹里没有找到 XorTransform 返回值")
    if stack_bytes is None:
        raise ValueError("轨迹里没有找到 XorTransform 栈快照")
    if not steps:
        raise ValueError("轨迹里没有找到可重放的步骤")

    return TraceMetadata(
        entry_address=entry_address,
        function_size=function_size,
        steps=tuple(steps),
        entry_arguments=entry_arguments,
        entry_registers=entry_registers,
        stack_pointer=stack_pointer,
        return_address=return_address,
        result_value=result_value,
        result_bytes=result_bytes,
        stack_bytes=stack_bytes,
    )
