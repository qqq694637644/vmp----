from __future__ import annotations

import re
from pathlib import Path

from .models import EntryArguments, EntryRegisters, EntryVectorState, MemorySnapshot, TraceMetadata, TraceStep


ENTRY_RE = re.compile(r"已定位 XorTransform，地址=(0x[0-9A-Fa-f]+)(?:，RVA=(0x[0-9A-Fa-f]+))?，大小=(\d+)")
ENTER_RE = re.compile(r"^已进入 XorTransform，返回地址=(0x[0-9A-Fa-f]+)$")
REGS_RE = re.compile(
    r"^寄存器：RAX=(0x[0-9A-Fa-f]+)，RBX=(0x[0-9A-Fa-f]+)，RCX=(0x[0-9A-Fa-f]+)，RDX=(0x[0-9A-Fa-f]+)，RSI=(0x[0-9A-Fa-f]+)，RDI=(0x[0-9A-Fa-f]+)，RBP=(0x[0-9A-Fa-f]+)，RSP=(0x[0-9A-Fa-f]+)，R8=(0x[0-9A-Fa-f]+)，R9=(0x[0-9A-Fa-f]+)，R10=(0x[0-9A-Fa-f]+)，R11=(0x[0-9A-Fa-f]+)，R12=(0x[0-9A-Fa-f]+)，R13=(0x[0-9A-Fa-f]+)，R14=(0x[0-9A-Fa-f]+)，R15=(0x[0-9A-Fa-f]+)，EFLAGS=(0x[0-9A-Fa-f]+)，CS=(0x[0-9A-Fa-f]+)，DS=(0x[0-9A-Fa-f]+)，ES=(0x[0-9A-Fa-f]+)，FS=(0x[0-9A-Fa-f]+)，GS=(0x[0-9A-Fa-f]+)，SS=(0x[0-9A-Fa-f]+)$"
)
FLOAT_STATE_RE = re.compile(r"^浮点状态：MXCSR=(0x[0-9A-Fa-f]+)，MXCSR_MASK=(0x[0-9A-Fa-f]+)$")
XMM_PREFIX = "XMM寄存器："
YMM_PREFIX = "YMM高位："
VM_CONTEXT_BASE_RE = re.compile(r"^VM上下文基址=(0x[0-9A-Fa-f]+)$")
VM_CONTEXT_RE = re.compile(r"^VM上下文快照=([0-9A-Fa-f ]+)$")
EXTRA_SNAPSHOT_RE = re.compile(
    r"^附加快照\[(\d+)\]：基址=(0x[0-9A-Fa-f]+)，大小=(\d+)，快照=([0-9A-Fa-f ]+)$"
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
    entry_vector_state: EntryVectorState | None = None
    vm_context_base: int | None = None
    vm_context_bytes: bytes | None = None
    extra_memory_snapshots: list[MemorySnapshot] = []
    stack_pointer: int | None = None
    return_address: int | None = None
    result_value: int | None = None
    result_bytes: bytes | None = None
    stack_bytes: bytes | None = None
    mxcsr: int | None = None
    mxcsr_mask: int | None = None
    xmm_registers: tuple[bytes, ...] | None = None
    ymm_high_registers: tuple[bytes, ...] | None = None

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
                cs=int(regs_match.group(18), 16),
                ds=int(regs_match.group(19), 16),
                es=int(regs_match.group(20), 16),
                fs=int(regs_match.group(21), 16),
                gs=int(regs_match.group(22), 16),
                ss=int(regs_match.group(23), 16),
            )
            continue

        float_match = FLOAT_STATE_RE.match(raw_line)
        if float_match is not None:
            mxcsr = int(float_match.group(1), 16)
            mxcsr_mask = int(float_match.group(2), 16)
            continue

        if raw_line.startswith(XMM_PREFIX):
            if mxcsr is None or mxcsr_mask is None:
                raise ValueError("XMM 寄存器快照先于浮点状态出现，日志格式不正确")

            vector_payload = raw_line.removeprefix(XMM_PREFIX)
            xmm_items = vector_payload.split("，")
            if len(xmm_items) != 16:
                raise ValueError("XMM 寄存器数量不正确")

            parsed_xmm_registers: list[bytes] = []
            for index, item in enumerate(xmm_items):
                name, value = item.split("=", 1)
                expected_name = f"XMM{index}"
                if name != expected_name:
                    raise ValueError(f"XMM 寄存器顺序错误: 期望 {expected_name}，实际 {name}")
                parsed_xmm_registers.append(parse_hex_bytes(value))

            xmm_registers = tuple(parsed_xmm_registers)
            continue

        if raw_line.startswith(YMM_PREFIX):
            if mxcsr is None or mxcsr_mask is None:
                raise ValueError("YMM 高位快照先于浮点状态出现，日志格式不正确")

            vector_payload = raw_line.removeprefix(YMM_PREFIX)
            ymm_items = vector_payload.split("，")
            if len(ymm_items) != 16:
                raise ValueError("YMM 高位数量不正确")

            parsed_ymm_registers: list[bytes] = []
            for index, item in enumerate(ymm_items):
                name, value = item.split("=", 1)
                expected_name = f"YMM{index}"
                if name != expected_name:
                    raise ValueError(f"YMM 高位顺序错误: 期望 {expected_name}，实际 {name}")
                parsed_ymm_registers.append(parse_hex_bytes(value))

            ymm_high_registers = tuple(parsed_ymm_registers)
            continue

        context_base_match = VM_CONTEXT_BASE_RE.match(raw_line)
        if context_base_match is not None:
            vm_context_base = int(context_base_match.group(1), 16)
            continue

        context_match = VM_CONTEXT_RE.match(raw_line)
        if context_match is not None:
            vm_context_bytes = parse_hex_bytes(context_match.group(1))
            continue

        extra_snapshot_match = EXTRA_SNAPSHOT_RE.match(raw_line)
        if extra_snapshot_match is not None:
            snapshot_base = int(extra_snapshot_match.group(2), 16)
            snapshot_size = int(extra_snapshot_match.group(3))
            snapshot_bytes = parse_hex_bytes(extra_snapshot_match.group(4))
            if len(snapshot_bytes) != snapshot_size:
                raise ValueError("附加快照长度与日志不一致")
            extra_memory_snapshots.append(MemorySnapshot(base=snapshot_base, bytes=snapshot_bytes))
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
    if mxcsr is None or mxcsr_mask is None or xmm_registers is None or ymm_high_registers is None:
        raise ValueError("轨迹里没有找到 XorTransform 向量状态快照")
    if vm_context_base is None or vm_context_bytes is None:
        raise ValueError("轨迹里没有找到 XorTransform VM 上下文快照")
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
        entry_vector_state=EntryVectorState(
            mxcsr=mxcsr,
            mxcsr_mask=mxcsr_mask,
            xmm_registers=xmm_registers,
            ymm_high_registers=ymm_high_registers,
        ),
        vm_context_base=vm_context_base,
        vm_context_bytes=vm_context_bytes,
        extra_memory_snapshots=tuple(extra_memory_snapshots),
        stack_pointer=stack_pointer,
        return_address=return_address,
        result_value=result_value,
        result_bytes=result_bytes,
        stack_bytes=stack_bytes,
    )
