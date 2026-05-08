from __future__ import annotations

import re
import struct
from pathlib import Path

from .models import (
    ConcreteRegisterSnapshot,
    EntryArguments,
    EntryRegisters,
    EntryVectorState,
    MemorySnapshot,
    TraceMetadata,
    TraceStep,
)


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
ENTRY_MEMORY_SNAPSHOT_FILE_RE = re.compile(r"^入口全量快照文件=(.+?)，区域数=(\d+)，总字节=(\d+)$")
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
STEP_STATE_PREFIX = "步骤状态："
EXIT_RE = re.compile(r"已离开 XorTransform，步骤数=(\d+)")


def parse_hex_bytes(text: str) -> bytes:
    return bytes.fromhex(text.replace(" ", ""))


def parse_step_register_snapshot(raw_line: str) -> ConcreteRegisterSnapshot:
    if not raw_line.startswith(STEP_STATE_PREFIX):
        raise ValueError("步骤状态行格式不正确")

    payload = raw_line.removeprefix(STEP_STATE_PREFIX)
    items = payload.split("，")
    expected_names = (
        "RIP",
        "RAX",
        "RBX",
        "RCX",
        "RDX",
        "RSI",
        "RDI",
        "RBP",
        "RSP",
        "R8",
        "R9",
        "R10",
        "R11",
        "R12",
        "R13",
        "R14",
        "R15",
        "EFLAGS",
        "CS",
        "DS",
        "ES",
        "FS",
        "GS",
        "SS",
    )
    if len(items) != len(expected_names):
        raise ValueError("步骤状态寄存器数量不正确")

    values: dict[str, int] = {}
    for index, item in enumerate(items):
        name, value = item.split("=", 1)
        expected_name = expected_names[index]
        if name != expected_name:
            raise ValueError(f"步骤状态寄存器顺序错误: 期望 {expected_name}，实际 {name}")
        values[name.lower()] = int(value, 16)

    return ConcreteRegisterSnapshot(
        rip=values["rip"],
        rax=values["rax"],
        rbx=values["rbx"],
        rcx=values["rcx"],
        rdx=values["rdx"],
        rsi=values["rsi"],
        rdi=values["rdi"],
        rbp=values["rbp"],
        rsp=values["rsp"],
        r8=values["r8"],
        r9=values["r9"],
        r10=values["r10"],
        r11=values["r11"],
        r12=values["r12"],
        r13=values["r13"],
        r14=values["r14"],
        r15=values["r15"],
        eflags=values["eflags"],
        cs=values["cs"],
        ds=values["ds"],
        es=values["es"],
        fs=values["fs"],
        gs=values["gs"],
        ss=values["ss"],
    )


def load_entry_memory_snapshots(snapshot_path: Path) -> tuple[MemorySnapshot, ...]:
    raw = snapshot_path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"入口全量快照文件过短: {snapshot_path}")
    if raw[:4] != b"VMSN":
        raise ValueError(f"入口全量快照文件头不正确: {snapshot_path}")

    version = struct.unpack_from("<I", raw, 4)[0]
    if version != 1:
        raise ValueError(f"入口全量快照版本不受支持: {version}")

    offset = 8
    snapshots: list[MemorySnapshot] = []
    while offset < len(raw):
        if offset + 16 > len(raw):
            raise ValueError(f"入口全量快照文件被截断: {snapshot_path}")

        base, size = struct.unpack_from("<QQ", raw, offset)
        offset += 16
        if size == 0:
            raise ValueError(f"入口全量快照区域大小为 0: {snapshot_path}")
        if offset + size > len(raw):
            raise ValueError(f"入口全量快照文件被截断: {snapshot_path}")

        snapshots.append(MemorySnapshot(base=base, bytes=raw[offset:offset + size]))
        offset += size

    if not snapshots:
        raise ValueError(f"入口全量快照文件为空: {snapshot_path}")

    return tuple(snapshots)


def parse_trace(trace_path: Path) -> TraceMetadata:
    entry_address = 0
    function_size = 0
    steps: list[TraceStep] = []
    entry_arguments: EntryArguments | None = None
    entry_registers: EntryRegisters | None = None
    entry_vector_state: EntryVectorState | None = None
    vm_context_base: int | None = None
    vm_context_bytes: bytes | None = None
    entry_memory_snapshot_file: Path | None = None
    entry_memory_snapshot_count: int | None = None
    entry_memory_snapshot_total_bytes: int | None = None
    entry_memory_snapshots: tuple[MemorySnapshot, ...] = ()
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
    pending_step: TraceStep | None = None

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

        entry_snapshot_match = ENTRY_MEMORY_SNAPSHOT_FILE_RE.match(raw_line)
        if entry_snapshot_match is not None:
            entry_memory_snapshot_file = Path(entry_snapshot_match.group(1))
            entry_memory_snapshot_count = int(entry_snapshot_match.group(2))
            entry_memory_snapshot_total_bytes = int(entry_snapshot_match.group(3))
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
            if pending_step is not None:
                raise ValueError("步骤状态缺失，上一条步骤没有对应的状态快照")

            pending_step = TraceStep(
                index=int(step_match.group(1)),
                address=int(step_match.group(2), 16),
                opcode=parse_hex_bytes(step_match.group(3)),
                line_number=int(step_match.group(4)) if step_match.group(4) is not None else None,
            )
            continue

        if raw_line.startswith(STEP_STATE_PREFIX):
            if pending_step is None:
                raise ValueError("步骤状态出现时没有对应的步骤记录")
            steps.append(
                TraceStep(
                    index=pending_step.index,
                    address=pending_step.address,
                    opcode=pending_step.opcode,
                    line_number=pending_step.line_number,
                    state=parse_step_register_snapshot(raw_line),
                )
            )
            pending_step = None
            continue

        if EXIT_RE.search(raw_line) is not None:
            if pending_step is not None:
                raise ValueError("步骤状态缺失，日志在退出前未完成最后一条步骤")
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
    if entry_memory_snapshot_file is None:
        raise ValueError("轨迹里没有找到入口全量内存快照文件")
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
    if pending_step is not None:
        raise ValueError("步骤状态缺失，日志结尾前未完成最后一条步骤")

    entry_memory_snapshots = load_entry_memory_snapshots(entry_memory_snapshot_file)
    if entry_memory_snapshot_count is not None and len(entry_memory_snapshots) != entry_memory_snapshot_count:
        raise ValueError("入口全量快照区域数与日志不一致")
    if entry_memory_snapshot_total_bytes is not None:
        total_bytes = sum(snapshot.size for snapshot in entry_memory_snapshots)
        if total_bytes != entry_memory_snapshot_total_bytes:
            raise ValueError("入口全量快照总字节数与日志不一致")

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
        entry_memory_snapshots=entry_memory_snapshots,
        extra_memory_snapshots=tuple(extra_memory_snapshots),
        stack_pointer=stack_pointer,
        return_address=return_address,
        result_value=result_value,
        result_bytes=result_bytes,
        stack_bytes=stack_bytes,
    )
