from __future__ import annotations

from typing import Callable

from triton import ARCH, EXCEPTION, Instruction, REG, TritonContext

from .models import RecoveryConfig, TraceStep


StepObserver = Callable[[TraceStep, Instruction, TritonContext], None]


def make_register(ctx: TritonContext, reg_const: int):
    return ctx.getRegister(reg_const)


class ReplayStateMismatch(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)


def compare_register_snapshot(ctx: TritonContext, step: TraceStep) -> None:
    if step.state is None:
        raise ValueError(f"步骤 {step.index} 缺少状态快照，无法做分歧比较")

    expected = step.state
    comparisons: tuple[tuple[str, int, int], ...] = (
        ("RIP", ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.RIP), False), expected.rip),
        ("RAX", ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.RAX), False), expected.rax),
        ("RBX", ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.RBX), False), expected.rbx),
        ("RCX", ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.RCX), False), expected.rcx),
        ("RDX", ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.RDX), False), expected.rdx),
        ("RSI", ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.RSI), False), expected.rsi),
        ("RDI", ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.RDI), False), expected.rdi),
        ("RBP", ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.RBP), False), expected.rbp),
        ("RSP", ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.RSP), False), expected.rsp),
        ("R8", ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.R8), False), expected.r8),
        ("R9", ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.R9), False), expected.r9),
        ("R10", ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.R10), False), expected.r10),
        ("R11", ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.R11), False), expected.r11),
        ("R12", ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.R12), False), expected.r12),
        ("R13", ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.R13), False), expected.r13),
        ("R14", ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.R14), False), expected.r14),
        ("R15", ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.R15), False), expected.r15),
    )

    for name, actual, expected_value in comparisons:
        if actual != expected_value:
            raise ReplayStateMismatch(
                f"step={step.index} rip={step.address:#x} register={name} 期望={expected_value:#x} 实际={actual:#x}"
            )


def zero_general_registers(ctx: TritonContext) -> None:
    # 只保留我们手工布置的参数和栈状态，其余寄存器全部清零。
    for reg_const in (
        REG.X86_64.RAX,
        REG.X86_64.RBX,
        REG.X86_64.RCX,
        REG.X86_64.RDX,
        REG.X86_64.RSI,
        REG.X86_64.RDI,
        REG.X86_64.RBP,
        REG.X86_64.R8,
        REG.X86_64.R9,
        REG.X86_64.R10,
        REG.X86_64.R11,
        REG.X86_64.R12,
        REG.X86_64.R13,
        REG.X86_64.R14,
        REG.X86_64.R15,
        REG.X86_64.RIP,
        REG.X86_64.EFLAGS,
        REG.X86_64.CS,
        REG.X86_64.DS,
        REG.X86_64.ES,
        REG.X86_64.FS,
        REG.X86_64.GS,
        REG.X86_64.SS,
    ):
        ctx.setConcreteRegisterValue(make_register(ctx, reg_const), 0)


def zero_vector_registers(ctx: TritonContext) -> None:
    # 向量寄存器和 MXCSR 也是执行语义的一部分，不能沿用 Triton 上一次残留的状态。
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.MXCSR), 0)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.MXCSR_MASK), 0)
    for index in range(32):
        xmm_register = getattr(REG.X86_64, f"XMM{index}")
        ymm_register = getattr(REG.X86_64, f"YMM{index}")
        ctx.setConcreteRegisterValue(make_register(ctx, xmm_register), 0)
        ctx.setConcreteRegisterValue(make_register(ctx, ymm_register), 0)


def apply_entry_registers(ctx: TritonContext, config: RecoveryConfig) -> None:
    if config.entry_registers is None:
        raise ValueError("入口寄存器快照缺失，无法初始化 Triton 上下文")

    entry = config.entry_registers
    register_values = (
        (REG.X86_64.RAX, entry.rax),
        (REG.X86_64.RBX, entry.rbx),
        (REG.X86_64.RCX, entry.rcx),
        (REG.X86_64.RDX, entry.rdx),
        (REG.X86_64.RSI, entry.rsi),
        (REG.X86_64.RDI, entry.rdi),
        (REG.X86_64.RBP, entry.rbp),
        (REG.X86_64.RSP, entry.rsp),
        (REG.X86_64.R8, entry.r8),
        (REG.X86_64.R9, entry.r9),
        (REG.X86_64.R10, entry.r10),
        (REG.X86_64.R11, entry.r11),
        (REG.X86_64.R12, entry.r12),
        (REG.X86_64.R13, entry.r13),
        (REG.X86_64.R14, entry.r14),
        (REG.X86_64.R15, entry.r15),
    )
    for reg_const, value in register_values:
        ctx.setConcreteRegisterValue(make_register(ctx, reg_const), value)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.EFLAGS), entry.eflags)


def apply_entry_vector_state(ctx: TritonContext, config: RecoveryConfig) -> None:
    if config.entry_vector_state is None:
        raise ValueError("入口向量状态快照缺失，无法初始化 Triton 上下文")

    vector_state = config.entry_vector_state
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.MXCSR), vector_state.mxcsr)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.MXCSR_MASK), vector_state.mxcsr_mask)
    if len(vector_state.xmm_registers) != 16:
        raise ValueError("入口 XMM 寄存器数量不正确")
    if len(vector_state.ymm_high_registers) != 16:
        raise ValueError("入口 YMM 高位寄存器数量不正确")

    for index, xmm_bytes in enumerate(vector_state.xmm_registers):
        xmm_register = getattr(REG.X86_64, f"XMM{index}")
        ctx.setConcreteRegisterValue(make_register(ctx, xmm_register), int.from_bytes(xmm_bytes, byteorder="little"))

    for index, ymm_high_bytes in enumerate(vector_state.ymm_high_registers):
        ymm_register = getattr(REG.X86_64, f"YMM{index}")
        ymm_bytes = vector_state.xmm_registers[index] + ymm_high_bytes
        ctx.setConcreteRegisterValue(make_register(ctx, ymm_register), int.from_bytes(ymm_bytes, byteorder="little"))


def initialize_context(config: RecoveryConfig) -> TritonContext:
    ctx = TritonContext()
    ctx.setArchitecture(ARCH.X86_64)
    zero_general_registers(ctx)
    zero_vector_registers(ctx)
    apply_entry_registers(ctx, config)
    apply_entry_vector_state(ctx, config)

    for snapshot in config.entry_memory_snapshots:
        ctx.setConcreteMemoryAreaValue(snapshot.base, snapshot.bytes)

    stack_bytes = config.stack_bytes if config.stack_bytes is not None else b"\x00" * config.stack_size
    if len(stack_bytes) != config.stack_size:
        raise ValueError("入口栈快照长度与配置不一致")
    ctx.setConcreteMemoryAreaValue(config.stack_base, stack_bytes)
    stack_top = config.entry_registers.rsp if config.entry_registers is not None else config.stack_base + config.stack_size - 0x20
    ctx.setConcreteMemoryAreaValue(stack_top, config.return_address.to_bytes(8, byteorder="little"))

    if config.vm_context_region is not None:
        if config.vm_context_bytes is None:
            raise ValueError("VM 上下文快照缺失，无法初始化 Triton 上下文")
        if len(config.vm_context_bytes) != config.vm_context_region.size:
            raise ValueError("VM 上下文快照长度与配置不一致")
        ctx.setConcreteMemoryAreaValue(config.vm_context_region.base, config.vm_context_bytes)

    for snapshot in config.extra_memory_snapshots:
        ctx.setConcreteMemoryAreaValue(snapshot.base, snapshot.bytes)

    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.RCX), config.plaintext_value)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.RDX), config.key_value)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.RIP), config.entry_address)

    # 入口参数直接作为符号化污点源，返回值公式只保留对输入寄存器的依赖。
    ctx.symbolizeRegister(make_register(ctx, REG.X86_64.RCX), "plaintext")
    ctx.symbolizeRegister(make_register(ctx, REG.X86_64.RDX), "key")
    ctx.taintRegister(make_register(ctx, REG.X86_64.RCX))
    ctx.taintRegister(make_register(ctx, REG.X86_64.RDX))

    return ctx


def replay_trace(
    ctx: TritonContext,
    steps: tuple[TraceStep, ...],
    observer: StepObserver | None = None,
    state_validator: Callable[[TritonContext, TraceStep], None] | None = None,
) -> None:
    for index, step in enumerate(steps):
        if state_validator is not None:
            state_validator(ctx, step)

        instruction = Instruction()
        instruction.setAddress(step.address)
        instruction.setOpcode(step.opcode)

        status = ctx.processing(instruction)
        if status != EXCEPTION.NO_FAULT:
            raise RuntimeError(f"Triton 处理失败: step={step.index}, status={status}, addr={hex(step.address)}")

        if index + 1 < len(steps):
            expected_next_rip = steps[index + 1].address
            actual_next_rip = ctx.getConcreteRegisterValue(make_register(ctx, REG.X86_64.RIP), False)
            if actual_next_rip != expected_next_rip:
                raise ReplayStateMismatch(
                    f"step={step.index} next_rip 期望={expected_next_rip:#x} 实际={actual_next_rip:#x}"
                )

        if observer is not None:
            observer(step, instruction, ctx)
