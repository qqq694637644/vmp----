from __future__ import annotations

from typing import Callable

from triton import ARCH, EXCEPTION, Instruction, MemoryAccess, REG, TritonContext

from .models import RecoveryConfig, TraceStep


StepObserver = Callable[[TraceStep, Instruction, TritonContext], None]


def make_register(ctx: TritonContext, reg_const: int):
    return ctx.getRegister(reg_const)


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
    ):
        ctx.setConcreteRegisterValue(make_register(ctx, reg_const), 0)


def seed_buffer(ctx: TritonContext, base_address: int, data: bytes, symbol_prefix: str) -> None:
    # 先写 concrete，再创建符号变量，最后打污点。
    ctx.setConcreteMemoryAreaValue(base_address, data)
    for offset in range(len(data)):
        memory = MemoryAccess(base_address + offset, 1)
        ctx.symbolizeMemory(memory, f"{symbol_prefix}_{offset}")
        ctx.taintMemory(memory)


def initialize_context(config: RecoveryConfig) -> TritonContext:
    ctx = TritonContext()
    ctx.setArchitecture(ARCH.X86_64)
    zero_general_registers(ctx)

    ctx.setConcreteMemoryAreaValue(config.stack_base, b"\x00" * config.stack_size)
    stack_top = config.stack_base + config.stack_size - 0x20
    ctx.setConcreteMemoryAreaValue(stack_top, config.return_address.to_bytes(8, byteorder="little"))

    seed_buffer(ctx, config.plaintext_base, config.plaintext, "plaintext")
    seed_buffer(ctx, config.key_base, config.key, "key")
    ctx.setConcreteMemoryAreaValue(config.output_base, b"\x00" * config.output_size)

    if config.context_region is not None:
        ctx.setConcreteMemoryAreaValue(config.context_region.base, b"\x00" * config.context_region.size)

    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.RCX), config.plaintext_base)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.RDX), config.key_base)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.R8), config.output_base)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.R9), config.output_size)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.RSP), stack_top)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.RBP), stack_top)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.RIP), config.entry_address)

    # 这里只把输入参数寄存器当作符号源，避免把输出指针和长度也污染进去。
    # 之前把所有参数都打污点会把大量内部地址运算误判成输出数据流。
    for reg_const, alias in (
        (REG.X86_64.RCX, "arg_plaintext_ptr"),
        (REG.X86_64.RDX, "arg_key_ptr"),
    ):
        register = make_register(ctx, reg_const)
        ctx.symbolizeRegister(register, alias)
        ctx.taintRegister(register)

    return ctx


def replay_trace(ctx: TritonContext, steps: tuple[TraceStep, ...], observer: StepObserver | None = None) -> None:
    for step in steps:
        instruction = Instruction()
        instruction.setAddress(step.address)
        instruction.setOpcode(step.opcode)

        status = ctx.processing(instruction)
        if status != EXCEPTION.NO_FAULT:
            raise RuntimeError(f"Triton 处理失败: step={step.index}, status={status}, addr={hex(step.address)}")

        if observer is not None:
            observer(step, instruction, ctx)
