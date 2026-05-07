#!/usr/bin/env python3
from __future__ import annotations

"""从 trace 中还原 VMP 保护函数的输入输出公式。

这份脚本的核心思路不是“暴力枚举所有符号表达式”，而是：
1. 先把目标函数的指令 trace 重放到 Triton 里；
2. 只从 output 内存出发做依赖切片，找到真正影响输出的那条符号链；
3. 递归展开 `ref!N` 引用，直到回到 plaintext / key 这类输入符号；
4. 再把 zero-extend、extract 这类中间包装折叠掉，恢复出人能读懂的公式；
5. 最后用 Triton 的求值接口把公式结果和 concrete 输出逐字节对拍。

这样做的目的，是把 VMP 里的虚拟指令、上下文搬运、临时寄存器噪声全部过滤掉，
只保留真正的算法语义。
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from triton import ARCH, EXCEPTION, Instruction, MemoryAccess, REG, TritonContext


ENTRY_RE = re.compile(r"已定位 XorTransform，地址=(0x[0-9A-Fa-f]+)，大小=(\d+)")
STEP_RE = re.compile(
    r"^步骤\s+(\d+)\s+\|\s+RIP=(0x[0-9A-Fa-f]+)\s+\|\s+字节=([0-9A-Fa-f ]+)(?:\s+\|\s+行号=(\d+))?$"
)
EXIT_RE = re.compile(r"已离开 XorTransform，步骤数=(\d+)")


@dataclass(frozen=True)
class TraceStep:
    index: int
    address: int
    opcode: bytes
    line_number: int | None


def parse_hex_bytes(text: str) -> bytes:
    return bytes.fromhex(text.replace(" ", ""))


def parse_trace(trace_path: Path) -> tuple[int, int, list[TraceStep]]:
    # 解析 tracer 输出，拿到三类关键信息：
    # 1) 函数入口地址和函数大小，用来定位要还原的目标函数；
    # 2) 每一步指令的地址和字节，用来在 Triton 中逐条重放；
    # 3) 函数退出信息，用来知道 trace 到哪里结束。
    entry_address = 0
    function_size = 0
    steps: list[TraceStep] = []

    for raw_line in trace_path.read_text(encoding="utf-8").splitlines():
        entry_match = ENTRY_RE.search(raw_line)
        if entry_match is not None:
            entry_address = int(entry_match.group(1), 16)
            function_size = int(entry_match.group(2))
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

        exit_match = EXIT_RE.search(raw_line)
        if exit_match is not None:
            break

    if entry_address == 0 or function_size == 0:
        raise ValueError("轨迹里没有找到 XorTransform 入口信息")
    if not steps:
        raise ValueError("轨迹里没有找到可重放的步骤")

    return entry_address, function_size, steps


def make_register(ctx: TritonContext, reg_const: int):
    return ctx.getRegister(reg_const)


def zero_general_registers(ctx: TritonContext) -> None:
    # 还原时我们不依赖宿主进程的真实寄存器状态，所以先把通用寄存器全部清零。
    # 这样做的原因是：trace 只记录了目标函数内部指令，没有完整记录调用者的上下文。
    # 如果不清零，Triton 会把“上一次残留的寄存器值”当成输入，导致公式里混入脏状态。
    # RCX/RDX/R8/R9 之后会被我们手工写入，作为 Windows x64 调用约定下的参数寄存器。
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


def symbolize_and_taint_buffer(ctx: TritonContext, base_address: int, data: bytes, alias_prefix: str) -> None:
    # 这里先把缓冲区写成 concrete，再为每个字节创建符号变量并打污点。
    #
    # 顺序不能反过来：
    # - 先 concrete：告诉 Triton 这个地址当前真实存着什么；
    # - 再 symbolize：把每个字节变成可追踪的输入变量；
    # - 再 taint：标记这些字节来自“受关注输入”，后面就能快速看出谁受它影响。
    #
    # 这样做的结果是，plaintext / key 的每个字节都会形成独立的 symbolic leaf，
    # 后续公式里出现 `plaintext_0`、`key_0` 这种名字时，就能直接对应回输入。
    ctx.setConcreteMemoryAreaValue(base_address, data)
    for offset in range(len(data)):
        mem = MemoryAccess(base_address + offset, 1)
        ctx.symbolizeMemory(mem, f"{alias_prefix}_{offset}")
        ctx.taintMemory(mem)


def initialize_context(
    entry_address: int,
    plaintext: bytes,
    key: bytes,
    stack_base: int,
    plaintext_base: int,
    key_base: int,
    output_base: int,
    return_address: int,
) -> TritonContext:
    # 初始化一个“干净的” Triton 上下文。
    #
    # 这里不是在模拟整个进程，而是只搭一个足够还原目标函数的最小执行环境：
    # - fake stack：满足函数 prologue / epilogue 对栈的读写；
    # - 参数寄存器：把 plaintext / key / output / length 放到 Windows x64 ABI 对应位置；
    # - return address：让函数执行到 ret 时有地方返回，trace 才能自然结束。
    ctx = TritonContext()
    ctx.setArchitecture(ARCH.X86_64)
    zero_general_registers(ctx)

    # 栈空间不需要很大，只要覆盖这次函数调用过程中会访问到的范围就够了。
    stack_size = 0x2000
    stack_top = stack_base + stack_size - 0x20
    ctx.setConcreteMemoryAreaValue(stack_base, b"\x00" * stack_size)
    ctx.setConcreteMemoryAreaValue(stack_top, return_address.to_bytes(8, byteorder="little"))

    # 输入缓冲区先变成 symbolic + tainted。
    # 这一步的意义是：后面只要算法真的读取了这些字节，Triton 就能把依赖一路追出来。
    symbolize_and_taint_buffer(ctx, plaintext_base, plaintext, "plaintext")
    symbolize_and_taint_buffer(ctx, key_base, key, "key")
    # 输出缓冲区初始化为全 0。这样一来，只有在函数内部真正写出去的值才会出现在最终结果里。
    ctx.setConcreteMemoryAreaValue(output_base, b"\x00" * len(plaintext))

    # 按 Windows x64 调用约定布置参数寄存器：
    # RCX = plaintext
    # RDX = key
    # R8  = output
    # R9  = length
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.RCX), plaintext_base)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.RDX), key_base)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.R8), output_base)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.R9), len(plaintext))
    # 栈指针和基指针也要摆好，否则函数里的局部变量访问会失真。
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.RSP), stack_top)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.RBP), stack_top)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.RIP), entry_address)

    return ctx


def replay_trace(
    ctx: TritonContext,
    steps: list[TraceStep],
    output_base: int,
    output_size: int,
) -> tuple[list[TraceStep], dict[int, object]]:
    # 这里是“重放 trace”的核心。
    # 每一步做的事情都很简单：
    # 1. 用 trace 里的地址和字节构造一条 Triton Instruction；
    # 2. 让 Triton 在当前 concrete/symbolic 状态下执行这条指令；
    # 3. Triton 自动更新寄存器、内存和符号表达式；
    # 4. 只把“写到了 output 地址范围”的表达式收集起来。
    #
    # 注意：我们并不是在分析全部指令的语义，只保留和 output 相关的那部分。
    # 这就是为什么这个方案比“全量枚举所有符号表达式”要干净得多。
    tainted_steps: list[TraceStep] = []
    output_exprs: dict[int, object] = {}

    for step in steps:
        inst = Instruction()
        inst.setAddress(step.address)
        inst.setOpcode(step.opcode)

        status = ctx.processing(inst)
        if status != EXCEPTION.NO_FAULT:
            raise RuntimeError(f"Triton 处理失败: step={step.index}, status={status}, addr={hex(step.address)}")

        if inst.isTainted():
            tainted_steps.append(step)

        # 只抓 output 地址上的新符号表达式。
        # 这里得到的通常不是最终公式，而只是“某个内存写入点”的表达式引用，
        # 后面还要继续沿 ref!N 递归展开。
        for expr in inst.getSymbolicExpressions():
            origin = expr.getOrigin()
            origin_address = getattr(origin, "getAddress", None)
            if origin_address is None:
                continue
            addr = origin.getAddress()
            if output_base <= addr < output_base + output_size:
                output_exprs[addr] = expr

    return tainted_steps, output_exprs


def format_hex(value: int) -> str:
    return f"0x{value:016X}"


def render_symbolic_expression(expression, ctx, ast_ctx):
    # 这里不自己重写 AST 简化器，直接用 Triton 内置的 `unroll` 展开引用，
    # 再交给 Triton 自带的 simplify 处理。
    #
    # 这条链的职责很明确：
    # - `unroll`：把 ref!N 展开成真正定义它的表达式；
    # - `simplify`：做 Triton 内置的结构化化简；
    # - `evaluateAstViaSolver`：只负责校验，不负责“推导公式”。
    unrolled = ast_ctx.unroll(expression.getAst())
    return ctx.simplify(unrolled)


def main() -> int:
    parser = argparse.ArgumentParser(description="从 trace 中还原 XorTransform 的输出公式。")
    parser.add_argument("trace_file", help="trace_xor.exe 的输出文件")
    parser.add_argument("--plaintext", default="1234", help="明文输入，默认 1234")
    parser.add_argument("--key", default="key!", help="密钥输入，默认 key!")
    parser.add_argument("--stack-base", default="0x70000000")
    parser.add_argument("--plaintext-base", default="0x10000000")
    parser.add_argument("--key-base", default="0x10001000")
    parser.add_argument("--output-base", default="0x10002000")
    args = parser.parse_args()

    plaintext = args.plaintext.encode("utf-8")
    key = args.key.encode("utf-8")
    if len(plaintext) != len(key):
        raise ValueError("plaintext 和 key 的长度必须一致")

    trace_path = Path(args.trace_file)
    entry_address, function_size, steps = parse_trace(trace_path)

    stack_base = int(args.stack_base, 16)
    plaintext_base = int(args.plaintext_base, 16)
    key_base = int(args.key_base, 16)
    output_base = int(args.output_base, 16)
    return_address = entry_address + function_size + 0x1000

    ctx = initialize_context(
        entry_address=entry_address,
        plaintext=plaintext,
        key=key,
        stack_base=stack_base,
        plaintext_base=plaintext_base,
        key_base=key_base,
        output_base=output_base,
        return_address=return_address,
    )

    # 先重放 trace，拿到“输出地址对应的符号表达式”。
    # 这一步结束后，我们只需要验证 output，而不是关心 trace 里的所有中间状态。
    tainted_steps, output_exprs = replay_trace(ctx, steps, output_base, len(plaintext))
    ast_ctx = ctx.getAstContext()
    # 这里的 concrete_output 是重放完成后 output 缓冲区里的真实字节。
    # 后面会拿它和符号公式求值结果做逐字节对拍。
    concrete_output = ctx.getConcreteMemoryAreaValue(output_base, len(plaintext))
    print(f"已读取轨迹: {trace_path}")
    print(f"函数入口: {format_hex(entry_address)}")
    print(f"函数大小: {function_size}")
    print(f"明文长度: {len(plaintext)}")
    print(f"污点步骤数: {len(tainted_steps)}")
    print(f"输出字节数: {len(output_exprs)}")
    print("污点步骤:")
    for step in tainted_steps:
        print(f"  #{step.index:06d} {format_hex(step.address)} {step.line_number or 0:>4}")

    print("输出公式:")
    verified_count = 0
    for offset in range(len(plaintext)):
        address = output_base + offset
        expr = output_exprs.get(address)
        if expr is None:
            print(f"  out[{offset}]@{format_hex(address)}: 未找到符号表达式")
            continue

        # sliceExpressions 只返回当前输出表达式真正依赖的那小撮符号表达式。
        # 这比遍历整个 symbolic table 更合理，也更符合“只看输出依赖链”的目标。
        slice_exprs = ctx.sliceExpressions(expr)
        # 这里的公式完全由 Triton 内置的 AST 展开和简化得到。
        formula = render_symbolic_expression(expr, ctx, ast_ctx)
        # 用 Triton 直接对公式求值，确认公式和当前 concrete 输入/输出是一致的。
        evaluated_value = ctx.evaluateAstViaSolver(formula)
        concrete_value = concrete_output[offset]
        if evaluated_value != concrete_value:
            raise RuntimeError(
                f"公式校验失败: out[{offset}] 公式值={evaluated_value:#x}，实际值={concrete_value:#x}"
            )
        verified_count += 1
        print(
            f"  out[{offset}]@{format_hex(address)}: {formula} "
            f"=> {evaluated_value:#04x} (slice={len(slice_exprs)})"
        )

    print(f"公式校验: {verified_count}/{len(plaintext)} 字节一致")
    print(f"实际输出: {concrete_output.hex(' ').upper()}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
