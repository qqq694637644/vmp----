#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from triton import AST_NODE, ARCH, EXCEPTION, Instruction, MemoryAccess, REG, TritonContext


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
    # 这里只保留参数寄存器和栈指针，其他通用寄存器统一清零，避免 replay 时被脏状态污染。
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
    # 先放 concrete 值，再打符号和污点，避免把 symbolic state 搞乱。
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
    ctx = TritonContext()
    ctx.setArchitecture(ARCH.X86_64)
    zero_general_registers(ctx)

    stack_size = 0x2000
    stack_top = stack_base + stack_size - 0x20
    ctx.setConcreteMemoryAreaValue(stack_base, b"\x00" * stack_size)
    ctx.setConcreteMemoryAreaValue(stack_top, return_address.to_bytes(8, byteorder="little"))

    symbolize_and_taint_buffer(ctx, plaintext_base, plaintext, "plaintext")
    symbolize_and_taint_buffer(ctx, key_base, key, "key")
    ctx.setConcreteMemoryAreaValue(output_base, b"\x00" * len(plaintext))

    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.RCX), plaintext_base)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.RDX), key_base)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.R8), output_base)
    ctx.setConcreteRegisterValue(make_register(ctx, REG.X86_64.R9), len(plaintext))
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


def parse_reference_id(node_text: str) -> int:
    match = re.fullmatch(r"ref!(\d+)", node_text)
    if match is None:
        raise ValueError(f"非法引用节点: {node_text}")
    return int(match.group(1))


def simplify_ast_node(node, ctx, ast_ctx, memo, visiting):
    node_type = node.getType()

    if node_type in (AST_NODE.VARIABLE, AST_NODE.BV, AST_NODE.INTEGER, AST_NODE.STRING):
        return node

    if node_type == AST_NODE.REFERENCE:
        ref_id = parse_reference_id(str(node))
        if ref_id in memo:
            return memo[ref_id]
        if ref_id in visiting:
            raise RuntimeError(f"符号表达式递归引用: ref!{ref_id}")
        visiting.add(ref_id)
        try:
            if not ctx.isSymbolicExpressionExists(ref_id):
                raise KeyError(f"找不到符号表达式: ref!{ref_id}")
            simplified = simplify_ast_node(ctx.getSymbolicExpression(ref_id).getAst(), ctx, ast_ctx, memo, visiting)
            memo[ref_id] = simplified
            return simplified
        finally:
            visiting.remove(ref_id)

    children = [simplify_ast_node(child, ctx, ast_ctx, memo, visiting) for child in node.getChildren()]

    if node_type == AST_NODE.ZX:
        extension_size = int(str(node.getChildren()[0]))
        child = children[1]
        if child.getType() == AST_NODE.ZX:
            inner = child.getChildren()[1]
            inner_extension = child.getBitvectorSize() - inner.getBitvectorSize()
            return simplify_ast_node(ast_ctx.zx(extension_size + inner_extension, inner), ctx, ast_ctx, memo, visiting)
        return ast_ctx.zx(extension_size, child)

    if node_type == AST_NODE.SX:
        extension_size = int(str(node.getChildren()[0]))
        return ast_ctx.sx(extension_size, children[1])

    if node_type == AST_NODE.EXTRACT:
        high = int(str(node.getChildren()[0]))
        low = int(str(node.getChildren()[1]))
        child = children[2]
        if child.getType() == AST_NODE.ZX:
            inner = child.getChildren()[1]
            inner_size = inner.getBitvectorSize()
            width = high - low + 1
            if low == 0:
                if width > inner_size:
                    return simplify_ast_node(ast_ctx.zx(width - inner_size, inner), ctx, ast_ctx, memo, visiting)
                if width == inner_size:
                    return inner
            if high < inner_size:
                return simplify_ast_node(ast_ctx.extract(high, low, inner), ctx, ast_ctx, memo, visiting)
        return ast_ctx.extract(high, low, child)

    if node_type == AST_NODE.CONCAT:
        return ast_ctx.concat(children)

    if node_type == AST_NODE.BVXOR:
        left, right = children
        if left.getType() == AST_NODE.ZX and right.getType() == AST_NODE.ZX:
            left_inner = left.getChildren()[1]
            right_inner = right.getChildren()[1]
            left_extension = left.getBitvectorSize() - left_inner.getBitvectorSize()
            right_extension = right.getBitvectorSize() - right_inner.getBitvectorSize()
            if left_extension == right_extension:
                inner = ast_ctx.bvxor(left_inner, right_inner)
                return simplify_ast_node(ast_ctx.zx(left_extension, inner), ctx, ast_ctx, memo, visiting)
        return ast_ctx.bvxor(left, right)

    if node_type == AST_NODE.BVADD:
        return ast_ctx.bvadd(children[0], children[1])

    if node_type == AST_NODE.BVAND:
        return ast_ctx.bvand(children[0], children[1])

    if node_type == AST_NODE.BVOR:
        return ast_ctx.bvor(children[0], children[1])

    if node_type == AST_NODE.BVSUB:
        return ast_ctx.bvsub(children[0], children[1])

    if node_type == AST_NODE.BVNOT:
        return ast_ctx.bvnot(children[0])

    if node_type == AST_NODE.BVNEG:
        return ast_ctx.bvneg(children[0])

    if node_type == AST_NODE.BVNAND:
        return ast_ctx.bvnand(children[0], children[1])

    if node_type == AST_NODE.BVNOR:
        return ast_ctx.bvnor(children[0], children[1])

    if node_type == AST_NODE.BVXNOR:
        return ast_ctx.bvxnor(children[0], children[1])

    if node_type == AST_NODE.BVSHL:
        return ast_ctx.bvshl(children[0], children[1])

    if node_type == AST_NODE.BVLSHR:
        return ast_ctx.bvlshr(children[0], children[1])

    if node_type == AST_NODE.BVASHR:
        return ast_ctx.bvashr(children[0], children[1])

    if node_type == AST_NODE.BVROL:
        return ast_ctx.bvrol(children[0], children[1])

    if node_type == AST_NODE.BVROR:
        return ast_ctx.bvror(children[0], children[1])

    if node_type == AST_NODE.BVUDIV:
        return ast_ctx.bvudiv(children[0], children[1])

    if node_type == AST_NODE.BVSDIV:
        return ast_ctx.bvsdiv(children[0], children[1])

    if node_type == AST_NODE.BVUREM:
        return ast_ctx.bvurem(children[0], children[1])

    if node_type == AST_NODE.BVSREM:
        return ast_ctx.bvsrem(children[0], children[1])

    if node_type == AST_NODE.BVSMOD:
        return ast_ctx.bvsmod(children[0], children[1])

    if node_type == AST_NODE.BVULT:
        return ast_ctx.bvult(children[0], children[1])

    if node_type == AST_NODE.BVULE:
        return ast_ctx.bvule(children[0], children[1])

    if node_type == AST_NODE.BVUGT:
        return ast_ctx.bvugt(children[0], children[1])

    if node_type == AST_NODE.BVUGE:
        return ast_ctx.bvuge(children[0], children[1])

    if node_type == AST_NODE.BVSLT:
        return ast_ctx.bvslt(children[0], children[1])

    if node_type == AST_NODE.BVSLE:
        return ast_ctx.bvsle(children[0], children[1])

    if node_type == AST_NODE.BVSGT:
        return ast_ctx.bvsgt(children[0], children[1])

    if node_type == AST_NODE.BVSGE:
        return ast_ctx.bvsge(children[0], children[1])

    if node_type == AST_NODE.EQUAL:
        return ast_ctx.equal(children[0], children[1])

    if node_type == AST_NODE.DISTINCT:
        return ast_ctx.distinct(children[0], children[1])

    if node_type == AST_NODE.ITE:
        return ast_ctx.ite(children[0], children[1], children[2])

    if node_type == AST_NODE.SELECT:
        return ast_ctx.select(children[0], children[1])

    if node_type == AST_NODE.STORE:
        return ast_ctx.store(children[0], children[1], children[2])

    raise NotImplementedError(f"暂不支持的 AST 节点: type={node_type}, node={node}")


def simplify_symbolic_expression(expression, ctx, ast_ctx):
    memo: dict[int, object] = {}
    visiting: set[int] = set()
    return simplify_ast_node(expression.getAst(), ctx, ast_ctx, memo, visiting)


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

    tainted_steps, output_exprs = replay_trace(ctx, steps, output_base, len(plaintext))
    ast_ctx = ctx.getAstContext()
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

        slice_exprs = ctx.sliceExpressions(expr)
        formula = simplify_symbolic_expression(expr, ctx, ast_ctx)
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
