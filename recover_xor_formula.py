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


def parse_reference_id(node_text: str) -> int:
    # Triton 在 AST 里会用 `ref!123` 表示“引用第 123 号符号表达式”。
    # 这里把这个文本编号提取出来，后面才能顺着引用继续往回追。
    match = re.fullmatch(r"ref!(\d+)", node_text)
    if match is None:
        raise ValueError(f"非法引用节点: {node_text}")
    return int(match.group(1))


def simplify_ast_node(node, ctx, ast_ctx, memo, visiting):
    # 递归展开并折叠 AST。
    #
    # 这里的目标不是做完整 SMT 简化，而是把 Triton 生成的“中间包装”去掉：
    # - `ref!N`：追到真正定义这个值的符号表达式；
    # - `zx/sx/extract`：压掉多余的位宽扩展和截断；
    # - `bvxor`：尽量把多个输入字节的异或合并到一层。
    #
    # 如果遇到不支持的节点，直接报错。开发阶段不能悄悄吞掉，因为那会让公式看起来“能跑”，
    # 实际上却可能已经偏了。
    node_type = node.getType()

    # 叶子节点直接返回：
    # - VARIABLE：输入符号，比如 plaintext_0 / key_0；
    # - BV / INTEGER / STRING：常量或标量，不需要再展开。
    if node_type in (AST_NODE.VARIABLE, AST_NODE.BV, AST_NODE.INTEGER, AST_NODE.STRING):
        return node

    if node_type == AST_NODE.REFERENCE:
        # 关键点：这里不是枚举所有符号表达式，而是只沿当前节点引用的那个 ref 继续回溯。
        # 这样能把搜索范围从“全表扫描”缩成“输出依赖链”。
        ref_id = parse_reference_id(str(node))
        if ref_id in memo:
            return memo[ref_id]
        if ref_id in visiting:
            raise RuntimeError(f"符号表达式递归引用: ref!{ref_id}")
        # 用 ctx.isSymbolicExpressionExists / ctx.getSymbolicExpression 去取被引用的定义，
        # 再递归展开它自己的 AST。
        visiting.add(ref_id)
        try:
            if not ctx.isSymbolicExpressionExists(ref_id):
                raise KeyError(f"找不到符号表达式: ref!{ref_id}")
            simplified = simplify_ast_node(ctx.getSymbolicExpression(ref_id).getAst(), ctx, ast_ctx, memo, visiting)
            memo[ref_id] = simplified
            return simplified
        finally:
            visiting.remove(ref_id)

    # 先把所有子节点也递归展开，再根据当前节点类型做局部折叠。
    children = [simplify_ast_node(child, ctx, ast_ctx, memo, visiting) for child in node.getChildren()]

    if node_type == AST_NODE.ZX:
        # zero-extend 的常见情况是：一个字节被先扩展到 32/64 位，再参与运算。
        # 这里尽量把连续扩展压平，避免最后看到一串冗长的 `zero_extend`。
        extension_size = int(str(node.getChildren()[0]))
        child = children[1]
        if child.getType() == AST_NODE.ZX:
            inner = child.getChildren()[1]
            inner_extension = child.getBitvectorSize() - inner.getBitvectorSize()
            return simplify_ast_node(ast_ctx.zx(extension_size + inner_extension, inner), ctx, ast_ctx, memo, visiting)
        return ast_ctx.zx(extension_size, child)

    if node_type == AST_NODE.SX:
        # sign-extend 在这份示例里不多，但仍然保留统一处理。
        extension_size = int(str(node.getChildren()[0]))
        return ast_ctx.sx(extension_size, children[1])

    if node_type == AST_NODE.EXTRACT:
        # extract 常出现在“从更宽的中间值里切回 8 位结果”的地方。
        # 如果它只是从 zero-extend 的结果里取低位，这里就把包装去掉。
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
        # CONCAT 用来拼接字节序列，例如把 8 个字节拼成 64 位寄存器值。
        return ast_ctx.concat(children)

    if node_type == AST_NODE.BVXOR:
        # 这就是示例算法的核心：把 plaintext 和 key 做按字节异或。
        # 如果两边都只是同样宽度的 zero-extend 包装，就把包装剥掉后再合并。
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
        # 理论上输出路径里可能出现内存 store 节点；这里保留显式处理，避免漏掉写内存语义。
        return ast_ctx.store(children[0], children[1], children[2])

    # 开发阶段不做“兜底兼容”。
    # 如果 VMP 或样本换了 AST 形态，必须先暴露出来，再补对应节点处理。
    raise NotImplementedError(f"暂不支持的 AST 节点: type={node_type}, node={node}")


def simplify_symbolic_expression(expression, ctx, ast_ctx):
    # 对单个符号表达式做“引用展开 + 结构简化”。
    # 这里的输出才是最终给人看的公式。
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
        # 这里得到的是人能读懂的最终公式，比如：
        # (bvxor plaintext_0 key_0)
        formula = simplify_symbolic_expression(expr, ctx, ast_ctx)
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
