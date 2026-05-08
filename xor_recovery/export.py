"""算法导出。

这一层不参与污点分析，也不参与符号回放，只把 Triton 已经恢复出来的表达式转换成三种文本：
1. 简化后的 AST，保留严格语义，适合做精确检查。
2. LLVM IR，展示 Triton 的抬升结果和 LLVM 优化后的中间表示。
3. 人能读懂的算法，直接把 AST 渲染成表达式树，避免 LLVM 线性化后过度膨胀。

这里不尝试做完整反编译，因为 Triton 本身提供的是 AST / LLVM IR / solver 语义，而不是源码级恢复器。
"""

from __future__ import annotations

import contextlib
import re

from triton import AST_REPRESENTATION, TritonContext

from .models import RecoveredAlgorithm


@contextlib.contextmanager
def temporary_ast_representation(ctx: TritonContext, mode: AST_REPRESENTATION):
    """临时切换 AST 文本表示，结束后恢复原状。"""
    previous_mode = ctx.getAstRepresentationMode()
    ctx.setAstRepresentationMode(mode)
    try:
        yield
    finally:
        ctx.setAstRepresentationMode(previous_mode)


def format_bitvector_constant(value: int, bit_width: int) -> str:
    """把常量格式化成固定宽度的十六进制字面量。"""
    mask = (1 << bit_width) - 1
    width = max(1, bit_width // 4)
    return f"0x{(value & mask):0{width}X}"


def render_simplified_ast_text(ctx: TritonContext, ast_node) -> str:
    """导出简化后的 AST 文本。

    这里保留 Triton 自己的 AST 语义，不再做人为降级。
    """
    with temporary_ast_representation(ctx, AST_REPRESENTATION.SMT):
        return str(ast_node)


def render_llvm_ir(ctx: TritonContext, expr_node, function_name: str, optimize: bool = True) -> str:
    """把 AST 抬升成 LLVM IR。"""
    return ctx.liftToLLVM(expr_node, function_name, optimize)


def _ast_type_name(node) -> str:
    node_type = node.getType()
    return getattr(node_type, "name", str(node_type))


def _render_unknown_ast(ctx: TritonContext, node) -> str:
    """对当前还没有单独格式化规则的节点，直接退回 Triton 的 PCODE 表示。"""
    with temporary_ast_representation(ctx, AST_REPRESENTATION.PCODE):
        return str(node)


def _split_top_level_arguments(argument_text: str) -> list[str]:
    """按顶层逗号切分函数参数。

    Triton 的 PCODE 文本里经常会出现嵌套括号，不能直接按逗号切。
    """
    arguments: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(argument_text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            arguments.append(argument_text[start:index].strip())
            start = index + 1
    arguments.append(argument_text[start:].strip())
    return arguments


def _normalize_rotation_call(text: str) -> str:
    """把 Triton PCODE 里的 `rol/ror` 归一化成更直观的 `rotl/rotr`。"""
    stripped = text.strip()
    match = re.fullmatch(r"(?P<name>rol|ror)\((?P<body>.*)\)", stripped)
    if match is None:
        return text

    function_name = match.group("name")
    arguments = _split_top_level_arguments(match.group("body"))
    if len(arguments) != 3:
        return text

    shift_text = arguments[1]
    width_text = arguments[2]
    try:
        shift_value = int(shift_text, 0)
        width_value = int(width_text, 0)
    except ValueError:
        return text

    normalized_name = "rotl" if function_name == "rol" else "rotr"
    return f"{normalized_name}{width_value}({arguments[0]}, {shift_value})"


def _render_symbolic_alias(token: str, input_names: tuple[str, ...]) -> str:
    """把 Triton 生成的符号变量名映射回更直观的输入名。

    当前样本只把两个入口参数符号化，因此这里优先按顺序把 SymVar_0 / SymVar_1
    映射成 plaintext / key；如果 Triton 已经保留了原始名字，就直接原样输出。
    """
    alias_map = {f"SymVar_{index}": name for index, name in enumerate(input_names)}
    alias_map.update({name: name for name in input_names})
    return alias_map.get(token, token)


def _render_ast_node(ctx: TritonContext, node, input_names: tuple[str, ...]) -> str:
    """把 AST 递归渲染成人类可读的表达式。"""
    node_type = _ast_type_name(node)

    if node_type == "INTEGER":
        return format_bitvector_constant(node.getInteger(), node.getBitvectorSize())

    if node_type == "VARIABLE":
        symbol_name = node.getSymbolicVariable().getName()
        return _render_symbolic_alias(symbol_name, input_names)

    if node_type == "REFERENCE":
        # 引用节点直接展开对应的符号表达式，这样导出的结果还是一棵表达式树。
        symbolic_expression = node.getSymbolicExpression()
        return _render_ast_node(ctx, symbolic_expression.getAst(), input_names)

    children = node.getChildren()

    if node_type == "BSWAP" and len(children) == 1:
        return f"bswap{node.getBitvectorSize()}({_render_ast_node(ctx, children[0], input_names)})"

    if node_type in {"BVROL", "BVROR"} and len(children) == 2:
        op_name = "rotl" if node_type == "BVROL" else "rotr"
        return (
            f"{op_name}{node.getBitvectorSize()}("
            f"{_render_ast_node(ctx, children[0], input_names)}, "
            f"{_render_ast_node(ctx, children[1], input_names)})"
        )

    binary_ops = {
        "BVADD": "+",
        "BVSUB": "-",
        "BVXOR": "^",
        "BVAND": "&",
        "BVOR": "|",
        "BVMUL": "*",
        "BVSHL": "<<",
        "BVLSHR": ">>",
        "BVASHR": ">>_s",
        "LAND": "&&",
        "LOR": "||",
        "LXOR": "^^",
    }
    if node_type in binary_ops and len(children) == 2:
        left = _render_ast_node(ctx, children[0], input_names)
        right = _render_ast_node(ctx, children[1], input_names)
        operator = binary_ops[node_type]
        if operator == ">>_s":
            return f"({left} >>_s {right})"
        return f"({left} {operator} {right})"

    unary_ops = {
        "BVNEG": "-",
        "BVNOT": "~",
        "LNOT": "!",
    }
    if node_type in unary_ops and len(children) == 1:
        operand = _render_ast_node(ctx, children[0], input_names)
        return f"({unary_ops[node_type]}{operand})"

    comparison_ops = {
        "EQUAL": "==",
        "DISTINCT": "!=",
        "BVUGT": ">",
        "BVUGE": ">=",
        "BVULT": "<",
        "BVULE": "<=",
        "BVSGT": ">",
        "BVSGE": ">=",
        "BVSLT": "<",
        "BVSLE": "<=",
    }
    if node_type in comparison_ops and len(children) == 2:
        left = _render_ast_node(ctx, children[0], input_names)
        right = _render_ast_node(ctx, children[1], input_names)
        return f"({left} {comparison_ops[node_type]} {right})"

    if node_type == "CONCAT" and children:
        rendered_children = ", ".join(_render_ast_node(ctx, child, input_names) for child in children)
        return f"concat({rendered_children})"

    if node_type in {"ZX", "SX"} and children:
        rendered_children = [child for child in children if _ast_type_name(child) != "INTEGER"]
        operand = _render_ast_node(ctx, rendered_children[0], input_names) if rendered_children else _render_unknown_ast(ctx, node)
        prefix = "zext" if node_type == "ZX" else "sext"
        return f"{prefix}{node.getBitvectorSize()}({operand})"

    if node_type == "EXTRACT" and children:
        integer_children = [child for child in children if _ast_type_name(child) == "INTEGER"]
        value_children = [child for child in children if _ast_type_name(child) != "INTEGER"]
        if len(integer_children) >= 2 and len(value_children) == 1:
            bounds = sorted(child.getInteger() for child in integer_children[:2])
            value = _render_ast_node(ctx, value_children[0], input_names)
            return f"{value}[{bounds[1]}:{bounds[0]}]"
        rendered_children = ", ".join(_render_ast_node(ctx, child, input_names) for child in children)
        return f"extract({rendered_children})"

    if node_type == "ITE" and len(children) == 3:
        condition = _render_ast_node(ctx, children[0], input_names)
        true_branch = _render_ast_node(ctx, children[1], input_names)
        false_branch = _render_ast_node(ctx, children[2], input_names)
        return f"({condition} ? {true_branch} : {false_branch})"

    # 对于当前样本以外的 AST 形态，不静默丢语义，直接退回 Triton 自带表示。
    return _render_unknown_ast(ctx, node)


def render_human_readable_algorithm(ctx: TritonContext, ast_node, input_names: tuple[str, ...]) -> str:
    """把恢复出的 AST 渲染成可读算法。"""
    rendered_expression = _normalize_rotation_call(_render_ast_node(ctx, ast_node, input_names))
    result_bits = ast_node.getBitvectorSize()
    result_prefix = f"u{result_bits}" if result_bits > 1 else "u1"
    header = f"// 输入: {', '.join(input_names)}"
    return "\n".join(
        [
            header,
            f"{result_prefix} result = {rendered_expression};",
            "return result;",
        ]
    )


def build_recovered_algorithm(
    ctx: TritonContext,
    result_name: str,
    ast_node,
    input_names: tuple[str, ...] = ("plaintext", "key"),
) -> RecoveredAlgorithm:
    """把同一个结果表达式导出成三种形式。"""
    # 这里只做求解器级简化，不走 LLVM 级别的重写。
    # LLVM 会把很多高层运算打散成位级 extract / concat，反而不利于导出人能看懂的算法。
    simplified_ast_node = ctx.simplify(ast_node, True, False)
    simplified_ast_text = render_simplified_ast_text(ctx, simplified_ast_node)
    llvm_function_name = f"recover_{result_name}".replace(":", "_")
    llvm_ir = render_llvm_ir(ctx, simplified_ast_node, llvm_function_name, True)
    # 人类可读算法直接从原始恢复 AST 渲染，保留高层语义，避免被简化器压成位级碎片。
    human_readable_text = render_human_readable_algorithm(ctx, ast_node, input_names)
    return RecoveredAlgorithm(
        result_name=result_name,
        simplified_ast_text=simplified_ast_text,
        llvm_ir=llvm_ir,
        human_readable_text=human_readable_text,
    )
