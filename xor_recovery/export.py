"""算法导出。

这一层不参与污点分析，也不参与符号回放，只把 Triton 已经恢复出来的返回值表达式转换成三种文本：
1. 结构化 AST，保留原始 SSA / 引用层级，便于检查是否被过早展开。
2. LLVM IR，直接由 Triton 抬升成参数化函数，作为可复用的算法中间表示。
3. 伪代码，把同一棵 AST 渲染成更接近源码的表达式，方便人工阅读。

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


def _render_ast_node(
    ctx: TritonContext,
    node,
    input_names: tuple[str, ...],
    expand_references: bool,
) -> str:
    """把 AST 递归渲染成人类可读的表达式。"""
    node_type = _ast_type_name(node)

    if node_type == "INTEGER":
        return format_bitvector_constant(node.getInteger(), node.getBitvectorSize())

    if node_type == "VARIABLE":
        symbol_name = node.getSymbolicVariable().getName()
        return _render_symbolic_alias(symbol_name, input_names)

    if node_type == "REFERENCE":
        symbolic_expression = node.getSymbolicExpression()
        if not expand_references:
            return f"ref!{symbolic_expression.getId()}"
        # 只有在确实需要展开时，才把引用递归展开为完整表达式。
        return _render_ast_node(ctx, symbolic_expression.getAst(), input_names, expand_references)

    children = node.getChildren()

    if node_type == "BSWAP" and len(children) == 1:
        return f"bswap{node.getBitvectorSize()}({_render_ast_node(ctx, children[0], input_names, expand_references)})"

    if node_type in {"BVROL", "BVROR"} and len(children) == 2:
        op_name = "rotl" if node_type == "BVROL" else "rotr"
        return (
            f"{op_name}{node.getBitvectorSize()}("
            f"{_render_ast_node(ctx, children[0], input_names, expand_references)}, "
            f"{_render_ast_node(ctx, children[1], input_names, expand_references)})"
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
        left = _render_ast_node(ctx, children[0], input_names, expand_references)
        right = _render_ast_node(ctx, children[1], input_names, expand_references)
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
        operand = _render_ast_node(ctx, children[0], input_names, expand_references)
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
        left = _render_ast_node(ctx, children[0], input_names, expand_references)
        right = _render_ast_node(ctx, children[1], input_names, expand_references)
        return f"({left} {comparison_ops[node_type]} {right})"

    if node_type == "CONCAT" and children:
        rendered_children = ", ".join(
            _render_ast_node(ctx, child, input_names, expand_references) for child in children
        )
        return f"concat({rendered_children})"

    if node_type in {"ZX", "SX"} and children:
        rendered_children = [child for child in children if _ast_type_name(child) != "INTEGER"]
        operand = (
            _render_ast_node(ctx, rendered_children[0], input_names, expand_references)
            if rendered_children
            else _render_unknown_ast(ctx, node)
        )
        prefix = "zext" if node_type == "ZX" else "sext"
        return f"{prefix}{node.getBitvectorSize()}({operand})"

    if node_type == "EXTRACT" and children:
        integer_children = [child for child in children if _ast_type_name(child) == "INTEGER"]
        value_children = [child for child in children if _ast_type_name(child) != "INTEGER"]
        if len(integer_children) >= 2 and len(value_children) == 1:
            bounds = sorted(child.getInteger() for child in integer_children[:2])
            value = _render_ast_node(ctx, value_children[0], input_names, expand_references)
            return f"{value}[{bounds[1]}:{bounds[0]}]"
        rendered_children = ", ".join(
            _render_ast_node(ctx, child, input_names, expand_references) for child in children
        )
        return f"extract({rendered_children})"

    if node_type == "ITE" and len(children) == 3:
        condition = _render_ast_node(ctx, children[0], input_names, expand_references)
        true_branch = _render_ast_node(ctx, children[1], input_names, expand_references)
        false_branch = _render_ast_node(ctx, children[2], input_names, expand_references)
        return f"({condition} ? {true_branch} : {false_branch})"

    # 对于当前样本以外的 AST 形态，不静默丢语义，直接退回 Triton 自带表示。
    return _render_unknown_ast(ctx, node)


def render_human_readable_algorithm(ctx: TritonContext, ast_node, input_names: tuple[str, ...]) -> str:
    """把恢复出的 AST 渲染成可读算法。"""
    rendered_expression = _normalize_rotation_call(
        _render_ast_node(ctx, ast_node, input_names, expand_references=False)
    )
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
    # 这里只做结构保持，不启用求解器级展开。
    # 过早调用 solver 简化会把原始 SSA / 引用层级炸成大表达式，反而不利于后续导出。
    simplified_ast_node = ctx.simplify(ast_node, False, False)
    simplified_ast_text = render_simplified_ast_text(ctx, simplified_ast_node)
    llvm_function_name = f"recover_{result_name}".replace(":", "_")
    # LLVM 这层直接抬升原始返回值表达式，让参数从符号变量里自然保留下来。
    llvm_ir = render_llvm_ir(ctx, ast_node, llvm_function_name, False)
    # 伪代码和 LLVM IR 必须来自同一棵表达式树，否则会出现“一个是常量、一个是变量”的错位。
    human_readable_text = render_human_readable_algorithm(ctx, ast_node, input_names)
    return RecoveredAlgorithm(
        result_name=result_name,
        simplified_ast_text=simplified_ast_text,
        llvm_ir=llvm_ir,
        human_readable_text=human_readable_text,
    )
