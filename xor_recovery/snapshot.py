from __future__ import annotations


MINIMAL_SNAPSHOT_ITEMS: tuple[str, ...] = (
    "入口整数寄存器：RAX-R15、RSP、RIP、EFLAGS",
    "入口向量状态：MXCSR、MXCSR_MASK、XMM0-XMM15、YMM0-YMM15 高位",
    "入口栈快照：覆盖当前栈帧、返回地址和局部变量",
    "VM 上下文：函数内部虚拟机状态块、状态槽、表驱动区",
    "参数指针指向的内存：只要参数是指针，就必须一起截取它指向的缓冲区",
    "返回值：RAX，作为函数最终输出锚点",
)


def get_minimal_snapshot_items() -> tuple[str, ...]:
    return MINIMAL_SNAPSHOT_ITEMS
