from __future__ import annotations


MINIMAL_SNAPSHOT_ITEMS: tuple[str, ...] = (
    "入口整数寄存器：RAX-R15、RSP、RIP、EFLAGS",
    "入口向量寄存器：MXCSR、MXCSR_MASK、XMM0-XMM15",
    "入口栈快照：覆盖当前栈帧、返回地址和局部变量",
    "关键 VM context 内存：虚拟机上下文、状态槽、表驱动区",
    "参数指针指向的内存：只要参数是指针，就必须一起截取它指向的缓冲区",
    "外部副作用：API 返回值、异常处理结果、线程本地状态",
)


def get_minimal_snapshot_items() -> tuple[str, ...]:
    return MINIMAL_SNAPSHOT_ITEMS
