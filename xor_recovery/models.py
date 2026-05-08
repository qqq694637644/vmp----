from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TraceStep:
    index: int
    address: int
    opcode: bytes
    line_number: int | None


@dataclass(frozen=True)
class EntryArguments:
    plaintext_value: int
    key_value: int
    plaintext: bytes
    key: bytes


@dataclass(frozen=True)
class EntryRegisters:
    rax: int
    rbx: int
    rcx: int
    rdx: int
    rsi: int
    rdi: int
    rbp: int
    rsp: int
    r8: int
    r9: int
    r10: int
    r11: int
    r12: int
    r13: int
    r14: int
    r15: int
    eflags: int


@dataclass(frozen=True)
class EntryVectorState:
    mxcsr: int
    mxcsr_mask: int
    xmm_registers: tuple[bytes, ...]


@dataclass(frozen=True)
class TraceMetadata:
    entry_address: int
    function_size: int
    steps: tuple[TraceStep, ...]
    entry_arguments: EntryArguments | None = None
    entry_registers: EntryRegisters | None = None
    entry_vector_state: EntryVectorState | None = None
    stack_pointer: int | None = None
    return_address: int | None = None
    result_value: int | None = None
    result_bytes: bytes | None = None
    stack_bytes: bytes | None = None

    def __iter__(self):
        yield self.entry_address
        yield self.function_size
        yield self.steps


@dataclass(frozen=True)
class MemoryRegion:
    name: str
    base: int
    size: int

    def contains(self, address: int) -> bool:
        return self.base <= address < self.base + self.size

    def describe(self, address: int) -> str:
        return f"{self.name}+0x{address - self.base:X}"


@dataclass(frozen=True)
class RecoveryConfig:
    plaintext_value: int
    key_value: int
    entry_address: int
    stack_base: int
    return_address: int
    operand_size: int = 4
    entry_registers: EntryRegisters | None = None
    entry_vector_state: EntryVectorState | None = None
    stack_bytes: bytes | None = None
    context_region: MemoryRegion | None = None
    stack_size: int = 0x2000

    @property
    def result_size(self) -> int:
        return self.operand_size

    def tracked_regions(self) -> tuple[MemoryRegion, ...]:
        regions = [MemoryRegion("stack", self.stack_base, self.stack_size)]
        if self.context_region is not None:
            regions.append(self.context_region)
        return tuple(regions)


@dataclass(frozen=True)
class DependencyNode:
    expr_id: int
    step_index: int | None
    origin: str
    ast: str


@dataclass(frozen=True)
class TaintAnalysisResult:
    tainted_steps: tuple[TraceStep, ...]
    dependency_nodes: dict[int, DependencyNode]
    dependency_graph: dict[int, tuple[int, ...]]
    result_roots: dict[str, int]
    result_slices: dict[str, tuple[int, ...]]
    result_sizes: dict[str, int]
    result_value: int
    result_bytes: bytes
    tainted_memory: tuple[str, ...]
    tainted_registers: tuple[str, ...]
    context_hits: tuple[str, ...]


@dataclass(frozen=True)
class FormulaResult:
    result_name: str
    byte_offset: int
    expr_id: int
    slice_size: int
    formula_text: str
    evaluated_value: int
    concrete_value: int


@dataclass(frozen=True)
class RecoveryResult:
    trace_path: Path
    entry_address: int
    function_size: int
    taint: TaintAnalysisResult
    formulas: tuple[FormulaResult, ...]
