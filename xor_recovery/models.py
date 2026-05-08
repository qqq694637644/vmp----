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
    plaintext: bytes
    key: bytes
    entry_address: int
    stack_base: int
    plaintext_base: int
    key_base: int
    output_base: int
    return_address: int
    context_region: MemoryRegion | None = None
    stack_size: int = 0x2000

    @property
    def output_size(self) -> int:
        return len(self.plaintext)

    def tracked_regions(self) -> tuple[MemoryRegion, ...]:
        regions = [
            MemoryRegion("plaintext", self.plaintext_base, len(self.plaintext)),
            MemoryRegion("key", self.key_base, len(self.key)),
            MemoryRegion("output", self.output_base, len(self.plaintext)),
            MemoryRegion("stack", self.stack_base, self.stack_size),
        ]
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
    output_roots: dict[int, int]
    output_slices: dict[int, tuple[int, ...]]
    output_sizes: dict[int, int]
    tainted_memory: tuple[str, ...]
    tainted_registers: tuple[str, ...]
    context_hits: tuple[str, ...]


@dataclass(frozen=True)
class FormulaResult:
    output_address: int
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
