"""Plain data exchanged between the observer, detector, planner, and executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ProcKind(StrEnum):
    OWNER = "owner"
    SPOT = "spot"
    UNKNOWN = "unknown"
    IGNORED = "ignored"  # host process on the ignore list, e.g. the display server (SPEC 2.1)


@dataclass(frozen=True)
class ClassifiedProcess:
    pid: int
    kind: ProcKind
    container_id: str | None
    used_memory: int
    sm_util: int


@dataclass(frozen=True)
class GpuObservation:
    """Everything the detector needs about one GPU for one tick. `ok=False` means UNKNOWN."""

    uuid: str
    index: int
    ok: bool
    total_memory: int = 0
    used_memory: int = 0
    processes: tuple[ClassifiedProcess, ...] = ()
    owner_containers: frozenset[str] = frozenset()
    owner_cpu_cores: float | None = None  # None: unmeasurable, skip the CPU rule
    error: str | None = None
    model: str = ""

    @property
    def free_memory(self) -> int:
        return self.total_memory - self.used_memory

    @property
    def owner_util(self) -> int:
        return sum(p.sm_util for p in self.processes if p.kind is ProcKind.OWNER)

    @property
    def owner_memory(self) -> int:
        return sum(p.used_memory for p in self.processes if p.kind is ProcKind.OWNER)

    @property
    def has_unknown(self) -> bool:
        return any(p.kind is ProcKind.UNKNOWN for p in self.processes)


class GpuState(StrEnum):
    BUSY = "BUSY"
    IDLE = "IDLE"
    LENDABLE = "LENDABLE"
    LENT = "LENT"
    RECLAIMING = "RECLAIMING"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class GpuVerdict:
    """The detector's per-tick output for one GPU."""

    uuid: str
    state: GpuState
    lendable: bool
    must_reclaim: bool
    reasons: tuple[str, ...] = ()
    lendable_memory: int = 0  # bytes a new spot job may use
    idle_for: float = 0.0
    model: str = ""


@dataclass(frozen=True)
class RunningSpot:
    job_id: int
    gpu_uuid: str
    container: str
    reclaiming: bool = False


@dataclass(frozen=True)
class QueuedJob:
    job_id: int
    priority: int
    submitted_at: float
    gpu_mem: int
    ram: int
    gpu_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class Launch:
    job_id: int
    gpu_uuid: str
    gpu_mem_limit: int
    ram: int


@dataclass(frozen=True)
class Reclaim:
    job_id: int
    gpu_uuid: str
    container: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
