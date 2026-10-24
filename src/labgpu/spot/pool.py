"""
Shared spot pool settings and slot figures (SPEC 2.16). Pure: no NVML, Docker, or Backend.AI.

The pool is opt-in per node (`monitor/pool/enabled`). Values that cannot be trusted switch the
pool off rather than guessing (fail safe, SPEC 2.16.1).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import PoolConfig

# Slot capacity reported when `overcommit_max_sessions` is 0 (no limit). A spot session asks for
# at most one share, so this many sessions fit; the manager needs a number, not "unlimited".
UNLIMITED_SLOTS = 1024


@dataclass(frozen=True)
class Pool:
    """A validated pool: nominal CPU and RAM, session limit, and the memory floor."""

    cpu_cores: int
    mem_gib: float
    max_sessions: int  # 0: no limit
    mem_floor_gib: float

    @property
    def slots(self) -> int:
        """Spot slot capacity: one slot per allowed session (a session asks for at most one)."""
        return self.max_sessions or UNLIMITED_SLOTS


def resolve(
    cfg: PoolConfig, node_cores: int | None, node_mem_gib: float | None
) -> tuple[Pool | None, list[tuple[str, str]]]:
    """
    The pool to use and the log lines ((level, text), level "warning" or "error") explaining it.
    None means the pool is off: not enabled, or something is wrong (then there is an error line).
    """
    if not cfg.enabled:
        return None, []
    if cfg.preset_name:
        return None, [(
            "error",
            f"pool.preset_name={cfg.preset_name!r}: Backend.AI presets live in the manager and "
            "cannot be read here; set pool.cpu_cores and pool.mem_gib instead. Pool disabled.",
        )]
    if cfg.cpu_cores <= 0 or cfg.mem_gib <= 0 or cfg.overcommit_max_sessions < 0 or cfg.mem_floor_gib < 0:
        return None, [("error", f"pool settings out of range ({cfg}). Pool disabled.")]
    if node_cores is None or node_mem_gib is None or node_cores <= 0 or node_mem_gib <= 0:
        return None, [("error", "pool: cannot read the node's CPU count or RAM. Pool disabled.")]
    notes: list[tuple[str, str]] = []
    cores, mem = cfg.cpu_cores, cfg.mem_gib
    if cores > node_cores:
        notes.append(("warning", f"pool.cpu_cores={cores} is more than the node has; using {node_cores}."))
        cores = node_cores
    if mem > node_mem_gib:
        notes.append(("warning", f"pool.mem_gib={mem:g} is more than the node has; using {node_mem_gib:.1f}."))
        mem = node_mem_gib
    return Pool(cores, mem, cfg.overcommit_max_sessions, cfg.mem_floor_gib), notes


def memory_pressure(pool: Pool, mem_available_gib: float | None) -> bool:
    """Node MemAvailable below the floor. Unreadable counts as pressure (fail safe: no new GPU grants)."""
    return mem_available_gib is None or mem_available_gib < pool.mem_floor_gib
