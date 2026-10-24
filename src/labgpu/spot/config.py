"""
Spot monitor configuration (SPEC 2.2). In the agent it comes from the spot plugin's etcd config
(`config/plugins/accelerator/gpu_spot_N/monitor/<section>/<key>`, all values strings); the
`labgpu-spot` CLI can also read the same sections from a TOML file.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Self

from ..paths import default_cuda_checkpoint
from ..sizes import MiB

# The agent's own default var-base-path plus our subdirectory; the plugins always pass the real
# `<var-base-path>/labgpu` of the agent they run in (SPEC 2.13).
DEFAULT_STATE_DIR = Path("./var/lib/backend.ai/labgpu")


@dataclass(frozen=True)
class ControllerConfig:
    poll_interval: float = 5.0
    state_dir: Path = DEFAULT_STATE_DIR
    # A restart picks up the saved judgement only if it is at most this old (SPEC 2.10).
    resume_max_age_seconds: float = 600.0


@dataclass(frozen=True)
class IdleConfig:
    idle_minutes: float = 10.0
    owner_util_threshold: int = 5
    owner_mem_delta_mib: int = 512
    owner_cpu_threshold: float = 0.5
    unclaimed_grace_seconds: float = 60.0
    ignored_processes: tuple[str, ...] = ("Xorg",)

    @property
    def idle_seconds(self) -> float:
        return self.idle_minutes * 60

    @property
    def owner_mem_delta(self) -> int:
        return self.owner_mem_delta_mib * MiB


@dataclass(frozen=True)
class ReclaimConfig:
    mem_reserve_mib: int = 2048

    @property
    def mem_reserve(self) -> int:
        return self.mem_reserve_mib * MiB


@dataclass(frozen=True)
class SpotConfig:
    """Moving and evicting spot sessions (SPEC 2.12)."""

    enabled: bool = True
    cuda_checkpoint: Path = field(default_factory=default_cuda_checkpoint)
    checkpoint_timeout_seconds: float = 60.0
    # After the out-of-memory error, a program still on the reclaimed GPU this long is parked off it.
    oom_grace_seconds: float = 60.0


@dataclass(frozen=True)
class PoolConfig:
    """
    Shared spot pool (SPEC 2.16). `enabled = false` (the default) keeps the behaviour of SPEC 2.12
    exactly: idle spot sessions hold GPU shares and the slot capacity is the number of GPUs.
    With it on, an idle spot session holds no GPU share (the GPU goes to whichever spot session
    starts using it first), spot slot capacity follows `overcommit_max_sessions`, and the pool
    figures are published in the status file. The CPU and RAM sizes are nominal in this version:
    nothing enforces them yet (SPEC 2.16.3).
    """

    enabled: bool = False
    cpu_cores: int = 8
    mem_gib: float = 32.0
    preset_name: str = ""  # a Backend.AI resource preset lives in the manager; the plugin cannot read it
    overcommit_max_sessions: int = 16  # 0: no limit
    mem_floor_gib: float = 8.0


@dataclass(frozen=True)
class Config:
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    idle: IdleConfig = field(default_factory=IdleConfig)
    reclaim: ReclaimConfig = field(default_factory=ReclaimConfig)
    spot: SpotConfig = field(default_factory=SpotConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)

    @classmethod
    def load(cls, path: Path | None) -> Self:
        if path is None:
            return cls()
        with path.open("rb") as f:
            return cls.from_dict(tomllib.load(f))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        unknown = set(raw) - {"controller", "idle", "reclaim", "spot", "pool"}
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        return cls(
            controller=_build(ControllerConfig, raw.get("controller", {})),
            idle=_build(IdleConfig, raw.get("idle", {})),
            reclaim=_build(ReclaimConfig, raw.get("reclaim", {})),
            spot=_build(SpotConfig, raw.get("spot", {})),
            pool=_build(PoolConfig, raw.get("pool", {})),
        )


def _coerce(kind: str, value: Any) -> Any:
    """etcd hands every value over as a string; TOML already has the right types."""
    if kind == "bool":
        return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes")
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "Path":
        return Path(value)
    if kind.startswith("tuple"):
        items = value.split(",") if isinstance(value, str) else value
        return tuple(str(v).strip() for v in items if str(v).strip())
    return str(value)


def _build[T](dc: type[T], values: dict[str, Any]) -> T:
    types = {f.name: str(f.type) for f in fields(dc)}  # type: ignore[arg-type]
    unknown = set(values) - set(types)
    if unknown:
        raise ValueError(f"unknown keys for {dc.__name__}: {sorted(unknown)}")
    return dc(**{k: _coerce(types[k], v) for k, v in values.items()})
