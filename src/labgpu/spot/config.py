"""Spot controller configuration (SPEC 2.2)."""

from __future__ import annotations

import shutil
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Self

from ..sizes import MiB

DEFAULT_CONFIG_PATH = Path("/etc/labgpu/spot.toml")
# Where scripts/install_cuda_checkpoint.sh puts the tool: the plugin checkout's own .venv/bin.
REPO_DIR = Path(__file__).resolve().parents[3]


def default_cuda_checkpoint() -> Path:
    """<plugin checkout>/.venv/bin/cuda-checkpoint, else whatever is on PATH (SPEC 2.2)."""
    local = REPO_DIR / ".venv" / "bin" / "cuda-checkpoint"
    if local.exists():
        return local
    found = shutil.which("cuda-checkpoint")
    return Path(found) if found else local


@dataclass(frozen=True)
class ControllerConfig:
    poll_interval: float = 5.0
    state_dir: Path = Path("/var/lib/labgpu")


@dataclass(frozen=True)
class IdleConfig:
    idle_minutes: float = 30.0
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
    park_seconds: float = 300.0
    evict_signal: str = "SIGINT"
    evict_grace_seconds: float = 30.0


@dataclass(frozen=True)
class Config:
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    idle: IdleConfig = field(default_factory=IdleConfig)
    reclaim: ReclaimConfig = field(default_factory=ReclaimConfig)
    spot: SpotConfig = field(default_factory=SpotConfig)

    @classmethod
    def load(cls, path: Path | None) -> Self:
        if path is None or not path.exists():
            if path is not None and path != DEFAULT_CONFIG_PATH:
                raise FileNotFoundError(path)
            return cls()
        with path.open("rb") as f:
            return cls.from_dict(tomllib.load(f))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        unknown = set(raw) - {"controller", "idle", "reclaim", "spot"}
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        c = {k: Path(v) if k == "state_dir" else v for k, v in raw.get("controller", {}).items()}
        return cls(
            controller=_build(ControllerConfig, c),
            idle=_build(IdleConfig, _tuple_field(raw.get("idle", {}), "ignored_processes")),
            reclaim=_build(ReclaimConfig, raw.get("reclaim", {})),
            spot=_build(SpotConfig, {
                k: Path(v) if k == "cuda_checkpoint" else v for k, v in raw.get("spot", {}).items()
            }),
        )


def _tuple_field(values: dict[str, Any], key: str) -> dict[str, Any]:
    if key in values:
        return {**values, key: tuple(str(v) for v in values[key])}
    return values


def _build[T](dc: type[T], values: dict[str, Any]) -> T:
    names = {f.name for f in fields(dc)}  # type: ignore[arg-type]
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown keys for {dc.__name__}: {sorted(unknown)}")
    return dc(**values)
