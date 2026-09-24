"""Spot controller configuration (SPEC 2.2)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Self

from ..sizes import MiB, parse_size

DEFAULT_CONFIG_PATH = Path("/etc/labgpu/spot.toml")


@dataclass(frozen=True)
class ControllerConfig:
    poll_interval: float = 5.0
    state_dir: Path = Path("/var/lib/labgpu")
    kill_switch_file: Path = Path("/etc/labgpu/spot.disabled")


@dataclass(frozen=True)
class IdleConfig:
    idle_minutes: float = 30.0
    owner_util_threshold: int = 5
    owner_mem_delta_mib: int = 512
    owner_cpu_threshold: float = 0.5
    unclaimed_grace_seconds: float = 60.0

    @property
    def idle_seconds(self) -> float:
        return self.idle_minutes * 60

    @property
    def owner_mem_delta(self) -> int:
        return self.owner_mem_delta_mib * MiB


@dataclass(frozen=True)
class ReclaimConfig:
    grace_seconds: int = 30
    mem_reserve_mib: int = 2048

    @property
    def mem_reserve(self) -> int:
        return self.mem_reserve_mib * MiB


@dataclass(frozen=True)
class SpotConfig:
    hook_path: Path = Path("/opt/labgpu/lib/libvgpu.so")
    allow_unenforced: bool = False
    cpu_shares: int = 64
    default_ram: int = 16 * 2**30
    host_ram_reserve: int = 32 * 2**30
    allowed_mount_roots: tuple[Path, ...] = (Path("/vfroot"),)
    max_attempts: int = 20


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
        c = raw.get("controller", {})
        s = dict(raw.get("spot", {}))
        for key in ("default_ram", "host_ram_reserve"):
            if key in s:
                s[key] = parse_size(s[key])
        if "hook_path" in s:
            s["hook_path"] = Path(s["hook_path"])
        if "allowed_mount_roots" in s:
            s["allowed_mount_roots"] = tuple(Path(p) for p in s["allowed_mount_roots"])
        c = {k: Path(v) if k in ("state_dir", "kill_switch_file") else v for k, v in c.items()}
        return cls(
            controller=_build(ControllerConfig, c),
            idle=_build(IdleConfig, raw.get("idle", {})),
            reclaim=_build(ReclaimConfig, raw.get("reclaim", {})),
            spot=_build(SpotConfig, s),
        )


def _build[T](dc: type[T], values: dict[str, Any]) -> T:
    names = {f.name for f in fields(dc)}  # type: ignore[arg-type]
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown keys for {dc.__name__}: {sorted(unknown)}")
    return dc(**values)
