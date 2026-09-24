"""Spot job specification files (SPEC 2.8)."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Self

from ..sizes import parse_size


class JobSpecError(ValueError):
    pass


@dataclass(frozen=True)
class Mount:
    src: str
    dst: str
    readonly: bool = False


@dataclass(frozen=True)
class JobSpec:
    name: str
    image: str
    command: tuple[str, ...]
    workdir: str | None = None
    entrypoint: str | None = None
    gpu_mem: int = 0
    ram: int = 0  # 0: use the controller's default_ram
    priority: int = 0
    env: dict[str, str] = field(default_factory=dict)
    mounts: tuple[Mount, ...] = ()
    gpu_models: tuple[str, ...] = ()

    @classmethod
    def from_toml(cls, path: Path) -> Self:
        with path.open("rb") as f:
            return cls.from_dict(tomllib.load(f))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        allowed = {
            "name", "image", "command", "workdir", "entrypoint",
            "gpu_mem", "ram", "priority", "env", "mounts", "gpu_models",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise JobSpecError(f"unknown keys: {sorted(unknown)}")
        for key in ("name", "image", "command"):
            if key not in raw:
                raise JobSpecError(f"missing required key: {key}")
        command = raw["command"]
        if isinstance(command, str) or not command or not all(isinstance(c, str) for c in command):
            raise JobSpecError("command must be a non-empty list of strings")
        env = raw.get("env", {})
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise JobSpecError("env values must be strings")
        gpu_models = raw.get("gpu_models", [])
        if isinstance(gpu_models, str) or not all(isinstance(m, str) and m for m in gpu_models):
            raise JobSpecError("gpu_models must be a list of non-empty strings")
        mounts = []
        for m in raw.get("mounts", []):
            try:
                mounts.append(Mount(src=m["src"], dst=m["dst"], readonly=bool(m.get("readonly", False))))
            except KeyError as e:
                raise JobSpecError(f"mount is missing {e}") from None
        try:
            return cls(
                name=str(raw["name"]),
                image=str(raw["image"]),
                command=tuple(command),
                workdir=raw.get("workdir"),
                entrypoint=raw.get("entrypoint"),
                gpu_mem=parse_size(raw["gpu_mem"]) if "gpu_mem" in raw else 0,
                ram=parse_size(raw["ram"]) if "ram" in raw else 0,
                priority=int(raw.get("priority", 0)),
                env=dict(env),
                mounts=tuple(mounts),
                gpu_models=tuple(gpu_models),
            )
        except ValueError as e:
            raise JobSpecError(str(e)) from None

    def validate_mounts(self, allowed_roots: Sequence[Path], *, resolve: bool = True) -> None:
        """Every mount source must resolve to a path under one of the allowed roots."""
        for m in self.mounts:
            src = os.path.realpath(m.src) if resolve else os.path.normpath(m.src)
            src_path = PurePosixPath(src.replace("\\", "/"))
            if not src_path.is_absolute():
                raise JobSpecError(f"mount source must be absolute: {m.src}")
            if not any(src_path.is_relative_to(PurePosixPath(str(r).replace("\\", "/"))) for r in allowed_roots):
                raise JobSpecError(
                    f"mount source {m.src} is outside allowed roots "
                    f"{[str(r) for r in allowed_roots]}"
                )
            if not PurePosixPath(m.dst).is_absolute():
                raise JobSpecError(f"mount destination must be absolute: {m.dst}")

    def resolved(self) -> Self:
        """A copy with mount sources replaced by their resolved real paths."""
        mounts = tuple(Mount(os.path.realpath(m.src), m.dst, m.readonly) for m in self.mounts)
        return type(self)(
            **{**asdict(self), "command": self.command, "mounts": mounts, "gpu_models": self.gpu_models}
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> Self:
        raw = json.loads(text)
        raw["command"] = tuple(raw["command"])
        raw["mounts"] = tuple(Mount(**m) for m in raw["mounts"])
        raw["gpu_models"] = tuple(raw.get("gpu_models", ()))
        return cls(**raw)
