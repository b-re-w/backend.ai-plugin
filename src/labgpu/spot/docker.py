"""
Find the Backend.AI session containers on this node and the GPUs they hold (SPEC 2.3), telling
spot sessions (env `LABGPU_SPOT=1`, SPEC 2.12) apart from owners.

Parsing is pure; `DockerCli` only runs `docker ps` / `docker inspect`.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

KERNEL_LABEL = "ai.backend.kernel-id"
ENV_SPOT = "LABGPU_SPOT"
ENV_SPOT_UUIDS = "LABGPU_SPOT_UUIDS"


class DockerError(RuntimeError):
    pass


@dataclass(frozen=True)
class OwnerContainer:
    id: str
    pid: int
    gpu_refs: tuple[str, ...]  # UUIDs, NVML indices, or "all"


@dataclass(frozen=True)
class SpotContainer:
    id: str
    pid: int
    uuids: tuple[str, ...]  # every GPU attached to it: where it may run or be moved to


def _env(inspect: Mapping[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in (inspect.get("Config") or {}).get("Env") or []:
        key, _, value = item.partition("=")
        env[key] = value
    return env


def parse_gpu_refs(inspect: Mapping[str, Any]) -> tuple[str, ...]:
    """GPU references attached to a container, in SPEC 2.3 precedence order."""
    env = _env(inspect)
    if env.get("LABGPU_DEVICE_UUIDS"):
        return tuple(x for x in env["LABGPU_DEVICE_UUIDS"].split(",") if x)
    refs: list[str] = []
    for req in (inspect.get("HostConfig") or {}).get("DeviceRequests") or []:
        if req.get("Driver") == "nvidia":
            if req.get("Count") == -1:
                return ("all",)
            refs.extend(req.get("DeviceIDs") or [])
    if refs:
        return tuple(refs)
    visible = env.get("NVIDIA_VISIBLE_DEVICES", "")
    if visible in ("", "void", "none"):
        return ()
    if visible == "all":
        return ("all",)
    return tuple(x.strip() for x in visible.split(",") if x.strip())


def parse_owner(inspect: Mapping[str, Any]) -> OwnerContainer:
    return OwnerContainer(
        id=inspect["Id"],
        pid=int((inspect.get("State") or {}).get("Pid") or 0),
        gpu_refs=parse_gpu_refs(inspect),
    )


def is_spot(inspect: Mapping[str, Any]) -> bool:
    return _env(inspect).get(ENV_SPOT) == "1"


def parse_spot(inspect: Mapping[str, Any]) -> SpotContainer:
    env = _env(inspect)
    return SpotContainer(
        id=inspect["Id"],
        pid=int((inspect.get("State") or {}).get("Pid") or 0),
        uuids=tuple(u for u in env.get(ENV_SPOT_UUIDS, "").split(",") if u),
    )


def split_sessions(
    inspects: Sequence[Mapping[str, Any]],
) -> tuple[list[OwnerContainer], list[SpotContainer]]:
    owners: list[OwnerContainer] = []
    spots: list[SpotContainer] = []
    for i in inspects:
        if is_spot(i):
            spots.append(parse_spot(i))
        else:
            owners.append(parse_owner(i))
    return owners, spots


class DockerCli:
    def __init__(self, docker: str = "docker", timeout: float = 30.0) -> None:
        self._docker = docker
        self._timeout = timeout

    def _run(self, *args: str) -> str:
        try:
            proc = subprocess.run(
                [self._docker, *args], capture_output=True, text=True, timeout=self._timeout
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise DockerError(f"docker {args[0]}: {e}") from e
        if proc.returncode != 0:
            raise DockerError(f"docker {args[0]} failed: {proc.stderr.strip()}")
        return proc.stdout

    def _inspect(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        return json.loads(self._run("inspect", *ids))

    def sessions(self) -> tuple[list[OwnerContainer], list[SpotContainer]]:
        ids = self._run("ps", "-q", "--no-trunc", "--filter", f"label={KERNEL_LABEL}").split()
        return split_sessions(self._inspect(ids))
