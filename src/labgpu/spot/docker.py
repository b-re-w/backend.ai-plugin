"""
Find the Backend.AI session containers on this node and the GPUs they hold (SPEC 2.3).

Parsing is pure; `DockerCli` only runs `docker ps` / `docker inspect`.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

KERNEL_LABEL = "ai.backend.kernel-id"


class DockerError(RuntimeError):
    pass


@dataclass(frozen=True)
class OwnerContainer:
    id: str
    pid: int
    gpu_refs: tuple[str, ...]  # UUIDs, NVML indices, or "all"


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

    def owner_containers(self) -> list[OwnerContainer]:
        ids = self._run("ps", "-q", "--no-trunc", "--filter", f"label={KERNEL_LABEL}").split()
        return [parse_owner(i) for i in self._inspect(ids)]
