"""
Docker access through the `docker` CLI (SPEC 2.3, 2.6).

Argument building and inspect parsing are pure functions; `DockerCli` only runs commands.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..sizes import MiB
from .config import SpotConfig
from .jobspec import JobSpec

log = logging.getLogger("ai.backend.labgpu.spot.docker")

KERNEL_LABEL = "ai.backend.kernel-id"
SPOT_LABEL = "labgpu.spot"
JOB_LABEL = "labgpu.job-id"
GPU_LABEL = "labgpu.gpu-uuid"
HOOK_IN_CONTAINER = "/opt/labgpu/libvgpu.so"


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
    name: str
    job_id: int | None
    gpu_uuid: str | None
    running: bool
    exit_code: int | None


# ---- pure parsing ----


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


def parse_spot(inspect: Mapping[str, Any]) -> SpotContainer:
    labels = (inspect.get("Config") or {}).get("Labels") or {}
    state = inspect.get("State") or {}
    raw_job = labels.get(JOB_LABEL)
    return SpotContainer(
        id=inspect["Id"],
        name=inspect.get("Name", "").lstrip("/"),
        job_id=int(raw_job) if raw_job and raw_job.isdigit() else None,
        gpu_uuid=labels.get(GPU_LABEL),
        running=bool(state.get("Running")),
        exit_code=None if state.get("Running") else state.get("ExitCode"),
    )


def container_name(job_id: int, attempt: int) -> str:
    return f"labgpu-spot-{job_id}-{attempt}"


def build_run_args(
    *,
    job_id: int,
    attempt: int,
    spec: JobSpec,
    uid: int,
    gid: int,
    gpu_uuid: str,
    gpu_mem_limit: int,
    ram: int,
    cfg: SpotConfig,
    enforce: bool,
    attach_gpu: bool = True,
) -> list[str]:
    """The full `docker run` argv for a spot job (SPEC 2.6)."""
    if uid == 0:
        raise ValueError("spot jobs must not run as root")
    args = [
        "docker", "run", "-d", "--pull", "never",
        "--name", container_name(job_id, attempt),
        "--label", f"{SPOT_LABEL}=1",
        "--label", f"{JOB_LABEL}={job_id}",
        "--label", f"{GPU_LABEL}={gpu_uuid}",
        "--cpu-shares", str(cfg.cpu_shares),
        "--memory", str(ram),
        "--memory-swap", str(ram),
        "--oom-score-adj", "1000",
        "--user", f"{uid}:{gid}",
    ]
    if attach_gpu:
        args += ["--gpus", f"device={gpu_uuid}"]
    env = dict(spec.env)
    env.update({
        "LABGPU_SPOT": "1",
        "LABGPU_JOB_ID": str(job_id),
        "LABGPU_ATTEMPT": str(attempt),
    })
    if enforce:
        args += ["-v", f"{cfg.hook_path}:{HOOK_IN_CONTAINER}:ro"]
        env.update({
            "LD_PRELOAD": HOOK_IN_CONTAINER,
            "CUDA_DEVICE_MEMORY_LIMIT_0": f"{max(gpu_mem_limit // MiB, 1)}m",
            "CUDA_DEVICE_MEMORY_SHARED_CACHE": "/tmp/labgpu-vgpu.cache",
        })
    for key, value in env.items():
        args += ["-e", f"{key}={value}"]
    for m in spec.mounts:
        args += ["-v", f"{m.src}:{m.dst}" + (":ro" if m.readonly else "")]
    if spec.workdir:
        args += ["-w", spec.workdir]
    if spec.entrypoint is not None:
        args += ["--entrypoint", spec.entrypoint]
    args.append(spec.image)
    args.extend(spec.command)
    return args


# ---- command runner ----


class DockerCli:
    def __init__(self, docker: str = "docker", timeout: float = 30.0) -> None:
        self._docker = docker
        self._timeout = timeout
        self._stoppers: list[subprocess.Popen[bytes]] = []

    def _run(self, *args: str, timeout: float | None = None) -> str:
        try:
            proc = subprocess.run(
                [self._docker, *args],
                capture_output=True,
                text=True,
                timeout=timeout or self._timeout,
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

    def spot_containers(self) -> list[SpotContainer]:
        ids = self._run("ps", "-a", "-q", "--no-trunc", "--filter", f"label={SPOT_LABEL}=1").split()
        return [parse_spot(i) for i in self._inspect(ids)]

    def run(self, argv: Sequence[str]) -> str:
        if not argv or argv[0] != "docker":
            raise ValueError("argv must start with 'docker'")
        return self._run(*argv[1:], timeout=120).strip()

    def stop_async(self, container: str, grace_seconds: int) -> None:
        """SIGTERM, then SIGKILL after the grace period, without blocking the control loop."""
        self._reap()
        self._stoppers.append(
            subprocess.Popen(
                [self._docker, "stop", "-t", str(grace_seconds), container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )

    def save_logs_and_remove(self, container: str, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("wb") as f:
                subprocess.run(
                    [self._docker, "logs", container],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=self._timeout,
                )
        except (OSError, subprocess.TimeoutExpired) as e:
            log.warning("could not save logs of %s: %s", container, e)
        try:
            self._run("rm", "-f", container)
        except DockerError as e:
            log.warning("could not remove %s: %s", container, e)

    def wait_stops(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        for p in self._stoppers:
            try:
                p.wait(max(deadline - time.monotonic(), 0.1))
            except subprocess.TimeoutExpired:
                log.warning("docker stop still running at shutdown (pid %d)", p.pid)
        self._reap()

    def _reap(self) -> None:
        self._stoppers = [p for p in self._stoppers if p.poll() is None]
