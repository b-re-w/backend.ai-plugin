"""
Carry out spot placement actions (SPEC 2.12): NVIDIA cuda-checkpoint for moving and parking,
signals for eviction. Thin adapter; the decisions live in `labgpu.spot.placement`.

Everything goes through `docker exec -u root` into the spot container, the way the agent itself
works through the Docker daemon: the agent (and so the monitor) may run as an ordinary user in the
docker group, which cannot touch other users' processes on the host (SPEC 2.13). The spot plugin
mounts cuda-checkpoint into every spot container read-only at `CONTAINER_CUDA_CHECKPOINT`.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

log = logging.getLogger("ai.backend.labgpu.spot.ckpt")

MIN_DRIVER_FOR_MOVE = 580
CONTAINER_CUDA_CHECKPOINT = "/opt/labgpu/cuda-checkpoint"
EVICT_MARKER = "/tmp/labgpu-spot-evicted"
# The container env carries HAMi-core (LD_PRELOAD) and a reordered CUDA_VISIBLE_DEVICES meant for
# the user's program; cuda-checkpoint itself must run without them.
CLEAN_ENV = ("env", "-u", "LD_PRELOAD", "-u", "CUDA_VISIBLE_DEVICES")


class CheckpointError(RuntimeError):
    pass


def device_map(src: str, dst: str, visible: Sequence[str]) -> str:
    """cuda-checkpoint wants a bijection over every GPU the process can see: swap src and dst."""
    pairs = []
    for u in dict.fromkeys([*visible, src, dst]):
        new = dst if u == src else src if u == dst else u
        pairs.append(f"{u}={new}")
    return ",".join(pairs)


def driver_major(version: str) -> int:
    try:
        return int(version.split(".")[0])
    except ValueError:
        return 0


def container_pid(host_pid: int, proc_root: Path = Path("/proc")) -> int:
    """The PID a host process has inside its container (last NSpid entry; world-readable)."""
    try:
        for line in (proc_root / str(host_pid) / "status").read_text().splitlines():
            if line.startswith("NSpid:"):
                return int(line.split()[-1])
    except (OSError, ValueError) as e:
        raise CheckpointError(f"cannot map host pid {host_pid} into its container: {e}") from e
    raise CheckpointError(f"no NSpid for host pid {host_pid}")


class DockerExec:
    """Runs a command as root inside a container through the Docker daemon."""

    def __init__(self, docker: str = "docker", timeout: float = 60.0) -> None:
        self.docker = docker
        self.timeout = timeout

    def __call__(self, container_id: str, *argv: str) -> str:
        cmd = [self.docker, "exec", "-u", "root", container_id, *argv]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise CheckpointError(f"{' '.join(argv[:4])}: {e}") from e
        if proc.returncode != 0:
            raise CheckpointError(f"{' '.join(argv[:6])} failed: {(proc.stderr or proc.stdout).strip()}")
        return proc.stdout


class CudaCheckpoint:
    def __init__(
        self,
        host_path: Path,
        timeout: float = 60.0,
        *,
        run: Callable[..., str] | None = None,
        pid_in_container: Callable[[int], int] = container_pid,
    ) -> None:
        self.host_path = host_path  # what the spot plugin mounts; checked here, run in the container
        self.run = run or DockerExec(timeout=timeout)
        self.pid_in_container = pid_in_container

    def available(self, driver_version: str) -> str | None:
        """None when usable, otherwise why not."""
        if not os.access(self.host_path, os.X_OK):
            return f"{self.host_path} not found or not executable"
        if driver_major(driver_version) < MIN_DRIVER_FOR_MOVE:
            return f"driver {driver_version} < {MIN_DRIVER_FOR_MOVE}"
        return None

    def _each(self, cid: str, action: str, pids: Sequence[int], *extra: str) -> None:
        for pid in pids:
            self.run(
                cid, *CLEAN_ENV, CONTAINER_CUDA_CHECKPOINT,
                "--action", action, "--pid", str(self.pid_in_container(pid)), *extra,
            )

    def park(self, cid: str, pids: Sequence[int]) -> None:
        """Lock and checkpoint: the processes stop issuing GPU work and free the GPU."""
        self._each(cid, "lock", pids)
        try:
            self._each(cid, "checkpoint", pids)
        except CheckpointError:
            self._try(lambda: self._each(cid, "unlock", pids))
            raise

    def restore(self, cid: str, pids: Sequence[int], src: str, dst: str, visible: Sequence[str]) -> None:
        extra = () if src == dst else ("--device-map", device_map(src, dst, visible))
        self._each(cid, "restore", pids, *extra)
        self._each(cid, "unlock", pids)

    def move(self, cid: str, pids: Sequence[int], src: str, dst: str, visible: Sequence[str]) -> None:
        self.park(cid, pids)
        try:
            self.restore(cid, pids, src, dst, visible)
        except CheckpointError:
            # Put it back where it was rather than leave it frozen.
            self._try(lambda: self.restore(cid, pids, src, src, visible))
            raise

    @staticmethod
    def _try(fn: Callable[[], None]) -> None:
        try:
            fn()
        except CheckpointError as e:
            log.error("rollback failed: %s", e)


def evict(
    cid: str,
    pids: Sequence[int],
    *,
    sig: signal.Signals,
    grace: float,
    reason: str,
    run: Callable[..., str] | None = None,
    pid_in_container: Callable[[int], int] = container_pid,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """
    Leave a marker file in the container, send `sig` (SIGINT raises KeyboardInterrupt in Python),
    wait up to `grace` seconds, then SIGKILL what is left. The session itself keeps running.
    """
    run = run or DockerExec()
    inner = {p: pid_in_container(p) for p in pids if _exists(p)}
    if not inner:
        return
    name = sig.name.removeprefix("SIG")
    targets = " ".join(str(p) for p in inner.values())
    run(cid, "sh", "-c", f'printf "%s\\n" "$1" > {EVICT_MARKER}; kill -s {name} {targets}', "sh", reason)
    left = grace
    alive = list(inner)
    while alive and left > 0:
        sleep(1.0)
        left -= 1.0
        alive = [p for p in alive if _exists(p)]
    if alive:
        try:
            run(cid, "kill", "-s", "KILL", *(str(inner[p]) for p in alive))
        except CheckpointError as e:
            log.error("SIGKILL in %s failed: %s", cid[:12], e)


def _exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # another user's process: alive, we just may not signal it from the host
