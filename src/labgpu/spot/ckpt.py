"""
Carry out spot placement actions (SPEC 2.12): NVIDIA cuda-checkpoint for moving and parking,
signals for eviction. Thin adapter; the decisions live in `labgpu.spot.placement`.
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
EVICT_MARKER = "tmp/labgpu-spot-evicted"  # inside the container, via /proc/<pid>/root


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


class CudaCheckpoint:
    def __init__(self, path: Path, timeout: float = 60.0) -> None:
        self.path = path
        self.timeout = timeout

    def available(self, driver_version: str) -> str | None:
        """None when usable, otherwise why not."""
        if not os.access(self.path, os.X_OK):
            return f"{self.path} not found or not executable"
        if driver_major(driver_version) < MIN_DRIVER_FOR_MOVE:
            return f"driver {driver_version} < {MIN_DRIVER_FOR_MOVE}"
        return None

    def _run(self, *args: str) -> None:
        try:
            proc = subprocess.run(
                [str(self.path), *args], capture_output=True, text=True, timeout=self.timeout
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise CheckpointError(f"cuda-checkpoint {' '.join(args)}: {e}") from e
        if proc.returncode != 0:
            raise CheckpointError(
                f"cuda-checkpoint {' '.join(args)} failed: {(proc.stderr or proc.stdout).strip()}"
            )

    def _each(self, action: str, pids: Sequence[int], *extra: str) -> None:
        for pid in pids:
            self._run("--action", action, "--pid", str(pid), *extra)

    def park(self, pids: Sequence[int]) -> None:
        """Lock and checkpoint: the processes stop issuing GPU work and free the GPU."""
        self._each("lock", pids)
        try:
            self._each("checkpoint", pids)
        except CheckpointError:
            self._try(lambda: self._each("unlock", pids))
            raise

    def restore(self, pids: Sequence[int], src: str, dst: str, visible: Sequence[str]) -> None:
        extra = () if src == dst else ("--device-map", device_map(src, dst, visible))
        self._each("restore", pids, *extra)
        self._each("unlock", pids)

    def move(self, pids: Sequence[int], src: str, dst: str, visible: Sequence[str]) -> None:
        self.park(pids)
        try:
            self.restore(pids, src, dst, visible)
        except CheckpointError:
            # Put it back where it was rather than leave it frozen.
            self._try(lambda: self.restore(pids, src, src, visible))
            raise

    @staticmethod
    def _try(fn: Callable[[], None]) -> None:
        try:
            fn()
        except CheckpointError as e:
            log.error("rollback failed: %s", e)


def evict(
    pids: Sequence[int],
    *,
    sig: signal.Signals,
    grace: float,
    reason: str,
    proc_root: Path = Path("/proc"),
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """
    Leave a marker file in the container, send `sig` (SIGINT raises KeyboardInterrupt in Python),
    wait up to `grace` seconds, then SIGKILL what is left. The session itself keeps running.
    """
    for pid in pids:
        try:
            (proc_root / str(pid) / "root" / EVICT_MARKER).write_text(reason + "\n")
        except OSError:
            pass
    alive = [p for p in pids if _signal(p, sig)]
    deadline = grace
    while alive and deadline > 0:
        sleep(1.0)
        deadline -= 1.0
        alive = [p for p in alive if _exists(p)]
    for p in alive:
        _signal(p, signal.SIGKILL)


def _signal(pid: int, sig: signal.Signals) -> bool:
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False


def _exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
