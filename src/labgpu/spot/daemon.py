"""
The spot controller loop (SPEC section 2): observe the GPUs and the Backend.AI sessions holding
them, judge which GPUs their owners are idling on, and publish that in the status file.

Lending itself (running spot sessions) is not implemented yet; see SPEC section 2.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from ..nvml import GpuInfo, GpuSnapshot, NvmlError
from .config import Config
from .detector import GpuTracker
from .docker import DockerError, OwnerContainer
from .hostinfo import CpuSampler
from .model import GpuObservation, GpuVerdict
from .observer import observe

log = logging.getLogger("ai.backend.labgpu.spot")


class GpuSource(Protocol):
    def list_gpus(self) -> list[GpuInfo]: ...
    def snapshot(self, index: int) -> GpuSnapshot: ...
    def container_of_pids(self, pids: list[int]) -> dict[int, str | None]: ...


class SessionSource(Protocol):
    def owner_containers(self) -> list[OwnerContainer]: ...


class Controller:
    def __init__(
        self,
        cfg: Config,
        gpus: GpuSource,
        sessions: SessionSource,
        *,
        pid_mapper: Callable[[list[int]], Mapping[int, str | None]] | None = None,
        cpu_sampler: CpuSampler | None = None,
    ) -> None:
        self.cfg = cfg
        self.gpus = gpus
        self.sessions = sessions
        self.pid_mapper = pid_mapper or gpus.container_of_pids
        self.cpu = cpu_sampler or CpuSampler()
        self.trackers: dict[str, GpuTracker] = {}
        self._gpu_list: list[GpuInfo] = []
        self.last_verdicts: dict[str, GpuVerdict] = {}

    def tick(self, now: float) -> None:
        try:
            owners = self.sessions.owner_containers()
            docker_ok = True
        except DockerError as e:
            log.error("docker unavailable, treating all GPUs as unknown: %s", e)
            owners, docker_ok = [], False
        verdicts: dict[str, GpuVerdict] = {}
        for obs in self._observe(now, owners, docker_ok):
            tracker = self.trackers.get(obs.uuid)
            if tracker is None:
                tracker = self.trackers[obs.uuid] = GpuTracker(obs, now)
            verdicts[obs.uuid] = tracker.update(obs, now, self.cfg.idle, self.cfg.reclaim, lent=False)
        self.last_verdicts = verdicts

    def _observe(
        self, now: float, owners: list[OwnerContainer], docker_ok: bool
    ) -> list[GpuObservation]:
        try:
            self._gpu_list = self.gpus.list_gpus()
        except NvmlError as e:
            log.error("NVML unavailable: %s", e)
            return [
                GpuObservation(g.uuid, g.index, ok=False, model=g.name, error=str(e))
                for g in self._gpu_list
            ]
        if not docker_ok:
            return [
                GpuObservation(g.uuid, g.index, ok=False, model=g.name, error="docker unavailable")
                for g in self._gpu_list
            ]
        owner_cpu = {o.id: self.cpu.sample(o.id, o.pid, now) for o in owners if o.pid}
        self.cpu.forget_except({o.id for o in owners})
        return observe(
            self._gpu_list,
            self.gpus.snapshot,
            owners,
            self.pid_mapper,
            owner_cpu,
            ignored_names=self.cfg.idle.ignored_processes,
        )

    def write_status(self, path: Path, now: float) -> None:
        """Per-GPU state for `labgpu-spot status` and the accelerator plugin (SPEC 2.9.1)."""
        status = {
            "updated_at": now,
            "gpus": [
                {**asdict(v), "state": str(v.state), "lent_job": None, "lent_since": None}
                for v in self.last_verdicts.values()
            ],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2))
        os.replace(tmp, path)


def run_forever(controller: Controller, *, clock: Callable[[], float] = time.time) -> None:
    stop = threading.Event()

    def _on_signal(signum: int, _frame: object) -> None:
        log.info("received signal %d, shutting down", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    state_dir = controller.cfg.controller.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    status_path = state_dir / "status.json"
    while not stop.is_set():
        started = clock()
        try:
            controller.tick(started)
            controller.write_status(status_path, started)
        except Exception:
            log.exception("tick failed")
        stop.wait(max(controller.cfg.controller.poll_interval - (clock() - started), 0.5))
