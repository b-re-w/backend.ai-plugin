"""The spot controller loop: observe → reconcile → detect → plan → act (SPEC section 2)."""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from ..nvml import GpuInfo, GpuSnapshot, NvmlError
from .config import Config
from .detector import GpuTracker
from .docker import DockerError, OwnerContainer, SpotContainer, build_run_args
from .hostinfo import CpuSampler, mem_available
from .jobs import JobState, JobStore
from .model import GpuObservation, GpuVerdict, Launch, Reclaim
from .observer import observe
from .planner import plan

log = logging.getLogger("ai.backend.labgpu.spot")


class GpuSource(Protocol):
    def list_gpus(self) -> list[GpuInfo]: ...
    def snapshot(self, index: int) -> GpuSnapshot: ...
    def container_of_pids(self, pids: list[int]) -> dict[int, str | None]: ...


class ContainerRuntime(Protocol):
    def owner_containers(self) -> list[OwnerContainer]: ...
    def spot_containers(self) -> list[SpotContainer]: ...
    def run(self, argv: Sequence[str]) -> str: ...
    def stop_async(self, container: str, grace_seconds: int) -> None: ...
    def save_logs_and_remove(self, container: str, log_path: Path) -> None: ...
    def wait_stops(self, timeout: float) -> None: ...


class Controller:
    def __init__(
        self,
        cfg: Config,
        store: JobStore,
        gpus: GpuSource,
        docker: ContainerRuntime,
        *,
        pid_mapper: Callable[[list[int]], Mapping[int, str | None]] | None = None,
        ram_available: Callable[[], int] = mem_available,
        cpu_sampler: CpuSampler | None = None,
        hook_present: Callable[[], bool] | None = None,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.gpus = gpus
        self.docker = docker
        self.pid_mapper = pid_mapper or gpus.container_of_pids
        self.attach_gpus = not getattr(gpus, "is_fake", False)
        self.ram_available = ram_available
        self.cpu = cpu_sampler or CpuSampler()
        self.hook_present = hook_present or (lambda: cfg.spot.hook_path.is_file())
        self.trackers: dict[str, GpuTracker] = {}
        self.stop_issued: set[str] = set()
        self._gpu_list: list[GpuInfo] = []
        self.last_verdicts: dict[str, GpuVerdict] = {}

    # ---- one iteration ----

    def tick(self, now: float) -> None:
        paused_node, paused_gpus = self.store.paused()
        kill = self.cfg.controller.kill_switch_file.exists()

        docker_ok = True
        try:
            owners = self.docker.owner_containers()
            spots = self.docker.spot_containers()
        except DockerError as e:
            log.error("docker unavailable, treating all GPUs as unknown: %s", e)
            docker_ok, owners, spots = False, [], []

        observations = self._observe(now, owners, spots, docker_ok)
        if docker_ok:
            self._reconcile(spots)

        active = self.store.active()
        verdicts: dict[str, GpuVerdict] = {}
        for obs in observations:
            tracker = self.trackers.get(obs.uuid)
            if tracker is None:
                tracker = self.trackers[obs.uuid] = GpuTracker(obs, now)
            on_gpu = [a for a in active if a.gpu_uuid == obs.uuid]
            blocked = (
                "kill switch present" if kill
                else "node paused" if paused_node
                else "GPU paused" if obs.uuid in paused_gpus
                else None
            )
            verdicts[obs.uuid] = tracker.update(
                obs,
                now,
                self.cfg.idle,
                self.cfg.reclaim,
                lent=bool(on_gpu),
                reclaiming=bool(on_gpu) and all(a.reclaiming for a in on_gpu),
                blocked=blocked,
            )
        self.last_verdicts = verdicts

        # Stops that are wanted but not yet issued: user cancels, restarts mid-preemption.
        for a in active:
            if a.reclaiming and a.container and a.container not in self.stop_issued:
                self._stop(a.container)

        try:
            ram_budget = self.ram_available() - self.cfg.spot.host_ram_reserve
        except (OSError, ValueError) as e:
            log.error("cannot read host memory, not lending: %s", e)
            ram_budget = 0
        enforced = self.hook_present()
        p = plan(
            verdicts,
            active,
            self.store.queued(self.cfg.spot.default_ram) if docker_ok else [],
            ram_budget=ram_budget,
            enforced=enforced or self.cfg.spot.allow_unenforced,
        )
        for r in p.reclaims:
            self._reclaim(r, observations)
        for launch in p.launches:
            self._launch(launch, enforced, observations)

    def _observe(
        self,
        now: float,
        owners: list[OwnerContainer],
        spots: list[SpotContainer],
        docker_ok: bool,
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
            spots,
            self.pid_mapper,
            owner_cpu,
            ignored_names=self.cfg.idle.ignored_processes,
        )

    def _reconcile(self, spots: list[SpotContainer]) -> None:
        """Close out finished runs and remove spot containers we do not own (SPEC 2.10)."""
        by_id = {s.id: s for s in spots}
        known: set[str] = set()
        for a in self.store.active():
            c = by_id.get(a.container) if a.container else None
            if c is not None:
                known.add(c.id)
            if c is not None and c.running:
                continue
            new_state = self.store.finish(
                a.job_id,
                c.exit_code if c else None,
                max_attempts=self.cfg.spot.max_attempts,
                vanished=c is None,
            )
            self.stop_issued.discard(a.container or "")
            log.info(
                "job %d on %s ended (exit=%s, container %s) -> %s",
                a.job_id, a.gpu_uuid, c.exit_code if c else None,
                "gone" if c is None else "exited", new_state,
            )
            if c is not None:
                self._archive(c)
        for s in spots:
            if s.id in known:
                continue
            log.warning("removing unmanaged spot container %s (job label %s)", s.name, s.job_id)
            self._archive(s)

    def _archive(self, c: SpotContainer) -> None:
        name = f"{c.job_id if c.job_id is not None else 'unknown'}.{c.name}.log"
        self.docker.save_logs_and_remove(c.id, self.cfg.controller.state_dir / "logs" / name)

    def _stop(self, container: str) -> None:
        self.stop_issued.add(container)
        self.docker.stop_async(container, self.cfg.reclaim.grace_seconds)

    def _reclaim(self, r: Reclaim, observations: list[GpuObservation]) -> None:
        reason = "; ".join(r.reasons) or "reclaim"
        obs = next((o for o in observations if o.uuid == r.gpu_uuid), None)
        log.warning(
            "RECLAIM job %d from %s: %s [owner_util=%s owner_mem=%sMiB free=%sMiB]",
            r.job_id, r.gpu_uuid, reason,
            obs.owner_util if obs and obs.ok else "?",
            (obs.owner_memory >> 20) if obs and obs.ok else "?",
            (obs.free_memory >> 20) if obs and obs.ok else "?",
        )
        self.store.mark_preempting(r.job_id, reason)
        self._stop(r.container)

    def _launch(self, launch: Launch, enforced: bool, observations: list[GpuObservation]) -> None:
        rec = self.store.get(launch.job_id)
        if rec is None or rec.state is not JobState.QUEUED:
            return
        attempt = self.store.next_attempt(launch.job_id)
        try:
            argv = build_run_args(
                job_id=rec.id,
                attempt=attempt,
                spec=rec.spec,
                uid=rec.uid,
                gid=rec.gid,
                gpu_uuid=launch.gpu_uuid,
                gpu_mem_limit=launch.gpu_mem_limit,
                ram=launch.ram,
                cfg=self.cfg.spot,
                enforce=enforced,
                attach_gpu=self.attach_gpus,
            )
            container = self.docker.run(argv)
        except (DockerError, ValueError) as e:
            log.error("LAUNCH job %d failed: %s", rec.id, e)
            self.store.fail(rec.id, f"launch failed: {e}")
            return
        self.store.mark_running(rec.id, launch.gpu_uuid, container)
        obs = next((o for o in observations if o.uuid == launch.gpu_uuid), None)
        log.info(
            "LEND %s to job %d attempt %d: gpu_mem_limit=%dMiB ram=%dMiB "
            "[owner_util=%s owner_mem=%sMiB free=%sMiB]",
            launch.gpu_uuid, rec.id, attempt, launch.gpu_mem_limit >> 20, launch.ram >> 20,
            obs.owner_util if obs else "?", (obs.owner_memory >> 20) if obs else "?",
            (obs.free_memory >> 20) if obs else "?",
        )

    # ---- shutdown & status ----

    def reclaim_all(self, reason: str) -> None:
        for a in self.store.active():
            if a.container is None:
                continue
            if not a.reclaiming:
                self.store.mark_preempting(a.job_id, reason)
            self._stop(a.container)

    def write_status(self, path: Path, now: float) -> None:
        status = {
            "updated_at": now,
            "gpus": [
                {**asdict(v), "state": str(v.state)} for v in self.last_verdicts.values()
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
    status_path = state_dir / "status.json"
    while not stop.is_set():
        started = clock()
        try:
            controller.tick(started)
            controller.write_status(status_path, started)
        except Exception:
            log.exception("tick failed; reclaiming everything to stay safe")
            try:
                controller.reclaim_all("controller error")
            except Exception:
                log.exception("reclaim after error failed")
        stop.wait(max(controller.cfg.controller.poll_interval - (clock() - started), 0.5))
    # Without the controller nobody protects owners, so lent GPUs go back (SPEC 2.10).
    controller.reclaim_all("controller shutdown")
    controller.docker.wait_stops(controller.cfg.reclaim.grace_seconds + 10)
