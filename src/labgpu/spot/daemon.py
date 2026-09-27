"""
The spot monitor loop (SPEC section 2): observe the GPUs and the Backend.AI sessions on them,
judge which GPUs their owners are idling on, publish that for the spot plugins, and keep every
spot session on a GPU it may use (move, park, or evict it, SPEC 2.12).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

from ..nvml import GpuInfo, GpuSnapshot, NvmlError
from .ckpt import OOM_SIGNAL, CheckpointError, CudaCheckpoint, device_minor, has_handler
from .config import Config
from .detector import GpuTracker
from .docker import DockerError, OwnerContainer, SpotContainer
from .hostinfo import CpuSampler
from .model import GpuObservation, GpuVerdict, ProcKind
from .observer import observe
from .placement import Action, Hold, Move, Oom, Park, ParkedSpot, Restore, RunningSpot, plan

log = logging.getLogger("ai.backend.labgpu.spot")


class GpuSource(Protocol):
    def list_gpus(self) -> list[GpuInfo]: ...
    def snapshot(self, index: int) -> GpuSnapshot: ...
    def container_of_pids(self, pids: list[int]) -> dict[int, str | None]: ...
    def driver_version(self) -> str: ...


class SessionSource(Protocol):
    def sessions(self) -> tuple[list[OwnerContainer], list[SpotContainer]]: ...


@dataclass(frozen=True)
class ParkedRecord:
    spot: ParkedSpot
    pids: tuple[int, ...]
    memory: int  # GPU memory it held before parking
    hold: bool = False  # taken off a GPU it was not given: never restored automatically

    def to_json(self) -> dict:
        return {**asdict(self.spot), "allowed": list(self.spot.allowed), "pids": list(self.pids),
                "memory": self.memory, "hold": self.hold}

    @classmethod
    def from_json(cls, d: dict) -> ParkedRecord:
        spot = ParkedSpot(d["container_id"], d["src"], tuple(d["allowed"]), float(d["since"]))
        return cls(spot, tuple(int(p) for p in d["pids"]), int(d["memory"]), bool(d.get("hold")))


@dataclass(frozen=True)
class Gate:
    """Internal operation: set a spot container's device-file permissions."""

    container_id: str
    gpu: str


class Controller:
    def __init__(
        self,
        cfg: Config,
        gpus: GpuSource,
        sessions: SessionSource,
        *,
        pid_mapper: Callable[[list[int]], Mapping[int, str | None]] | None = None,
        cpu_sampler: CpuSampler | None = None,
        checkpointer: CudaCheckpoint | None = None,
        workers: ThreadPoolExecutor | None = None,
        handler_check: Callable[[int], bool] | None = None,
    ) -> None:
        self.cfg = cfg
        self.gpus = gpus
        self.sessions = sessions
        self.pid_mapper = pid_mapper or gpus.container_of_pids
        self.cpu = cpu_sampler or CpuSampler()
        self.trackers: dict[str, GpuTracker] = {}
        self._gpu_list: list[GpuInfo] = []
        self.last_verdicts: dict[str, GpuVerdict] = {}
        self.last_obs: dict[str, GpuObservation] = {}
        self.spot_seen: dict[tuple[str, str], float] = {}  # (container, gpu) -> first seen
        self.spot_info: dict[str, SpotContainer] = {}
        self.parked: dict[str, ParkedRecord] = {}
        self.busy: dict[str, str] = {}  # container -> operation in flight
        self.oomed: dict[str, float] = {}  # container -> when its program got the out-of-memory error
        self.assigned: dict[str, str] = {}  # container -> the GPU it was given (and moved to)
        self.held: dict[str, ParkedRecord] = {}  # processes caught on GPUs they were not given
        self.gated: dict[str, str] = {}  # container -> the GPU its device files are opened for
        self.minors: dict[str, int | None] = {}  # GPU uuid -> /dev/nvidia<minor>
        self._lock = threading.Lock()
        self._done: list[tuple[Action, bool, ParkedRecord | None]] = []
        self.checkpointer = checkpointer
        self.handler_check = handler_check or (lambda pid: has_handler(pid, OOM_SIGNAL))
        self.can_checkpoint = False
        self.workers = workers or ThreadPoolExecutor(max_workers=4, thread_name_prefix="spot-op")
        self._parked_path = cfg.controller.state_dir / "parked.json"

    # ---- setup ----

    def start(self) -> None:
        spot = self.cfg.spot
        if self.checkpointer is None:
            self.checkpointer = CudaCheckpoint(spot.cuda_checkpoint, spot.checkpoint_timeout_seconds)
        try:
            why = self.checkpointer.available(self.gpus.driver_version())
        except NvmlError as e:
            why = f"driver version unknown: {e}"
        self.can_checkpoint = why is None
        if why:
            log.error("spot sessions cannot be moved or parked (%s); they only get the out-of-memory error", why)
        self._load_parked()

    def _load_parked(self) -> None:
        try:
            records = [ParkedRecord.from_json(d) for d in json.loads(self._parked_path.read_text())]
            self.parked = {r.spot.container_id: r for r in records if not r.hold}
            self.held = {r.spot.container_id: r for r in records if r.hold}
        except (OSError, ValueError, KeyError, TypeError):
            self.parked, self.held = {}, {}
        if self.parked:
            log.warning("resuming %d parked spot sessions from %s", len(self.parked), self._parked_path)

    def _save_parked(self) -> None:
        tmp = self._parked_path.with_suffix(".tmp")
        records = [*self.parked.values(), *self.held.values()]
        tmp.write_text(json.dumps([r.to_json() for r in records]))
        os.replace(tmp, self._parked_path)

    # ---- one tick ----

    def tick(self, now: float) -> None:
        self._collect_done()
        try:
            owners, spots = self.sessions.sessions()
            docker_ok = True
        except DockerError as e:
            log.error("docker unavailable, treating all GPUs as unknown: %s", e)
            owners, spots, docker_ok = [], [], False
        self.spot_info = {s.id: s for s in spots}
        for s in spots:
            if s.gpu:
                self.assigned.setdefault(s.id, s.gpu)
        for cid in [c for c in self.assigned if c not in self.spot_info]:
            del self.assigned[cid]
        for cid in [c for c in self.held if c not in self.spot_info]:
            del self.held[cid]
        for cid in [c for c in self.gated if c not in self.spot_info]:
            del self.gated[cid]
        observations = self._observe(now, owners, spots, docker_ok)
        self.last_obs = {o.uuid: o for o in observations}
        busy_gpus = self._busy_gpus()
        verdicts: dict[str, GpuVerdict] = {}
        for obs in observations:
            tracker = self.trackers.get(obs.uuid)
            if tracker is None:
                tracker = self.trackers[obs.uuid] = GpuTracker(obs, now)
            verdicts[obs.uuid] = tracker.update(
                obs,
                now,
                self.cfg.idle,
                self.cfg.reclaim,
                lent=bool(obs.spot_containers),
                reclaiming=obs.uuid in busy_gpus,
            )
        self.last_verdicts = verdicts
        if docker_ok:
            self._place(now, spots)

    def _busy_gpus(self) -> set[str]:
        return {
            o.uuid for o in self.last_obs.values() if o.spot_containers & set(self.busy)
        }

    def _observe(
        self, now: float, owners: list[OwnerContainer], spots: list[SpotContainer], docker_ok: bool
    ) -> list[GpuObservation]:
        try:
            self._gpu_list = self.gpus.list_gpus()
            for g in self._gpu_list:
                if g.uuid not in self.minors:
                    self.minors[g.uuid] = device_minor(g.pci_bus_id)
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
            spots=spots,
        )

    # ---- spot placement (SPEC 2.12) ----

    def _running(self, now: float) -> list[RunningSpot]:
        gpus_of: dict[str, set[str]] = {}
        for o in self.last_obs.values():
            for cid in o.spot_containers:
                gpus_of.setdefault(cid, set()).add(o.uuid)
        seen = {(cid, g) for cid, gs in gpus_of.items() for g in gs}
        for key in seen - set(self.spot_seen):
            self.spot_seen[key] = now
        for key in set(self.spot_seen) - seen:
            del self.spot_seen[key]
        return [
            RunningSpot(
                cid,
                frozenset(gs),
                self.spot_info[cid].uuids if cid in self.spot_info else tuple(gs),
                min(self.spot_seen[(cid, g)] for g in gs),
                oom_ready=any(self.handler_check(pid) for pid in self._pids_on(cid, None)),
                assigned=self.assigned.get(cid),
            )
            for cid, gs in gpus_of.items()
        ]

    def _pids_on(self, cid: str, gpu: str | None) -> tuple[int, ...]:
        # A process with contexts on two GPUs is listed once.
        return tuple(dict.fromkeys(
            p.pid
            for o in self.last_obs.values()
            if gpu is None or o.uuid == gpu
            for p in o.processes
            if p.kind is ProcKind.SPOT and p.container_id == cid
        ))

    def _mem_on(self, cid: str, gpu: str) -> int:
        o = self.last_obs.get(gpu)
        if o is None:
            return 0
        return sum(p.used_memory for p in o.processes if p.kind is ProcKind.SPOT and p.container_id == cid)

    def _place(self, now: float, spots: list[SpotContainer]) -> None:
        # Parked sessions whose container is gone need nothing more.
        live = {s.id for s in spots}
        for cid in [c for c in self.parked if c not in live and c not in self.busy]:
            log.info("parked spot %s is gone", cid[:12])
            del self.parked[cid]
            self._save_parked()
        # Forget the OOM time of sessions that left their GPU (moved, parked, or finished).
        on_gpu = {c for o in self.last_obs.values() for c in o.spot_containers}
        for c in [c for c in self.oomed if c not in on_gpu and c not in self.busy]:
            del self.oomed[c]
        actions = plan(
            self.last_verdicts,
            self._running(now),
            [r.spot for r in self.parked.values()],
            now=now,
            can_checkpoint=self.can_checkpoint and self.cfg.spot.enabled,
            oom_grace=self.cfg.spot.oom_grace_seconds,
            oomed=self.oomed,
            busy=frozenset(self.busy),
        )
        for action in actions:
            self._dispatch(action, now)
        self._gate_new(spots)

    def _gate(self, cid: str, allow: Sequence[str] = (), deny: Sequence[str] = ()) -> None:
        """chmod the containers' /dev/nvidia<minor> files (SPEC 2.12 rule 10); unknown minors skipped."""
        a = [self.minors[u] for u in allow if self.minors.get(u) is not None]
        d = [self.minors[u] for u in deny if self.minors.get(u) is not None]
        if a or d:
            self.checkpointer.gate(cid, a, d)

    def _gate_new(self, spots: list[SpotContainer]) -> None:
        """
        A spot container can open only the GPU it was given, from the first time it is seen; a
        parked one (no GPU at all) none of them, so nothing new starts on the owner's GPU.
        """
        for s in spots:
            gpu = "" if s.id in self.parked else self.assigned.get(s.id)
            if gpu is None or self.gated.get(s.id) == gpu or s.id in self.busy:
                continue
            allow = [gpu] if gpu else []
            others = [u for u in s.uuids if u != gpu]
            self._submit(Gate(s.id, gpu), lambda s=s, allow=allow, others=others: self._gate(s.id, allow, others), None, None)

    def _lendable(self, uuid: str) -> int:
        v = self.last_verdicts.get(uuid)
        return v.lendable_memory if v is not None else 0

    def _dispatch(self, action: Action, now: float) -> None:
        cid = action.container_id
        visible = self.spot_info[cid].uuids if cid in self.spot_info else ()
        ck = self.checkpointer
        match action:
            case Move(_, src, dst, reason):
                pids = self._pids_on(cid, src)
                size = self._lendable(dst)
                log.info("spot %s: move %s -> %s (%s) pids=%s", cid[:12], src, dst, reason, pids)

                def job() -> None:
                    # Open the target GPU's device file only for the move, then close the old one.
                    self._gate(cid, allow=[dst])
                    try:
                        ck.move(cid, pids, src, dst, visible)
                    except CheckpointError:
                        self._gate(cid, deny=[dst])
                        raise
                    self._gate(cid, deny=[src])
                    ck.place(cid, pids[0], self._order(visible, dst), size)

                fallback = self._oom_job(cid, pids, self._order(visible, src), reason)
            case Park(_, src, reason):
                pids = self._pids_on(cid, src)
                record = ParkedRecord(ParkedSpot(cid, src, visible, now), pids, self._mem_on(cid, src))
                log.info("spot %s: park off %s (%s) pids=%s", cid[:12], src, reason, pids)

                def park_job() -> None:
                    ck.park(cid, pids)
                    # Nothing of it is on a GPU now: close every device in the same step, so no
                    # new program can open the GPU just handed back to its owner (SPEC 2.12).
                    self._gate(cid, deny=list(visible))

                self._submit(action, park_job, None, record)
                return
            case Restore(_, src, dst):
                record = self.parked[cid]
                size = self._lendable(dst)
                log.info("spot %s: restore parked %s -> %s", cid[:12], src, dst)

                def job() -> None:
                    self._gate(cid, allow=[dst])
                    try:
                        ck.restore(cid, record.pids, src, dst, record.spot.allowed)
                    except CheckpointError:
                        if dst != self.assigned.get(cid):
                            self._gate(cid, deny=[dst])  # never leave a failed target open
                        raise
                    if src != dst:
                        self._gate(cid, deny=[src])
                    ck.place(cid, record.pids[0], self._order(record.spot.allowed, dst), size)

                fallback = None
            case Hold(_, gpus, reason):
                pids = tuple(dict.fromkeys(p for g in gpus for p in self._pids_on(cid, g)))
                log.warning("spot %s: %s; parking pids=%s and holding them", cid[:12], reason, pids)
                record = ParkedRecord(
                    ParkedSpot(cid, gpus[0], visible, now), pids, sum(self._mem_on(cid, g) for g in gpus), hold=True
                )
                self._submit(action, lambda: ck.park(cid, pids), None, record)
                return
            case Oom(_, gpu, reason):
                # Per process (SPEC 2.12): one that installed the handler gets the error; one that
                # did not is parked right away so it does not keep the owner's GPU.
                pids = self._pids_on(cid, None)
                ready = tuple(p for p in pids if self.handler_check(p))
                silent = tuple(p for p in pids if p not in ready) if self.can_checkpoint else ()
                log.warning(
                    "spot %s: out-of-memory error on %s (%s) pids=%s, parked without it=%s",
                    cid[:12], gpu, reason, ready or pids, silent,
                )
                oom = self._oom_job(cid, ready or pids, self._order(visible, gpu), reason)
                record = (
                    ParkedRecord(ParkedSpot(cid, gpu, visible, now), silent, self._mem_on(cid, gpu))
                    if silent
                    else None
                )

                def job() -> None:
                    # The error goes out first (parking takes seconds), and even if parking the
                    # others fails; only a delivered error starts the grace period.
                    if ready or not silent:
                        oom()
                        self.oomed[cid] = now  # the tick's clock, as placement compares with it
                    if silent:
                        try:
                            ck.park(cid, silent)
                        except CheckpointError as e:
                            raise CheckpointError(f"processes {silent} are still on the GPU: {e}") from e

                self._submit(action, job, None, record)
                return
        self._submit(action, job, fallback, None)

    @staticmethod
    def _order(visible: Sequence[str], current: str) -> list[str]:
        """Attached GPUs with the one the session runs on first: its programs' cuda:0."""
        return [current, *(u for u in visible if u != current)]

    def _oom_job(self, cid: str, pids: tuple[int, ...], order: list[str], reason: str) -> Callable[[], None]:
        """torch.OutOfMemoryError in the program right away (SPEC 2.12). Nothing is ever killed."""
        ck = self.checkpointer
        return lambda: ck.raise_oom(cid, pids, order, reason)

    def _submit(
        self,
        action: Action,
        job: Callable[[], None],
        fallback: Callable[[], None] | None,
        record: ParkedRecord | None,
    ) -> None:
        cid = action.container_id
        self.busy[cid] = type(action).__name__

        def run() -> None:
            ok = True
            try:
                job()
            except (CheckpointError, OSError) as e:
                ok = False
                log.error("spot %s: %s failed: %s", cid[:12], type(action).__name__, e)
                if fallback is not None:
                    log.warning("spot %s: raising the out-of-memory error instead", cid[:12])
                    self.oomed[cid] = time.time()
                    try:
                        fallback()
                    except (CheckpointError, OSError) as e2:
                        log.error("spot %s: out-of-memory error failed: %s", cid[:12], e2)
            except Exception:
                ok = False
                log.exception("spot %s: %s crashed", cid[:12], type(action).__name__)
            with self._lock:
                self._done.append((action, ok, record))

        self.workers.submit(run)

    def _collect_done(self) -> None:
        with self._lock:
            done, self._done = self._done, []
        changed = False
        for action, ok, record in done:
            cid = action.container_id
            self.busy.pop(cid, None)
            if isinstance(action, Hold):
                # It got past the device permissions (e.g. sudo chmod): close them again.
                self.gated.pop(cid, None)
            if isinstance(action, Hold) and ok and record is not None:
                old = self.held.get(cid)
                self.held[cid] = (
                    replace(old, pids=tuple(dict.fromkeys(old.pids + record.pids))) if old else record
                )
                changed = True
            elif isinstance(action, (Move, Restore)) and ok:
                self.assigned[cid] = action.dst
                self.gated[cid] = action.dst
            elif isinstance(action, Gate) and ok:
                self.gated[cid] = action.gpu
            if isinstance(action, Park) and ok:
                self.gated[cid] = ""  # every device closed
            if isinstance(action, (Park, Oom)) and ok and record is not None:
                old = self.parked.get(cid)
                # A session may be parked in parts (processes without the handler first).
                self.parked[cid] = (
                    replace(old, pids=tuple(dict.fromkeys(old.pids + record.pids)), memory=old.memory + record.memory)
                    if old is not None
                    else record
                )
                changed = True
            elif isinstance(action, Restore) and ok and cid in self.parked:
                del self.parked[cid]
                changed = True
        if changed:
            self._save_parked()

    # ---- status ----

    def write_status(self, path: Path, now: float) -> None:
        """Per-GPU state for `labgpu-spot status` and the plugins (SPEC 2.9.1)."""
        gpus = []
        for v in self.last_verdicts.values():
            o = self.last_obs.get(v.uuid)
            spots = sorted(o.spot_containers) if o else []
            since = min((self.spot_seen.get((c, v.uuid), now) for c in spots), default=None)
            gpus.append({
                **asdict(v),
                "state": str(v.state),
                "lent_job": spots[0][:12] if spots else None,
                "lent_since": since,
            })
        status = {
            "updated_at": now,
            "spot_enabled": self.cfg.spot.enabled,
            "can_move": self.can_checkpoint,
            "gpus": gpus,
            "parked": [
                {"container": c[:12], "from": r.spot.src, "since": r.spot.since}
                for c, r in self.parked.items()
            ],
            "held": [
                {"container": c[:12], "from": r.spot.src, "pids": list(r.pids), "since": r.spot.since}
                for c, r in self.held.items()
            ],
            "busy": {c[:12]: op for c, op in self.busy.items()},
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2))
        os.replace(tmp, path)


def _loop(controller: Controller, stop: threading.Event, clock: Callable[[], float]) -> None:
    state_dir = controller.cfg.controller.state_dir
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        controller.start()
    except Exception:
        # Without a status file the spot plugins report no room: spot stays off, owners unaffected.
        log.exception("spot monitor cannot start (state_dir %s); spot slots stay at 0", state_dir)
        return
    status_path = state_dir / "status.json"
    while not stop.is_set():
        started = clock()
        try:
            controller.tick(started)
            controller.write_status(status_path, started)
        except Exception:
            log.exception("tick failed")
        stop.wait(max(controller.cfg.controller.poll_interval - (clock() - started), 0.5))
    # Operations in flight finish; parked sessions stay parked and are resumed on restart.
    controller.workers.shutdown(wait=True)


def run_forever(controller: Controller, *, clock: Callable[[], float] = time.time) -> None:
    """Foreground loop for `labgpu-spot daemon` (development and debugging)."""
    stop = threading.Event()

    def _on_signal(signum: int, _frame: object) -> None:
        log.info("received signal %d, shutting down", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    _loop(controller, stop, clock)


class BackgroundMonitor:
    """
    The monitor running in a thread of the Backend.AI agent (SPEC 2.13): started by the first
    spot plugin that initialises, stopped when that plugin is cleaned up. One per process.
    """

    _lock = threading.Lock()
    _running: BackgroundMonitor | None = None

    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=_loop, args=(controller, self.stop_event, time.time), name="labgpu-spot", daemon=True
        )

    @classmethod
    def ensure_started(cls, make: Callable[[], Controller]) -> BackgroundMonitor | None:
        """Start the monitor unless this process already runs one; returns it only to its starter."""
        with cls._lock:
            if cls._running is not None:
                return None
            monitor = cls(make())
            monitor.thread.start()
            cls._running = monitor
            log.info("spot monitor started in this process")
            return monitor

    def stop(self, timeout: float = 30.0) -> None:
        self.stop_event.set()
        self.thread.join(timeout)
        with self._lock:
            if BackgroundMonitor._running is self:
                BackgroundMonitor._running = None
        log.info("spot monitor stopped")
