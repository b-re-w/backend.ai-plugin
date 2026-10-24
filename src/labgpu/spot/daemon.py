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
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

from ..nvml import GpuInfo, GpuSnapshot, NvmlError
from ..procmap import pid_in_container
from .ckpt import (
    OOM_SIGNAL,
    STATE_FILE_NAME,
    CheckpointError,
    CudaCheckpoint,
    device_minor,
    has_handler,
    parse_spot_env,
)
from .config import Config
from .detector import GpuTracker
from .docker import DockerError, OwnerContainer, SpotContainer
from .hostinfo import CpuSampler, node_size, read_meminfo_gib
from .model import GpuObservation, GpuState, GpuVerdict, ProcKind
from .observer import observe
from .pool import Pool, memory_pressure, resolve
from .placement import (
    Action,
    Hold,
    IdleSpot,
    Move,
    Oom,
    Park,
    ParkedSpot,
    Rehome,
    Resize,
    Restore,
    RunningSpot,
    can_host,
    idle_stays,
    plan,
)

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
        spot = ParkedSpot(
            d["container_id"], d["src"], tuple(d["allowed"]), float(d["since"]), float(d.get("share", 1.0))
        )
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
        pid_check: Callable[[int, str], bool] | None = None,
        node_info: Callable[[], tuple[int | None, float | None]] | None = None,
        mem_available: Callable[[], float | None] | None = None,
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
        self.first_seen: dict[str, float] = {}  # spot container -> when the monitor first listed it
        self.spot_info: dict[str, SpotContainer] = {}
        self.parked: dict[str, ParkedRecord] = {}
        self.busy: dict[str, str] = {}  # container -> operation in flight
        self.oomed: dict[str, float] = {}  # container -> when its program got the out-of-memory error
        self.assigned: dict[str, str] = {}  # container -> the GPU it was given (and moved to)
        self.held: dict[str, ParkedRecord] = {}  # processes caught on GPUs they were not given
        self.gated: dict[str, str] = {}  # container -> the GPU its device files are opened for
        self.minors: dict[str, int | None] = {}  # GPU uuid -> /dev/nvidia<minor>
        self.caps: dict[str, int] = {}  # container -> memory limit last applied by the monitor, bytes
        self.limited: set[str] = set()  # containers whose limits an out-of-memory error dropped to 1 MiB
        self.waiting_since: dict[str, float] = {}  # idle spot sessions waiting for a GPU to be rehomed on
        self.lost_gpu: set[str] = set()  # parked sessions whose record was pruned: no GPU until a Rehome
        self.published: dict[str, str] = {}  # container -> state last written to its state file (SPEC 2.15)
        self._resume: dict | None = None  # saved judgement to carry on from at the first tick (SPEC 2.10)
        self._inspected: set[str] = set()  # spot containers whose env file was read (or is being read)
        self._found: dict[str, tuple[str, bool]] = {}  # container -> (GPU, limited) from its env file
        self._lock = threading.Lock()
        self._done: list[tuple[Action, bool, ParkedRecord | None]] = []
        self.checkpointer = checkpointer
        self.handler_check = handler_check or (lambda pid: has_handler(pid, OOM_SIGNAL))
        self.pid_check = pid_check or pid_in_container
        self.can_checkpoint = False
        # Shared spot pool (SPEC 2.16): None when off, which keeps the SPEC 2.12 behaviour.
        self.pool: Pool | None = None
        if cfg.pool.enabled:
            self.pool, notes = resolve(cfg.pool, *(node_info or node_size)())
            for level, text in notes:
                getattr(log, level)("spot pool: %s", text)
        self._mem_available = mem_available or (lambda: read_meminfo_gib("MemAvailable"))
        self.mem_pressure = False
        self.workers = workers or ThreadPoolExecutor(max_workers=4, thread_name_prefix="spot-op")
        self._parked_path = cfg.controller.state_dir / "parked.json"
        self._state_path = cfg.controller.state_dir / "monitor.json"

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
        self._load_state()

    def _load_state(self) -> None:
        try:
            saved = json.loads(self._state_path.read_text())
            self._resume = saved if isinstance(saved, dict) and float(saved.get("saved_at", 0)) > 0 else None
        except (OSError, ValueError, TypeError):
            self._resume = None

    def _save_state(self, now: float) -> None:
        """The judgement a restarted monitor carries on from (SPEC 2.10)."""
        state = {
            "saved_at": now,
            "gpus": {u: t.snapshot(now) for u, t in self.trackers.items()},
            "assigned": dict(self.assigned),
            "caps": dict(self.caps),
            "limited": sorted(self.limited),
        }
        try:
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state))
            os.replace(tmp, self._state_path)
        except OSError as e:
            log.warning("cannot save the monitor state to %s: %s", self._state_path, e)

    def _take_resume(self, now: float, live: set[str]) -> dict | None:
        """The saved judgement if it is recent enough, applied to spot sessions still here; once."""
        saved, self._resume = self._resume, None
        if saved is None:
            return None
        age = now - float(saved.get("saved_at", 0))
        if not 0 <= age <= self.cfg.controller.resume_max_age_seconds:
            log.info("saved monitor state is %.0fs old; every GPU starts as just used", age)
            return None
        log.warning("resuming the monitor state saved %.0fs ago", age)
        for cid, gpu in (saved.get("assigned") or {}).items():
            if cid in live:
                self.assigned[cid] = str(gpu)
        for cid, cap in (saved.get("caps") or {}).items():
            if cid in live:
                self.caps[cid] = int(cap)
        self.limited.update(c for c in saved.get("limited") or () if c in live)
        return saved.get("gpus") or {}

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
        resumed = self._take_resume(now, set(self.spot_info)) if docker_ok else None
        for s in spots:
            self.first_seen.setdefault(s.id, now)
        for cid in [c for c in self.first_seen if c not in self.spot_info]:
            del self.first_seen[cid]
        unknown = [s for s in spots if s.id not in self.assigned and s.id not in self._inspected]
        for s in spots:
            if s.gpu:
                self.assigned.setdefault(s.id, s.gpu)
        for s in unknown:
            self._inspect(s)
        self._inspected &= set(self.spot_info)
        self._apply_found()
        for cid in [c for c in self.assigned if c not in self.spot_info]:
            del self.assigned[cid]
        for cid in [c for c in self.held if c not in self.spot_info]:
            del self.held[cid]
        for cid in [c for c in self.gated if c not in self.spot_info]:
            del self.gated[cid]
        for cid in [c for c in self.caps if c not in self.spot_info]:
            del self.caps[cid]
        self.limited &= set(self.spot_info)
        for cid in [c for c in self.waiting_since if c not in self.spot_info]:
            del self.waiting_since[cid]
        self.lost_gpu &= set(self.spot_info)
        observations = self._observe(now, owners, spots, docker_ok)
        self.last_obs = {o.uuid: o for o in observations}
        busy_gpus = self._busy_gpus()
        verdicts: dict[str, GpuVerdict] = {}
        for obs in observations:
            tracker = self.trackers.get(obs.uuid)
            if tracker is None:
                tracker = self.trackers[obs.uuid] = GpuTracker(obs, now)
                if resumed and obs.uuid in resumed:
                    tracker.resume(resumed[obs.uuid], now)
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
            self._save_state(now)

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
                share=self._share(cid),
                short=len(gs) == 1 and self._short(cid, next(iter(gs))),
            )
            for cid, gs in gpus_of.items()
        ]

    def _inspect(self, s: SpotContainer) -> None:
        """
        A spot session the monitor has no state for (new, or seen first after a restart without
        saved state) may have been moved, or left at 1 MiB by an out-of-memory error, by an earlier
        monitor; the env file that monitor wrote says so (SPEC 2.10, 2.12 rule 1-2).
        """
        self._inspected.add(s.id)
        if not s.user or self.checkpointer is None:
            return
        ck = self.checkpointer

        def job() -> None:
            try:
                text = ck.read_spot_env(s.id, s.user)
            except Exception:
                log.exception("spot %s: reading its env file crashed", s.id[:12])
                return
            if text:
                with self._lock:
                    self._found[s.id] = parse_spot_env(text)

        self.workers.submit(job)

    def _apply_found(self) -> None:
        with self._lock:
            found, self._found = self._found, {}
        for cid, (gpu, limited) in found.items():
            info = self.spot_info.get(cid)
            if info is None:
                continue
            if gpu and gpu in info.uuids:
                self.assigned[cid] = gpu
            if limited:
                self.limited.add(cid)
                log.warning("spot %s: its memory limit is 1 MiB from an earlier out-of-memory error", cid[:12])

    def _idle_spots(self) -> list[IdleSpot]:
        """Spot sessions with no GPU process yet (SPEC 2.12 rules 1-1, 1-2)."""
        on_gpu = {c for o in self.last_obs.values() for c in o.spot_containers}
        return [
            IdleSpot(cid, gpu, self.spot_info[cid].uuids, self._share(cid), cid in self.limited)
            for cid in sorted(self.spot_info)
            if (gpu := self.assigned.get(cid))
            and cid not in on_gpu and cid not in self.parked and cid not in self.held
        ]

    def _idle(self) -> dict[str, list[str]]:
        """Spot sessions with no GPU process that keep their share on the GPU they were given."""
        out: dict[str, list[str]] = {}
        for i in self._idle_spots():
            if idle_stays(i, self.last_verdicts):
                out.setdefault(i.assigned, []).append(i.container_id)
        return out

    def _short(self, cid: str, gpu: str) -> bool:
        """Its limit is clearly below share x capacity (fragmented placement), so raise it when possible."""
        applied = self.caps.get(cid) or (self.spot_info[cid].cap if cid in self.spot_info else 0)
        full = self._cap(cid, gpu)
        return bool(applied) and applied < full * 0.95

    def _share(self, cid: str) -> float:
        info = self.spot_info.get(cid)
        return info.share if info is not None else 1.0

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
        self._prune_parked()
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
            idle_spots=self._idle_spots(),
            idle_holds_share=self.pool is None,
        )
        actions = self._hold_back_new_grants(actions)
        # Idle sessions that cannot stay where they were and got no GPU this tick wait (rule 1-2).
        rehomed = {a.container_id for a in actions if isinstance(a, Rehome)}
        for i in self._idle_spots():
            if idle_stays(i, self.last_verdicts) or i.container_id in rehomed:
                self.waiting_since.pop(i.container_id, None)
            else:
                self.waiting_since.setdefault(i.container_id, now)
        for cid in [c for c in self.waiting_since if c not in {i.container_id for i in self._idle_spots()}]:
            del self.waiting_since[cid]
        for action in actions:
            self._dispatch(action, now)
        self._gate_new(spots)

    def _hold_back_new_grants(self, actions: list[Action]) -> list[Action]:
        """
        Pool mode, node memory below the floor: no new GPU is handed out (rehoming an idle session,
        restoring a parked one). Nothing is stopped or cleaned up (SPEC 2.16.6, 2.16.4).
        """
        if self.pool is None:
            return actions
        pressure = memory_pressure(self.pool, self._mem_available())
        if pressure != self.mem_pressure:
            self.mem_pressure = pressure
            if pressure:
                log.warning(
                    "spot pool: node MemAvailable is below %g GiB; no new GPU is granted to spot sessions "
                    "(none is stopped)", self.pool.mem_floor_gib,
                )
            else:
                log.info("spot pool: node memory is back above the floor; GPU grants resume")
        if not pressure:
            return actions
        return [a for a in actions if not isinstance(a, (Rehome, Restore))]

    def _prune_parked(self) -> None:
        """Drop recorded processes that ended (or whose PID was reused) while parked (SPEC 2.12 rule 5-1)."""
        changed = False
        for records in (self.parked, self.held):
            for cid, record in list(records.items()):
                if cid in self.busy:
                    continue
                alive = tuple(p for p in record.pids if self.pid_check(p, cid))
                if alive == record.pids:
                    continue
                changed = True
                gone = tuple(p for p in record.pids if p not in alive)
                if alive:
                    log.info("spot %s: parked pids %s ended, keeping %s", cid[:12], gone, alive)
                    records[cid] = replace(record, pids=alive)
                else:
                    log.info("spot %s: every parked pid %s ended, forgetting the record", cid[:12], gone)
                    del records[cid]
                    if records is self.parked:
                        # It has no GPU any more; the GPU it was given may be the owner's by now.
                        self.lost_gpu.add(cid)
        if changed:
            self._save_parked()

    def _gate(self, cid: str, allow: Sequence[str] = (), deny: Sequence[str] = ()) -> None:
        """chmod the containers' /dev/nvidia<minor> files (SPEC 2.12 rule 10); unknown minors skipped."""
        a = [self.minors[u] for u in allow if self.minors.get(u) is not None]
        d = [self.minors[u] for u in deny if self.minors.get(u) is not None]
        if a or d:
            self.checkpointer.gate(cid, a, d)

    @contextmanager
    def _all_open(self, cid: str, visible: Sequence[str]) -> Iterator[SimpleNamespace]:
        """
        Every attached device open while cuda-checkpoint works on the session: it needs the GPUs
        the process has used at checkpoint time and both source and target at restore time
        (2026-09-27 Primary measurement; the session user is subject to file permissions). On the
        way out, success or not, only `end.state` stays open ("" = none) (SPEC 2.12 rule 10).
        """
        end = SimpleNamespace(state="")
        self._gate(cid, allow=list(visible))
        try:
            yield end
        finally:
            keep = [end.state] if end.state else []
            self._gate(cid, allow=keep, deny=[u for u in visible if u not in keep])

    def _gate_new(self, spots: list[SpotContainer]) -> None:
        """
        A spot container can open only the GPU it was given, from the first time it is seen; a
        parked one (no GPU at all) none of them, so nothing new starts on the owner's GPU. A parked
        session whose record was pruned (its process ended) and now waits to be rehomed has no GPU
        either: its old `assigned` GPU may be the one the owner reclaimed, so every device stays
        closed until a Rehome opens its destination.
        """
        on_gpu = {c for o in self.last_obs.values() for c in o.spot_containers}
        for s in spots:
            gpu = "" if s.id in self.parked or s.id in self.lost_gpu else self.assigned.get(s.id)
            if gpu and s.id not in on_gpu and s.id not in self.held and not self._may_open(gpu):
                # No GPU process, and its GPU is the owner's right now (or unknown). `lost_gpu` is
                # in memory only, so after a monitor restart a session that lost its GPU looks new:
                # open nothing until a Rehome, Restore, or Move places it, or the GPU is free.
                gpu = ""
            if (
                self.pool is not None
                and gpu
                and s.id not in on_gpu
                and s.id not in self.held
                and not can_host(self.last_verdicts.get(gpu))
            ):
                # Pool mode creates sessions even when no GPU is lendable (SPEC 2.16.5): an idle
                # one keeps every device closed until a Rehome opens a GPU that can host it.
                gpu = ""
            if gpu is None or self.gated.get(s.id) == gpu or s.id in self.busy:
                continue
            allow = [gpu] if gpu else []
            others = [u for u in s.uuids if u != gpu]
            self._submit(Gate(s.id, gpu), lambda s=s, allow=allow, others=others: self._gate(s.id, allow, others), None, None)

    def _may_open(self, gpu: str) -> bool:
        """Whether a session with no GPU process may have this GPU's device file opened (fail safe)."""
        v = self.last_verdicts.get(gpu)
        return v is not None and not v.must_reclaim and v.state in (GpuState.IDLE, GpuState.LENDABLE, GpuState.LENT)

    def _cap(self, cid: str, uuid: str) -> int:
        """Memory limit for a session on `uuid`: its share of the GPU's spot capacity (SPEC 2.12)."""
        v = self.last_verdicts.get(uuid)
        capacity = (v.spot_capacity or v.lendable_memory) if v is not None else 0
        return int(capacity * self._share(cid))

    def _dispatch(self, action: Action, now: float) -> None:
        cid = action.container_id
        visible = self.spot_info[cid].uuids if cid in self.spot_info else ()
        ck = self.checkpointer
        match action:
            case Move(_, src, dst, reason):
                pids = self._pids_on(cid, src)
                size = self._cap(cid, dst)
                log.info("spot %s: move %s -> %s (%s) pids=%s", cid[:12], src, dst, reason, pids)

                def job() -> None:
                    with self._all_open(cid, visible) as end:
                        end.state = src  # if the move fails it stays where it was
                        ck.move(cid, pids, src, dst, visible)
                        end.state = dst
                    ck.place(cid, pids[0], self._order(visible, dst), size)
                    self.caps[cid] = size

                fallback = self._oom_job(cid, pids, self._order(visible, src), reason)
            case Park(_, src, reason):
                pids = self._pids_on(cid, src)
                record = ParkedRecord(
                    ParkedSpot(cid, src, visible, now, self._share(cid)), pids, self._mem_on(cid, src)
                )
                log.info("spot %s: park off %s (%s) pids=%s", cid[:12], src, reason, pids)

                def park_job() -> None:
                    with self._all_open(cid, visible) as end:
                        end.state = src
                        ck.park(cid, pids)
                        # Nothing of it is on a GPU now: every device closed in the same step,
                        # so no new program can open the GPU just handed back to its owner.
                        end.state = ""

                self._submit(action, park_job, None, record)
                return
            case Restore(_, src, dst):
                record = self.parked[cid]
                size = int((self.last_verdicts[dst].spot_capacity if dst in self.last_verdicts else 0) * record.spot.share)
                log.info("spot %s: restore parked %s -> %s", cid[:12], src, dst)

                def job() -> None:
                    with self._all_open(cid, record.spot.allowed) as end:
                        end.state = ""  # still parked if it fails: no device
                        ck.restore(cid, record.pids, src, dst, record.spot.allowed)
                        end.state = dst
                    ck.place(cid, record.pids[0], self._order(record.spot.allowed, dst), size)
                    self.caps[cid] = size

                fallback = None
            case Hold(_, gpus, reason):
                pids = tuple(dict.fromkeys(p for g in gpus for p in self._pids_on(cid, g)))
                log.warning("spot %s: %s; parking pids=%s and holding them", cid[:12], reason, pids)
                record = ParkedRecord(
                    ParkedSpot(cid, gpus[0], visible, now, self._share(cid)), pids,
                    sum(self._mem_on(cid, g) for g in gpus), hold=True,
                )
                home = self.assigned.get(cid, "")

                def hold_job() -> None:
                    with self._all_open(cid, visible) as end:
                        end.state = home
                        ck.park(cid, pids)

                self._submit(action, hold_job, None, record)
                return
            case Resize(_, gpu):
                pids = self._pids_on(cid, gpu)
                size = self._cap(cid, gpu)
                log.info("spot %s: memory limit raised to its share on %s (%d MiB)", cid[:12], gpu, size >> 20)

                def job() -> None:
                    ck.place(cid, pids[0], self._order(visible, gpu), size)
                    self.caps[cid] = size

                fallback = None
            case Rehome(_, src, dst):
                info = self.spot_info.get(cid)
                user = info.user if info is not None else ""
                size = self._cap(cid, dst)
                if not user:
                    log.error("spot %s: cannot rehome %s -> %s: session user unknown", cid[:12], src, dst)
                    return
                log.info("spot %s: rehome %s -> %s (no GPU process, %d MiB)", cid[:12], src, dst, size >> 20)

                def job() -> None:
                    ck.place_as(cid, user, self._order(visible, dst), size)
                    self._gate(cid, allow=[dst], deny=[u for u in visible if u != dst])
                    self.caps[cid] = size

                fallback = None
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
                    ParkedRecord(ParkedSpot(cid, gpu, visible, now, self._share(cid)), silent, self._mem_on(cid, gpu))
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
                            with self._all_open(cid, visible) as end:
                                end.state = gpu  # the others keep running there
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

        def job() -> None:
            ck.raise_oom(cid, pids, order, reason)
            with self._lock:
                self.limited.add(cid)  # every limit is 1 MiB now, until it is placed again

        return job

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
            elif isinstance(action, (Move, Restore, Rehome)) and ok:
                self.assigned[cid] = action.dst
                self.gated[cid] = action.dst
                self.limited.discard(cid)
                self.waiting_since.pop(cid, None)
                self.lost_gpu.discard(cid)
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
        idle = self._idle()
        for v in self.last_verdicts.values():
            o = self.last_obs.get(v.uuid)
            spots = sorted(o.spot_containers) if o else []
            since = min((self.spot_seen.get((c, v.uuid), now) for c in spots), default=None)
            gpus.append({
                **asdict(v),
                "state": str(v.state),
                "lent_job": spots[0][:12] if spots else None,
                "lent_since": since,
                "spots": [{"container": c[:12], "share": self._share(c), "seen": self.first_seen.get(c)} for c in spots]
                + [{"container": c[:12], "share": self._share(c), "seen": self.first_seen.get(c), "idle": True}
                   for c in idle.get(v.uuid, [])],
                # Pool mode: idle sessions hold no share (SPEC 2.16.4).
                "spot_share": round(
                    min(sum(self._share(c) for c in [*spots, *(idle.get(v.uuid, []) if self.pool is None else [])]), 1.0),
                    4,
                ),
            })
        sessions = self._spot_sessions()
        self._publish(sessions, now)
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
            "spot_sessions": sessions,
            "pool": self._pool_status(),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2))
        os.replace(tmp, path)

    def _pool_status(self) -> dict:
        """Pool figures for the status file (SPEC 2.9.1, 2.16.5); `enabled: false` when the pool is off."""
        if self.pool is None:
            return {"enabled": False}
        spots = list(self.spot_info.values())
        return {
            "enabled": True,
            "cpu_cores": self.pool.cpu_cores,
            "mem_gib": self.pool.mem_gib,
            "max_sessions": self.pool.max_sessions,
            "mem_floor_gib": self.pool.mem_floor_gib,
            "mem_pressure": self.mem_pressure,
            "sessions": len(spots),
            "requested_cpu_cores": sum(s.cpus for s in spots),
            "requested_mem_gib": round(sum(s.mem for s in spots) / (1 << 30), 2),
        }

    def _publish(self, sessions: Mapping[str, Mapping], now: float) -> None:
        """
        Each spot session's state to /home/config/labgpu-spot-state in its container, and the resume
        signal when a reclaimed session can use a GPU again (SPEC 2.15).
        """
        for cid in [c for c in self.published if c not in self.spot_info]:
            del self.published[cid]
        for cid, info in self.spot_info.items():
            entry = sessions.get(cid[:12])
            if entry is None:
                continue
            prev, state = self.published.get(cid), str(entry["state"])
            if state == prev:
                continue
            if info.config_dir and not self._write_state(Path(info.config_dir), entry, now):
                continue  # retried on the next tick
            self.published[cid] = state
            if prev in RECLAIMED_STATES and state in USABLE_STATES and info.user:
                self.workers.submit(self._resume_job(cid, info.user, prev, state))

    @staticmethod
    def _write_state(config_dir: Path, entry: Mapping, now: float) -> bool:
        target = config_dir / STATE_FILE_NAME
        tmp = config_dir / f".{STATE_FILE_NAME}.tmp"
        body = {"state": entry["state"], "gpu": entry.get("gpu"), "since": entry.get("since"), "updated_at": now}
        try:
            tmp.write_text(json.dumps(body) + "\n")
            os.replace(tmp, target)
            return True
        except OSError as e:
            log.warning("cannot write the spot state file %s: %s", target, e)
            return False

    def _resume_job(self, cid: str, user: str, prev: str, state: str) -> Callable[[], None]:
        ck = self.checkpointer

        def job() -> None:
            try:
                ck.notify_resume(cid, user)
                log.info("spot %s: %s -> %s, sent the resume signal", cid[:12], prev, state)
            except (CheckpointError, OSError) as e:
                log.warning("spot %s: resume signal failed: %s", cid[:12], e)

        return job

    def _spot_sessions(self) -> dict[str, dict]:
        """What each known spot session is going through, for its user (SPEC 2.9.1, 2.14)."""
        on_gpu: dict[str, str] = {}
        for o in self.last_obs.values():
            for c in o.spot_containers:
                on_gpu.setdefault(c, o.uuid)
        out: dict[str, dict] = {}
        for cid in sorted(set(self.spot_info) | set(self.parked) | set(self.held)):
            if cid in self.held:
                r = self.held[cid]
                entry = {"state": "held", "gpu": r.spot.src, "since": r.spot.since}
            elif cid in self.parked:
                r = self.parked[cid]
                entry = {"state": "parked", "gpu": r.spot.src, "since": r.spot.since}
            elif cid in on_gpu and cid in self.oomed:
                entry = {"state": "oom", "gpu": on_gpu[cid], "since": self.oomed[cid]}
            elif cid in on_gpu:
                gpu = on_gpu[cid]
                entry = {"state": "running", "gpu": gpu, "since": self.spot_seen.get((cid, gpu))}
            elif cid in self.waiting_since:
                # No GPU process, and the GPU it had can no longer host it: waits to be rehomed.
                entry = {"state": "parked", "gpu": self.assigned.get(cid), "since": self.waiting_since[cid]}
            else:
                entry = {"state": "idle", "gpu": self.assigned.get(cid), "since": None}
            out[cid[:12]] = entry
        return out


# SPEC 2.15: the resume signal goes out when a session moves from one of these to the other.
RECLAIMED_STATES = frozenset({"parked", "held", "oom"})
USABLE_STATES = frozenset({"idle", "running"})


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
