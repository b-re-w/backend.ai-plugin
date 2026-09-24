"""Per-GPU idle/activity state machine (SPEC 2.4). Pure: time is passed in."""

from __future__ import annotations

from dataclasses import replace

from .config import IdleConfig, ReclaimConfig
from .model import GpuObservation, GpuState, GpuVerdict, ProcKind


class GpuTracker:
    """Remembers one GPU's activity history across ticks."""

    def __init__(self, first: GpuObservation, now: float) -> None:
        self.uuid = first.uuid
        # A fresh controller treats every GPU as just used (SPEC 2.4, restart rule).
        self.last_active = now
        self.mem_baseline = first.owner_memory if first.ok else 0
        self.prev_owners = first.owner_containers if first.ok else frozenset()

    def update(
        self,
        obs: GpuObservation,
        now: float,
        idle: IdleConfig,
        reclaim: ReclaimConfig,
        *,
        lent: bool,
        reclaiming: bool = False,
        blocked: str | None = None,
    ) -> GpuVerdict:
        verdict = self._decide(obs, now, idle, reclaim, lent=lent, reclaiming=reclaiming, blocked=blocked)
        return replace(verdict, model=obs.model)

    def _decide(
        self,
        obs: GpuObservation,
        now: float,
        idle: IdleConfig,
        reclaim: ReclaimConfig,
        *,
        lent: bool,
        reclaiming: bool,
        blocked: str | None,
    ) -> GpuVerdict:
        if not obs.ok:
            self.last_active = now
            return GpuVerdict(
                uuid=obs.uuid,
                state=GpuState.UNKNOWN,
                lendable=False,
                must_reclaim=lent and not reclaiming,
                reasons=(f"observation failed: {obs.error}",),
            )

        activity = self._activity_reasons(obs, idle)
        self.prev_owners = obs.owner_containers
        if activity:
            self.last_active = now
            self.mem_baseline = obs.owner_memory
        else:
            self.mem_baseline = min(self.mem_baseline, obs.owner_memory)

        idle_for = now - self.last_active
        lendable_memory = max(obs.free_memory - reclaim.mem_reserve, 0)

        if lent:
            reasons = list(activity)
            if obs.free_memory < reclaim.mem_reserve // 2:
                reasons.append(
                    f"free memory {obs.free_memory >> 20}MiB < half of reserve "
                    f"{reclaim.mem_reserve_mib}MiB"
                )
            if blocked:
                reasons.append(blocked)
            must = bool(reasons) and not reclaiming
            state = GpuState.RECLAIMING if (reclaiming or must) else GpuState.LENT
            return GpuVerdict(obs.uuid, state, False, must, tuple(reasons), 0, idle_for)

        if activity:
            return GpuVerdict(obs.uuid, GpuState.BUSY, False, False, tuple(activity), 0, idle_for)
        wait = idle.idle_seconds if obs.owner_containers else idle.unclaimed_grace_seconds
        if blocked or idle_for < wait or lendable_memory <= 0:
            reasons = (blocked,) if blocked else ()
            return GpuVerdict(obs.uuid, GpuState.IDLE, False, False, reasons, 0, idle_for)
        return GpuVerdict(
            obs.uuid, GpuState.LENDABLE, True, False, (), lendable_memory, idle_for
        )

    def _activity_reasons(self, obs: GpuObservation, idle: IdleConfig) -> list[str]:
        reasons: list[str] = []
        if obs.owner_util > idle.owner_util_threshold:
            reasons.append(f"owner SM util {obs.owner_util}% > {idle.owner_util_threshold}%")
        if obs.owner_memory > self.mem_baseline + idle.owner_mem_delta:
            reasons.append(
                f"owner memory {obs.owner_memory >> 20}MiB grew over baseline "
                f"{self.mem_baseline >> 20}MiB + {idle.owner_mem_delta_mib}MiB"
            )
        if (
            idle.owner_cpu_threshold > 0
            and obs.owner_cpu_cores is not None
            and obs.owner_cpu_cores > idle.owner_cpu_threshold
        ):
            reasons.append(
                f"owner CPU {obs.owner_cpu_cores:.2f} cores > {idle.owner_cpu_threshold}"
            )
        if obs.has_unknown:
            pids = [p.pid for p in obs.processes if p.kind is ProcKind.UNKNOWN]
            reasons.append(f"unidentified GPU processes {pids}")
        new_owners = obs.owner_containers - self.prev_owners
        if new_owners:
            reasons.append(f"new owner containers {sorted(c[:12] for c in new_owners)}")
        return reasons
