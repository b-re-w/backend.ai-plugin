"""Turn verdicts, running spots, and the queue into actions (SPEC 2.5). Pure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..selection import model_matches
from ..sizes import GiB
from .model import GpuVerdict, Launch, QueuedJob, Reclaim, RunningSpot

DEFAULT_JOB_GPU_MEM = 1 * GiB


@dataclass(frozen=True)
class Plan:
    reclaims: tuple[Reclaim, ...]
    launches: tuple[Launch, ...]


def plan(
    verdicts: Mapping[str, GpuVerdict],
    running: Sequence[RunningSpot],
    queue: Sequence[QueuedJob],
    *,
    ram_budget: int,
    enforced: bool,
) -> Plan:
    """
    `ram_budget` is host MemAvailable minus the configured host reserve.
    `enforced` is whether spot GPU memory limits can be applied (HAMi-core present or
    explicitly waived).
    """
    reclaims: list[Reclaim] = []
    for spot in running:
        if spot.reclaiming:
            continue
        verdict = verdicts.get(spot.gpu_uuid)
        if verdict is None:
            reclaims.append(Reclaim(spot.job_id, spot.gpu_uuid, spot.container, ("GPU vanished",)))
        elif verdict.must_reclaim:
            reclaims.append(Reclaim(spot.job_id, spot.gpu_uuid, spot.container, verdict.reasons))

    launches: list[Launch] = []
    if not enforced:
        return Plan(tuple(reclaims), ())

    occupied = {s.gpu_uuid for s in running}
    ordered = sorted(queue, key=lambda j: (-j.priority, j.submitted_at, j.job_id))
    taken: set[int] = set()
    budget = ram_budget
    for uuid in sorted(verdicts):
        verdict = verdicts[uuid]
        if not verdict.lendable or uuid in occupied:
            continue
        for job in ordered:
            if job.job_id in taken:
                continue
            need_gpu = job.gpu_mem or DEFAULT_JOB_GPU_MEM
            if need_gpu > verdict.lendable_memory or job.ram > budget:
                continue
            if job.gpu_models and not model_matches(verdict.model, job.gpu_models):
                continue
            launches.append(Launch(job.job_id, uuid, verdict.lendable_memory, job.ram))
            taken.add(job.job_id)
            budget -= job.ram
            break
    return Plan(tuple(reclaims), tuple(launches))
