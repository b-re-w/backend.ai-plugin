"""Assemble per-GPU observations from NVML, Docker, and /proc (SPEC 2.3)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence

from ..nvml import GpuInfo, GpuSnapshot, NvmlError
from .docker import OwnerContainer, SpotContainer
from .model import ClassifiedProcess, GpuObservation, ProcKind

log = logging.getLogger("ai.backend.labgpu.spot.observer")


def resolve_gpu_refs(refs: Sequence[str], gpus: Sequence[GpuInfo]) -> set[str]:
    """Turn container GPU references (UUIDs, NVML indices, "all") into UUIDs."""
    by_index = {str(g.index): g.uuid for g in gpus}
    uuids = {g.uuid for g in gpus}
    result: set[str] = set()
    for ref in refs:
        if ref == "all":
            return set(uuids)
        if ref in uuids:
            result.add(ref)
        elif ref in by_index:
            result.add(by_index[ref])
        else:
            log.warning("unresolvable GPU reference %r", ref)
    return result


def build_observation(
    snap: GpuSnapshot,
    *,
    pid_to_container: Mapping[int, str | None],
    owners_on_gpu: Mapping[str, OwnerContainer],
    all_owner_ids: set[str],
    spot_gpu_by_container: Mapping[str, str | None],
    owner_cpu: Mapping[str, float | None],
) -> GpuObservation:
    """Classify each GPU process as owner, spot, or unknown for this GPU."""
    uuid = snap.info.uuid
    processes = []
    for p in snap.processes:
        cid = pid_to_container.get(p.pid)
        if cid is not None and cid in spot_gpu_by_container:
            kind = ProcKind.SPOT if spot_gpu_by_container[cid] == uuid else ProcKind.UNKNOWN
        elif cid is not None and cid in owners_on_gpu:
            kind = ProcKind.OWNER
        else:
            # Includes host processes and owner containers using a GPU they were not given.
            kind = ProcKind.UNKNOWN
            if cid in all_owner_ids:
                log.warning("container %s uses GPU %s it was not attached to", cid[:12], uuid)
        processes.append(ClassifiedProcess(p.pid, kind, cid, p.used_memory, p.sm_util))

    cpu_values = [owner_cpu.get(cid) for cid in owners_on_gpu]
    measured = [v for v in cpu_values if v is not None]
    return GpuObservation(
        uuid=uuid,
        index=snap.info.index,
        ok=True,
        model=snap.info.name,
        total_memory=snap.info.total_memory,
        used_memory=snap.used_memory,
        processes=tuple(processes),
        owner_containers=frozenset(owners_on_gpu),
        owner_cpu_cores=sum(measured) if measured else None,
    )


def observe(
    gpus: Sequence[GpuInfo],
    snapshot: Callable[[int], GpuSnapshot],
    owners: Sequence[OwnerContainer],
    spots: Sequence[SpotContainer],
    pid_mapper: Callable[[list[int]], Mapping[int, str | None]],
    owner_cpu: Mapping[str, float | None],
) -> list[GpuObservation]:
    owners_by_gpu: dict[str, dict[str, OwnerContainer]] = {g.uuid: {} for g in gpus}
    for o in owners:
        for uuid in resolve_gpu_refs(o.gpu_refs, gpus):
            owners_by_gpu[uuid][o.id] = o
    spot_gpu = {s.id: s.gpu_uuid for s in spots if s.running}
    all_owner_ids = {o.id for o in owners}

    observations = []
    for g in gpus:
        try:
            snap = snapshot(g.index)
            if snap.info.uuid != g.uuid:
                raise NvmlError(f"GPU index {g.index} changed identity")
            pid_map = pid_mapper([p.pid for p in snap.processes])
        except (NvmlError, OSError) as e:
            observations.append(
                GpuObservation(uuid=g.uuid, index=g.index, ok=False, model=g.name, error=str(e))
            )
            continue
        observations.append(
            build_observation(
                snap,
                pid_to_container=pid_map,
                owners_on_gpu=owners_by_gpu[g.uuid],
                all_owner_ids=all_owner_ids,
                spot_gpu_by_container=spot_gpu,
                owner_cpu=owner_cpu,
            )
        )
    return observations
