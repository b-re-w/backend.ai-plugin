"""
Where each spot session should run (SPEC 2.12). Pure: no NVML, Docker, or cuda-checkpoint.

Rules:
- A spot session runs on at most one GPU, and a GPU hosts at most one spot session.
- It may stay only on a GPU whose verdict is LENT without `must_reclaim`.
- Otherwise it moves to a LENDABLE GPU of the same model among the GPUs attached to it.
- With no such GPU it is parked (checkpointed off the GPU) and waits `park_seconds` for one;
  after that, or when it cannot be checkpointed at all, it is evicted.
- Parked sessions get free GPUs before running sessions that must move.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .model import GpuState, GpuVerdict


@dataclass(frozen=True)
class RunningSpot:
    container_id: str
    gpus: frozenset[str]  # GPUs its processes are on right now
    allowed: tuple[str, ...]  # GPUs attached to the container
    since: float  # when the monitor first saw it on its current GPU


@dataclass(frozen=True)
class ParkedSpot:
    container_id: str
    src: str  # GPU it was checkpointed from
    allowed: tuple[str, ...]
    since: float  # when it was parked


@dataclass(frozen=True)
class Move:
    container_id: str
    src: str
    dst: str
    reason: str


@dataclass(frozen=True)
class Park:
    container_id: str
    src: str
    reason: str


@dataclass(frozen=True)
class Restore:
    container_id: str
    src: str
    dst: str


@dataclass(frozen=True)
class Evict:
    container_id: str
    gpu: str | None  # None when parked
    reason: str


Action = Move | Park | Restore | Evict


def plan(
    verdicts: Mapping[str, GpuVerdict],
    running: Sequence[RunningSpot],
    parked: Sequence[ParkedSpot],
    *,
    now: float,
    can_checkpoint: bool,
    park_seconds: float,
    busy: frozenset[str] = frozenset(),
) -> list[Action]:
    """`busy`: containers with an operation in flight; they are left alone this tick."""
    actions: list[Action] = []
    taken: set[str] = set()  # GPUs promised to a spot this tick

    # Keep the longest-running spot on each GPU; later arrivals on the same GPU must leave.
    keep_on: dict[str, RunningSpot] = {}
    for s in sorted(running, key=lambda s: s.since):
        if len(s.gpus) == 1:
            g = next(iter(s.gpus))
            keep_on.setdefault(g, s)
    occupied = {g for s in running for g in s.gpus}

    def target(src: str, allowed: Sequence[str], *, may_return: bool = False) -> str | None:
        """A free LENDABLE GPU of src's model; a parked spot may go back to src itself."""
        model = verdicts[src].model if src in verdicts else None
        candidates = [
            verdicts[u]
            for u in allowed
            if u in verdicts
            and (u != src or may_return)
            and u not in taken
            and u not in occupied
            and verdicts[u].state is GpuState.LENDABLE
            and (model is None or verdicts[u].model == model)
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda v: v.lendable_memory)
        taken.add(best.uuid)
        return best.uuid

    for p in sorted(parked, key=lambda p: p.since):
        if p.container_id in busy:
            continue
        dst = target(p.src, p.allowed, may_return=True)
        if dst is not None:
            actions.append(Restore(p.container_id, p.src, dst))
        elif now - p.since >= park_seconds:
            actions.append(Evict(p.container_id, None, f"no free GPU for {park_seconds:.0f}s"))

    for s in sorted(running, key=lambda s: s.since):
        if s.container_id in busy:
            continue
        if len(s.gpus) != 1:
            actions.append(Evict(s.container_id, None, f"uses {len(s.gpus)} GPUs; spot allows one"))
            continue
        src = next(iter(s.gpus))
        v = verdicts.get(src)
        if keep_on.get(src) is not s:
            reason = "another spot session is on this GPU"
        elif v is None:
            reason = "GPU not observed"
        elif v.state is GpuState.LENT and not v.must_reclaim:
            continue
        else:
            reason = "; ".join(v.reasons) or f"GPU is {v.state}"
        dst = target(src, s.allowed)
        if dst is not None:
            actions.append(Move(s.container_id, src, dst, reason))
        elif can_checkpoint:
            actions.append(Park(s.container_id, src, reason))
        else:
            actions.append(Evict(s.container_id, src, reason))
    return actions
