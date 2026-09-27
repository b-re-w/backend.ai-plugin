"""
Where each spot session should run (SPEC 2.12). Pure: no NVML, Docker, or cuda-checkpoint.

Rules:
- A spot session runs on at most one GPU, and a GPU hosts at most one spot session.
- It may stay only on a GPU whose verdict is LENT without `must_reclaim`.
- Otherwise it moves to a LENDABLE GPU of the same model among the GPUs attached to it.
- With no such GPU (user decision, nothing is ever killed):
  - a program that installed the out-of-memory handler gets torch.OutOfMemoryError right away
    (`Oom`: its GPU memory limit drops to 1 MiB and a signal raises the error). If it is still on
    the GPU `oom_grace` seconds later it is parked, so it never keeps holding the owner's GPU;
  - any other program is parked at once (checkpointed off the GPU, no error).
  A parked session is restored when a GPU of its model frees up.
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
    oom_ready: bool = True  # at least one process has the out-of-memory signal handler installed


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
class Oom:
    container_id: str
    gpu: str
    reason: str


Action = Move | Park | Restore | Oom


def plan(
    verdicts: Mapping[str, GpuVerdict],
    running: Sequence[RunningSpot],
    parked: Sequence[ParkedSpot],
    *,
    now: float,
    can_checkpoint: bool,
    oom_grace: float,
    oomed: Mapping[str, float] | None = None,
    busy: frozenset[str] = frozenset(),
) -> list[Action]:
    """
    `oomed`: when each container last got the out-of-memory error; `busy`: containers with an
    operation in flight, left alone this tick.
    """
    oomed = oomed or {}
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

    running_ids = {s.container_id for s in running}
    for p in sorted(parked, key=lambda p: p.since):
        if p.container_id in busy or p.container_id in running_ids:
            continue  # partly parked: restore once all its processes are off the GPU
        dst = target(p.src, p.allowed, may_return=True)
        if dst is not None:
            actions.append(Restore(p.container_id, p.src, dst))

    for s in sorted(running, key=lambda s: s.since):
        if s.container_id in busy:
            continue
        src = sorted(s.gpus)[0]
        v = verdicts.get(src)
        if len(s.gpus) != 1:
            reason = f"uses {len(s.gpus)} GPUs; spot allows one"
        elif keep_on.get(src) is not s:
            reason = "another spot session is on this GPU"
        elif v is None:
            reason = "GPU not observed"
        elif v.state is GpuState.LENT and not v.must_reclaim:
            continue
        else:
            reason = "; ".join(v.reasons) or f"GPU is {v.state}"
        dst = target(src, s.allowed) if len(s.gpus) == 1 else None
        last = oomed.get(s.container_id)
        parkable = can_checkpoint and len(s.gpus) == 1
        if dst is not None:
            actions.append(Move(s.container_id, src, dst, reason))
        elif parkable and not s.oom_ready:
            actions.append(Park(s.container_id, src, reason))  # no process could see the error
        elif last is None:
            actions.append(Oom(s.container_id, src, reason))
        elif parkable and now - last >= oom_grace:
            actions.append(Park(s.container_id, src, reason))  # still holding the GPU
        elif not parkable and now - last >= oom_grace:
            actions.append(Oom(s.container_id, src, reason))
    return actions
