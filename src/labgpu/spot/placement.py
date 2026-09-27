"""
Where each spot session should run (SPEC 2.12). Pure: no NVML, Docker, or cuda-checkpoint.

Rules:
- A spot session runs on at most one GPU, with a share of it (0 < share <= 1). A GPU hosts spot
  sessions up to a total share of 1; over that, the latest arrivals leave.
- It may stay only on a GPU whose verdict is LENT without `must_reclaim`.
- Otherwise it moves to a GPU of the same model among the GPUs attached to it that is LENDABLE or
  LENT without `must_reclaim` and has room for its share (the fullest one that fits, so whole GPUs
  stay free for larger shares).
- With no such GPU (user decision, nothing is ever killed):
  - a program that installed the out-of-memory handler gets torch.OutOfMemoryError right away
    (`Oom`: its GPU memory limit drops to 1 MiB and a signal raises the error). If it is still on
    the GPU `oom_grace` seconds later it is parked, so it never keeps holding the owner's GPU;
  - any other program is parked at once (checkpointed off the GPU, no error).
  A parked session is restored when a GPU of its model frees up.
- Parked sessions get free GPUs before running sessions that must move.
- A spot session with no GPU process yet still holds its share on the GPU it was given, within
  the room the running sessions leave (user decision), so no one else is placed there.
- Consolidation only when a session must move and its share fits nowhere (user decision): smaller
  spots on one GPU are moved elsewhere so that GPU has room; the waiting session moves next tick.
  Spread-out spots are otherwise left alone, since every move pauses a program for seconds.
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
    assigned: str | None = None  # the GPU this session was given (None: not known)
    share: float = 1.0  # its share of one GPU
    short: bool = False  # its memory limit is below its share (placed on a fragmented GPU)


@dataclass(frozen=True)
class ParkedSpot:
    container_id: str
    src: str  # GPU it was checkpointed from
    allowed: tuple[str, ...]
    since: float  # when it was parked
    share: float = 1.0


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


@dataclass(frozen=True)
class Hold:
    """Processes on GPUs the session was not given: off those GPUs at once, not restored."""

    container_id: str
    gpus: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class Resize:
    """Raise a session's memory limit to its full share once its GPU has room (SPEC 2.12 rule 3-1)."""

    container_id: str
    gpu: str


Action = Move | Park | Restore | Oom | Hold | Resize


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
    idle: Mapping[str, float] | None = None,
) -> list[Action]:
    """
    `oomed`: when each container last got the out-of-memory error; `busy`: containers with an
    operation in flight, left alone this tick; `idle`: share held on each GPU by spot sessions
    with no GPU process yet.
    """
    oomed = oomed or {}
    actions: list[Action] = []
    eps = 1e-6

    # Share already on each GPU, oldest sessions first; a session that would push a GPU over a
    # total of 1 is "over" and must leave (e.g. two sessions placed at the same time).
    used: dict[str, float] = {}
    over: set[str] = set()
    for s in sorted(running, key=lambda s: s.since):
        if len(s.gpus) == 1:
            g = next(iter(s.gpus))
            if used.get(g, 0.0) + s.share > 1 + eps:
                over.add(s.container_id)
            else:
                used[g] = used.get(g, 0.0) + s.share
    # Idle sessions come after running ones: they never push a running session off its GPU.
    for g, share in (idle or {}).items():
        used[g] = used.get(g, 0.0) + max(min(share, 1.0 - used.get(g, 0.0)), 0.0)

    def room(u: str) -> float:
        return 1.0 - used.get(u, 0.0)

    def target(src: str, allowed: Sequence[str], share: float, *, may_return: bool = False) -> str | None:
        """The fullest GPU of src's model with room for `share`; a parked spot may go back to src."""
        model = verdicts[src].model if src in verdicts else None
        candidates = [
            verdicts[u]
            for u in allowed
            if u in verdicts
            and (u != src or may_return)
            and (verdicts[u].state is GpuState.LENDABLE
                 or (verdicts[u].state is GpuState.LENT and not verdicts[u].must_reclaim))
            and (model is None or verdicts[u].model == model)
            and room(u) + eps >= share
        ]
        if not candidates:
            return None
        best = min(candidates, key=lambda v: (room(v.uuid) - share, -(v.spot_capacity or v.lendable_memory)))
        used[best.uuid] = used.get(best.uuid, 0.0) + share  # promised this tick
        return best.uuid

    def hosts(u: str) -> bool:
        v = verdicts.get(u)
        return v is not None and (v.state is GpuState.LENDABLE or (v.state is GpuState.LENT and not v.must_reclaim))

    on_gpu: dict[str, list[RunningSpot]] = {}
    for r in running:
        if len(r.gpus) == 1 and r.container_id not in over:
            on_gpu.setdefault(next(iter(r.gpus)), []).append(r)
    moving: set[str] = set()  # sessions given a move this tick to make room for another

    def make_room(s: RunningSpot, src: str) -> list[Move]:
        """Moves that free one GPU enough for `s` (fewest moves wins), or [] if none can."""
        model = verdicts[src].model if src in verdicts else None
        best: tuple[str, list[Move], dict[str, float]] | None = None
        for u in s.allowed:
            # src itself counts when it may still host spots: a session that arrived over the top
            # of a fragmented GPU can stay once the smaller ones leave it.
            if not hosts(u) or (model is not None and verdicts[u].model != model):
                continue
            sim = dict(used)
            moves: list[Move] = []
            for h in sorted(on_gpu.get(u, []), key=lambda h: h.share):
                if 1.0 - sim.get(u, 0.0) + eps >= s.share:
                    break
                if h.container_id in busy or h.container_id in moving or h.container_id == s.container_id:
                    continue
                dests = [
                    w for w in h.allowed
                    if w not in (u, src) and hosts(w) and verdicts[w].model == verdicts[u].model
                    and 1.0 - sim.get(w, 0.0) + eps >= h.share
                ]
                if not dests:
                    continue
                w = min(dests, key=lambda w: (1.0 - sim.get(w, 0.0) - h.share,
                                              -(verdicts[w].spot_capacity or verdicts[w].lendable_memory)))
                sim[w] = sim.get(w, 0.0) + h.share
                sim[u] = sim.get(u, 0.0) - h.share
                moves.append(Move(h.container_id, u, w, f"making room for {s.container_id[:12]}"))
            if moves and 1.0 - sim.get(u, 0.0) + eps >= s.share and (best is None or len(moves) < len(best[1])):
                best = (u, moves, sim)
        if best is None:
            return []
        u, moves, sim = best
        used.clear()
        used.update(sim)
        used[u] = used.get(u, 0.0) + s.share  # held for the waiting session
        moving.update(m.container_id for m in moves)
        return moves

    running_ids = {s.container_id for s in running}
    for p in sorted(parked, key=lambda p: p.since):
        if p.container_id in busy or p.container_id in running_ids:
            continue  # partly parked: restore once all its processes are off the GPU
        dst = target(p.src, p.allowed, p.share, may_return=True)
        if dst is not None:
            actions.append(Restore(p.container_id, p.src, dst))

    for s in sorted(running, key=lambda s: s.since):
        if s.container_id in busy or s.container_id in moving:
            continue
        # The spot side is not trusted (SPEC 2.12): a process on any GPU but the one the session
        # was given, e.g. after changing CUDA_VISIBLE_DEVICES, leaves that GPU right away.
        off = tuple(sorted(s.gpus - {s.assigned})) if s.assigned else ()
        if off:
            actions.append(Hold(s.container_id, off, f"on GPU(s) it was not given: {', '.join(off)}"))
            continue
        src = sorted(s.gpus)[0]
        v = verdicts.get(src)
        if len(s.gpus) != 1:
            reason = f"uses {len(s.gpus)} GPUs; spot allows one"
        elif s.container_id in over:
            reason = "spot sessions on this GPU exceed one GPU in total"
        elif v is None:
            reason = "GPU not observed"
        elif v.state is GpuState.LENT and not v.must_reclaim:
            if s.short:
                actions.append(Resize(s.container_id, src))
            continue
        else:
            reason = "; ".join(v.reasons) or f"GPU is {v.state}"
        dst = target(src, s.allowed, s.share) if len(s.gpus) == 1 else None
        last = oomed.get(s.container_id)
        parkable = can_checkpoint and len(s.gpus) == 1
        if dst is not None:
            actions.append(Move(s.container_id, src, dst, reason))
            continue
        room_moves = make_room(s, src) if parkable else []
        if room_moves:
            actions.extend(room_moves)  # it moves next tick, into the room these make
        elif parkable and not s.oom_ready:
            actions.append(Park(s.container_id, src, reason))  # no process could see the error
        elif last is None:
            actions.append(Oom(s.container_id, src, reason))
        elif parkable and now - last >= oom_grace:
            actions.append(Park(s.container_id, src, reason))  # still holding the GPU
        elif not parkable and now - last >= oom_grace:
            actions.append(Oom(s.container_id, src, reason))
    return actions
