"""
Per-session GPU lending figures from the spot controller's status file (SPEC 1.13, 2.9.1).

Pure module: no Backend.AI, NVML, or Docker imports.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

STATUS_FILE = "status.json"  # inside the monitor state dir (SPEC 2.9.1)
DEFAULT_MAX_AGE = 60.0


# Monitor states in which a GPU can host a spot session (SPEC 2.4, 2.12).
SPOT_STATES = frozenset({"LENDABLE", "LENT"})


@dataclass(frozen=True)
class GpuLending:
    lent: bool
    since: float | None
    state: str = ""
    lendable_memory: int = 0  # bytes a spot session may use on this GPU right now
    spot_share: float = 0.0  # total share of the spot sessions on it (0..1)
    spot_capacity: int = 0  # bytes all spot sessions on it may use together (SPEC 2.9.1)
    spots: tuple[tuple[float, float], ...] = ()  # (share, first seen) of each spot session on it


@dataclass(frozen=True)
class SessionLending:
    lent: int  # how many of the session's GPUs are lent right now
    total: int  # how many GPUs of this plugin the session holds
    since: float  # earliest lend start (unix seconds), 0 when nothing is lent


def parse_status(text: str, now: float, max_age: float = DEFAULT_MAX_AGE) -> dict[str, GpuLending] | None:
    """GPU UUID -> lending state, or None when the file is stale or malformed (report nothing)."""
    try:
        data = json.loads(text)
        updated = float(data["updated_at"])
        gpus = data["gpus"]
    except (ValueError, KeyError, TypeError):
        return None
    if now - updated > max_age:
        return None
    # With spot turned off in the monitor, no GPU offers room for spot sessions.
    spot_on = data.get("spot_enabled", True) is not False
    result: dict[str, GpuLending] = {}
    for g in gpus:
        uuid = g.get("uuid")
        if not uuid:
            continue
        since = g.get("lent_since")
        result[uuid] = GpuLending(
            lent=g.get("lent_job") is not None,
            since=float(since) if since else None,
            state=str(g.get("state", "")) if spot_on else "",
            lendable_memory=int(g.get("lendable_memory") or 0),
            spot_share=float(g.get("spot_share") or (1.0 if g.get("lent_job") else 0.0)),
            spot_capacity=int(g.get("spot_capacity") or g.get("lendable_memory") or 0),
            spots=tuple(
                (float(sp.get("share") or 1.0), float(sp.get("seen") or 0.0))
                for sp in g.get("spots") or () if isinstance(sp, dict)
            ),
        )
    return result


def read_status(path: Path, now: float, max_age: float = DEFAULT_MAX_AGE) -> dict[str, GpuLending] | None:
    try:
        return parse_status(path.read_text(), now, max_age)
    except OSError:
        return None


def session_lending(
    uuids_by_container: Mapping[str, Iterable[str]],
    own_uuids: Collection[str],
    status: Mapping[str, GpuLending],
) -> dict[str, SessionLending]:
    """
    Figures for every container holding at least one of this plugin's GPUs. A GPU missing from the
    status file counts as not lent: the controller reports every GPU it can see.
    """
    result: dict[str, SessionLending] = {}
    for cid, uuids in uuids_by_container.items():
        mine = [u for u in uuids if u in own_uuids]
        if not mine:
            continue
        lent = [status[u] for u in mine if u in status and status[u].lent]
        starts = [g.since for g in lent if g.since]
        result[cid] = SessionLending(lent=len(lent), total=len(mine), since=min(starts) if starts else 0.0)
    return result


def unreported(
    handed: Iterable[tuple[str, float, float]], status: Mapping[str, GpuLending] | None
) -> dict[str, float]:
    """
    Shares this plugin handed out (uuid, time, share) that the status file does not list yet, per
    GPU. A listed spot on the same GPU with the same share, first seen after the hand-out, accounts
    for it (one to one), so no share is counted twice (SPEC 2.12).
    """
    listed = {u: list(g.spots) for u, g in (status or {}).items()}
    taken: dict[str, float] = {}
    for uuid, t, share in sorted(handed, key=lambda h: h[1]):
        match = next(
            (i for i, (sh, seen) in enumerate(listed.get(uuid, [])) if abs(sh - share) < 1e-6 and seen >= t),
            None,
        )
        if match is not None:
            del listed[uuid][match]
        else:
            taken[uuid] = taken.get(uuid, 0.0) + share
    return taken


def spot_capacity(own_uuids: Collection[str], status: Mapping[str, GpuLending] | None) -> int:
    """
    How many spot sessions this GPU model can hold right now: one per GPU that is lendable or
    already lent (SPEC 2.12). Unknown status means none (fail safe).
    """
    if status is None:
        return 0
    return sum(1 for u in own_uuids if u in status and status[u].state in SPOT_STATES)


def _rooms(
    own_uuids: Iterable[str], status: Mapping[str, GpuLending], taken: Mapping[str, float]
) -> list[tuple[str, float, int]]:
    """(uuid, share left, spot capacity) for every GPU that can host spot sessions now."""
    return [
        (u, 1.0 - status[u].spot_share - taken.get(u, 0.0), status[u].spot_capacity)
        for u in own_uuids
        if u in status and status[u].state in SPOT_STATES
    ]


def pick_spot_gpu(
    own_uuids: Iterable[str],
    status: Mapping[str, GpuLending] | None,
    share: float = 1.0,
    taken: Mapping[str, float] | None = None,
) -> tuple[str, int] | None:
    """
    The GPU a new spot session with `share` gets (SPEC 2.12): LENDABLE or LENT with room for the
    share (after `taken`, shares this plugin just handed out), the one left fullest, ties to the
    larger spot capacity. Returns (uuid, spot capacity in bytes) or None when nothing fits.
    """
    if status is None:
        return None
    fits = [r for r in _rooms(own_uuids, status, taken or {}) if r[1] + 1e-6 >= share]
    if not fits:
        return None
    u, _left, capacity = min(fits, key=lambda r: (r[1] - share, -r[2]))
    return u, capacity


def roomiest_spot_gpu(
    own_uuids: Iterable[str], status: Mapping[str, GpuLending] | None, taken: Mapping[str, float] | None = None
) -> tuple[str, int, float] | None:
    """When no GPU fits the share (fragmented): the GPU with the most share left, its capacity and room."""
    if status is None:
        return None
    rooms = [r for r in _rooms(own_uuids, status, taken or {}) if r[1] > 1e-6]
    if not rooms:
        return None
    u, left, capacity = max(rooms, key=lambda r: (r[1], r[2]))
    return u, capacity, left


def uuids_from_env(env: Iterable[str]) -> list[str]:
    """The GPUs a session holds, from its container env (LABGPU_DEVICE_UUIDS=...)."""
    for item in env:
        key, _, value = item.partition("=")
        if key == "LABGPU_DEVICE_UUIDS":
            return [u for u in value.split(",") if u]
    return []
