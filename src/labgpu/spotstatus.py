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


def spot_capacity(own_uuids: Collection[str], status: Mapping[str, GpuLending] | None) -> int:
    """
    How many spot sessions this GPU model can hold right now: one per GPU that is lendable or
    already lent (SPEC 2.12). Unknown status means none (fail safe).
    """
    if status is None:
        return 0
    return sum(1 for u in own_uuids if u in status and status[u].state in SPOT_STATES)


def pick_spot_gpu(
    own_uuids: Iterable[str], status: Mapping[str, GpuLending] | None, taken: Collection[str] = ()
) -> tuple[str, int] | None:
    """
    The GPU a new spot session gets (SPEC 2.12): LENDABLE, not lent, not just handed out
    (`taken`), with the most lendable memory. Returns (uuid, lendable bytes) or None.
    """
    if status is None:
        return None
    candidates = [
        (u, status[u].lendable_memory)
        for u in own_uuids
        if u in status and status[u].state == "LENDABLE" and not status[u].lent and u not in taken
    ]
    return max(candidates, key=lambda c: c[1]) if candidates else None


def uuids_from_env(env: Iterable[str]) -> list[str]:
    """The GPUs a session holds, from its container env (LABGPU_DEVICE_UUIDS=...)."""
    for item in env:
        key, _, value = item.partition("=")
        if key == "LABGPU_DEVICE_UUIDS":
            return [u for u in value.split(",") if u]
    return []
