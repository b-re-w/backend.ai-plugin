"""Map host PIDs to Docker container IDs via /proc/<pid>/cgroup."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

# Matches docker-<id>.scope (systemd driver), /docker/<id> (cgroupfs), and CRI paths.
_rx_container_id = re.compile(r"(?:^|[/-])([0-9a-f]{64})(?:\.scope)?(?:/|$)")


def parse_cgroup(text: str) -> str | None:
    """Return the 64-hex container ID found in a /proc/<pid>/cgroup body, if any."""
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        m = _rx_container_id.search(parts[2])
        if m:
            return m.group(1)
    return None


def cgroup_v2_path(text: str) -> str | None:
    """The unified (v2) cgroup path from a /proc/<pid>/cgroup body."""
    for line in text.splitlines():
        if line.startswith("0::"):
            return line[3:]
    return None


def container_of_pid(pid: int, proc_root: Path = Path("/proc")) -> str | None:
    try:
        return parse_cgroup((proc_root / str(pid) / "cgroup").read_text())
    except OSError:
        return None


def map_pids(pids: Iterable[int], proc_root: Path = Path("/proc")) -> dict[int, str | None]:
    return {pid: container_of_pid(pid, proc_root) for pid in pids}
