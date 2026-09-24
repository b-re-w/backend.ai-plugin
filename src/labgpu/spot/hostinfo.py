"""Host-level readings: available RAM and per-container CPU usage (cgroup v2)."""

from __future__ import annotations

from pathlib import Path

from .. import procmap


def parse_meminfo_available(text: str) -> int:
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("MemAvailable not found")


def mem_available(meminfo: Path = Path("/proc/meminfo")) -> int:
    return parse_meminfo_available(meminfo.read_text())


def parse_cpu_usage_usec(text: str) -> int:
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        if key == "usage_usec":
            return int(value)
    raise ValueError("usage_usec not found")


class CpuSampler:
    """CPU cores used by a container between two samples, from its cgroup v2 cpu.stat."""

    def __init__(
        self,
        proc_root: Path = Path("/proc"),
        cgroup_root: Path = Path("/sys/fs/cgroup"),
    ) -> None:
        self._proc_root = proc_root
        self._cgroup_root = cgroup_root
        self._last: dict[str, tuple[float, int]] = {}

    def sample(self, container_id: str, pid: int, now: float) -> float | None:
        """Cores used since the previous call; None on the first call or if unreadable."""
        try:
            cg_text = (self._proc_root / str(pid) / "cgroup").read_text()
            cg_path = procmap.cgroup_v2_path(cg_text)
            if cg_path is None:
                return None
            stat = (self._cgroup_root / cg_path.lstrip("/") / "cpu.stat").read_text()
            usage = parse_cpu_usage_usec(stat)
        except (OSError, ValueError):
            self._last.pop(container_id, None)
            return None
        prev = self._last.get(container_id)
        self._last[container_id] = (now, usage)
        if prev is None or now <= prev[0]:
            return None
        return (usage - prev[1]) / 1e6 / (now - prev[0])

    def forget_except(self, live_ids: set[str]) -> None:
        for cid in list(self._last):
            if cid not in live_ids:
                del self._last[cid]
