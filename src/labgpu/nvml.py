"""
Thin NVML adapter. The only module that imports `pynvml`.

All values returned are plain dataclasses so that callers stay testable without a GPU.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from . import procmap
from .sizes import parse_size


class NvmlError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuInfo:
    index: int
    uuid: str  # "GPU-xxxxxxxx-..." as NVML reports it
    name: str
    total_memory: int
    pci_bus_id: str


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    used_memory: int
    sm_util: int  # percent, max over samples since the previous read; 0 if unsampled
    name: str = ""  # /proc/<pid>/comm, "" if unknown


@dataclass(frozen=True)
class GpuSnapshot:
    info: GpuInfo
    used_memory: int
    gpu_util: int
    processes: tuple[GpuProcess, ...] = field(default_factory=tuple)

    @property
    def free_memory(self) -> int:
        return self.info.total_memory - self.used_memory


class NvmlReader:
    """Stateful reader: remembers the last utilization sample timestamp per GPU."""

    is_fake = False

    def __init__(self) -> None:
        try:
            import pynvml
        except ImportError as e:
            raise NvmlError("nvidia-ml-py is not installed") from e
        self._nvml = pynvml
        self._lock = threading.Lock()
        self._last_sample_ts: dict[int, int] = {}
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            raise NvmlError(f"nvmlInit failed: {e}") from e

    def close(self) -> None:
        try:
            self._nvml.nvmlShutdown()
        except self._nvml.NVMLError:
            pass

    def driver_version(self) -> str:
        return _str(self._nvml.nvmlSystemGetDriverVersion())

    def list_gpus(self) -> list[GpuInfo]:
        nv = self._nvml
        try:
            count = nv.nvmlDeviceGetCount()
            return [self._info(i, nv.nvmlDeviceGetHandleByIndex(i)) for i in range(count)]
        except nv.NVMLError as e:
            raise NvmlError(f"listing GPUs failed: {e}") from e

    def snapshot(self, index: int) -> GpuSnapshot:
        nv = self._nvml
        with self._lock:
            try:
                handle = nv.nvmlDeviceGetHandleByIndex(index)
                info = self._info(index, handle)
                mem = nv.nvmlDeviceGetMemoryInfo(handle)
                util = nv.nvmlDeviceGetUtilizationRates(handle)
                procs = nv.nvmlDeviceGetComputeRunningProcesses(handle)
                sm_by_pid = self._process_sm_util(index, handle)
            except nv.NVMLError as e:
                raise NvmlError(f"GPU {index} snapshot failed: {e}") from e
        processes = tuple(
            GpuProcess(
                pid=p.pid,
                used_memory=int(p.usedGpuMemory or 0),
                sm_util=sm_by_pid.get(p.pid, 0),
                name=procmap.process_name(p.pid),
            )
            for p in procs
        )
        return GpuSnapshot(info, int(mem.used), int(util.gpu), processes)

    def container_of_pids(self, pids: list[int]) -> dict[int, str | None]:
        return procmap.map_pids(pids)

    def _info(self, index: int, handle: object) -> GpuInfo:
        nv = self._nvml
        mem = nv.nvmlDeviceGetMemoryInfo(handle)
        return GpuInfo(
            index=index,
            uuid=_str(nv.nvmlDeviceGetUUID(handle)),
            name=_str(nv.nvmlDeviceGetName(handle)),
            total_memory=int(mem.total),
            pci_bus_id=_str(nv.nvmlDeviceGetPciInfo(handle).busId),
        )

    def _process_sm_util(self, index: int, handle: object) -> dict[int, int]:
        nv = self._nvml
        last_ts = self._last_sample_ts.get(index, 0)
        try:
            samples = nv.nvmlDeviceGetProcessUtilization(handle, last_ts)
        except nv.NVMLError_NotFound:
            # No samples since last_ts: nobody has run a kernel.
            return {}
        result: dict[int, int] = {}
        for s in samples:
            result[s.pid] = max(result.get(s.pid, 0), int(s.smUtil))
            last_ts = max(last_ts, int(s.timeStamp))
        self._last_sample_ts[index] = last_ts
        return result


def _str(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


FAKE_ENV = "LABGPU_FAKE_NVML"


class FakeNvmlReader:
    """
    Reads GPUs from a JSON file on every call (SPEC 1.12). For development and tests only:
    it lets a machine without the target GPUs impersonate another server's layout.
    """

    is_fake = True

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._load()  # fail early on a bad file

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError) as e:
            raise NvmlError(f"fake NVML file {self.path}: {e}") from e

    def close(self) -> None:
        pass

    def driver_version(self) -> str:
        return str(self._load().get("driver", "fake"))

    def list_gpus(self) -> list[GpuInfo]:
        return [self._info(i, g) for i, g in enumerate(self._load()["gpus"])]

    def snapshot(self, index: int) -> GpuSnapshot:
        gpus = self._load()["gpus"]
        if index >= len(gpus):
            raise NvmlError(f"GPU {index} not found")
        g = gpus[index]
        if g.get("fail"):
            raise NvmlError(f"GPU {index}: simulated failure")
        procs = tuple(
            GpuProcess(
                int(p["pid"]), parse_size(p.get("mem", 0)), int(p.get("sm", 0)), str(p.get("name", ""))
            )
            for p in g.get("processes", [])
        )
        used = parse_size(g["used"]) if "used" in g else sum(p.used_memory for p in procs)
        return GpuSnapshot(self._info(index, g), used, int(g.get("util", 0)), procs)

    def container_of_pids(self, pids: list[int]) -> dict[int, str | None]:
        mapping: dict[int, str | None] = {}
        for g in self._load()["gpus"]:
            for p in g.get("processes", []):
                mapping[int(p["pid"])] = p.get("container")
        return {pid: mapping.get(pid) for pid in pids}

    @staticmethod
    def _info(index: int, g: dict) -> GpuInfo:
        return GpuInfo(
            index=index,
            uuid=g["uuid"],
            name=g["name"],
            total_memory=parse_size(g["memory"]),
            pci_bus_id=g.get("pci", f"00000000:{index + 0x10:02X}:00.0"),
        )


def open_reader() -> NvmlReader | FakeNvmlReader:
    """The real NVML reader, or the fake one when LABGPU_FAKE_NVML is set."""
    fake = os.environ.get(FAKE_ENV)
    if fake:
        return FakeNvmlReader(fake)
    return NvmlReader()
