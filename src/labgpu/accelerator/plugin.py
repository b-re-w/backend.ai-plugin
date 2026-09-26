"""
Backend.AI accelerator plugins exposing CUDA GPUs as fractional `<key>.shares`, enforced by
HAMi-core (SPEC section 1).

`cuda_frac` takes every GPU under the key `cuda`; `gpu_slot_1..4` each take the GPUs matching
a model pattern under a configured key, so one agent can offer one slot per GPU model (SPEC 1.11).

Only this module imports `ai.backend.*`; all arithmetic lives in `labgpu.fraction`.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any

import aiodocker
from aiodocker.exceptions import DockerError

from ai.backend.agent.resources import (
    AbstractAllocMap,
    AbstractComputeDevice,
    AbstractComputePlugin,
    DeviceSlotInfo,
    DiscretePropertyAllocMap,
    FractionAllocMap,
)
from ai.backend.agent.stats import (
    ContainerMeasurement,
    Measurement,
    MetricTypes,
    NodeMeasurement,
    ProcessMeasurement,
    StatContext,
)
from ai.backend.agent.types import MountInfo
from ai.backend.common.types import (
    AcceleratorMetadata,
    BinarySize,
    DeviceId,
    DeviceModelInfo,
    DeviceName,
    MetricKey,
    SlotName,
    SlotTypes,
)

try:
    from ai.backend.agent.resources import AllocationStrategy  # type: ignore[attr-defined]
except ImportError:
    from ai.backend.agent.alloc_map import AllocationStrategy

try:
    from ai.backend.agent.resources import get_resource_spec_from_container  # type: ignore
except ImportError:
    from ai.backend.agent.docker.resources import get_resource_spec_from_container

from .. import __version__, devalloc, spotstatus
from ..fraction import build_hami_environ, compute_limits
from ..paths import agent_state_dir, default_hook_path
from ..nvml import FakeNvmlReader, GpuInfo, NvmlError, NvmlReader, open_reader
from ..selection import GpuSelector, claim_gpus, validate_key

log = logging.getLogger("ai.backend.labgpu.accelerator")

PROCESSING_UNITS = 100  # SM percent; NVML does not expose an SM count portably
DEVICE_CAPABILITIES = [["utility", "compute", "video", "graphics", "display"]]


class AllocationMode(StrEnum):
    FRACTIONAL = "fractional"
    DISCRETE = "discrete"


class CUDAFracDevice(AbstractComputeDevice):
    model_name: str
    uuid: str

    def __init__(self, model_name: str, uuid: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.uuid = uuid

    def __repr__(self) -> str:
        return (
            f"CUDAFracDevice(id={self.device_id}, uuid={self.uuid}, model={self.model_name}, "
            f"mem={self.memory_size}, numa={self.numa_node}, pci={self.hw_location})"
        )


class PluginNotConfigured(RuntimeError):
    pass


class LabGpuPlugin(AbstractComputePlugin):
    """Common implementation; subclasses fix the entry-point name and the default key."""

    config_watch_enabled = False

    entry_name: str = ""
    default_key: str | None = None

    key = DeviceName("cuda")
    slot_types: Sequence[tuple[SlotName, SlotTypes]] = ()
    exclusive_slot_types: set[str] = set()
    display_name: str | None = None
    display_unit: str | None = None
    selector: GpuSelector = GpuSelector()

    enabled: bool = True
    mode: AllocationMode = AllocationMode.FRACTIONAL
    fraction_enforced: bool = False
    shares_per_device: Decimal = Decimal(1)
    quantum_size: Decimal = Decimal("0.05")
    allocation_strategy: str = "fill"
    hook_path: Path = default_hook_path()
    reserved_memory: int = 0
    sm_limit: bool = True

    _nvml: NvmlReader | FakeNvmlReader | None = None
    spot_status_path: Path = Path("./var/lib/backend.ai/labgpu") / spotstatus.STATUS_FILE
    spot_status_max_age: float = spotstatus.DEFAULT_MAX_AGE
    _container_gpus: dict[str, list[str]] | None = None
    _devices: list[CUDAFracDevice] | None = None

    @property
    def slot_shares(self) -> SlotName:
        return SlotName(f"{self.key}.shares")

    @property
    def slot_device(self) -> SlotName:
        return SlotName(f"{self.key}.device")

    @property
    def is_fake(self) -> bool:
        return bool(getattr(self._nvml, "is_fake", False))

    async def init(self, context: Any | None = None) -> None:
        self._read_config(self.plugin_config)
        try:
            self._nvml = open_reader()
        except NvmlError as e:
            log.error("[%s] NVML unavailable (%s); disabled.", self.entry_name, e)
            self.enabled = False
            self._set_slot_types()
            return
        if self.is_fake:
            log.warning(
                "[%s] FAKE NVML mode (%s): GPUs are simulated and NOT attached to containers.",
                self.entry_name,
                getattr(self._nvml, "path", "?"),
            )
        elif not await self._has_nvidia_runtime():
            self.enabled = False
            self._set_slot_types()
            return
        try:
            devices = await self.list_devices()
        except NvmlError as e:
            log.error("[%s] NVML unavailable (%s); disabled.", self.entry_name, e)
            self.enabled = False
            self._set_slot_types()
            return
        # SPEC 1.11: refuse to share a GPU with another labgpu plugin in this agent.
        claim_gpus([d.uuid for d in devices], self.entry_name)

        if self.mode == AllocationMode.FRACTIONAL:
            if self.hook_path.is_file():
                self.fraction_enforced = True
            else:
                # SPEC 1.4: never hand out fractions that cannot be enforced.
                log.error(
                    "HAMi-core library %s not found; fractional limits cannot be enforced. "
                    "Falling back to whole-GPU (%s) allocation.",
                    self.hook_path,
                    self.slot_device,
                )
                self.mode = AllocationMode.DISCRETE
        self._set_slot_types()
        log.info(
            "[%s] labgpu %s: key=%s mode=%s enforced=%s devices=%s",
            self.entry_name,
            __version__,
            self.key,
            self.mode,
            self.fraction_enforced,
            devices,
        )

    async def _has_nvidia_runtime(self) -> bool:
        try:
            async with aiodocker.Docker() as docker:
                info = await docker.system.info()
        except DockerError as e:
            log.error("[%s] cannot reach Docker (%r); disabled.", self.entry_name, e)
            return False
        if "nvidia" not in info.get("Runtimes", {}):
            log.error("[%s] NVIDIA container runtime not found; disabled.", self.entry_name)
            return False
        return True

    def _set_slot_types(self) -> None:
        self.slot_types = ((self._slot, SlotTypes.COUNT),)
        self.exclusive_slot_types = {str(self.slot_device), str(self.slot_shares)}

    def _read_config(self, cfg: Mapping[str, Any]) -> None:
        raw_key = cfg.get("key", self.default_key)
        if not raw_key:
            raise PluginNotConfigured(
                f"{self.entry_name}: set config/plugins/accelerator/{self.entry_name}/key to use it"
            )
        self.key = DeviceName(validate_key(str(raw_key)))
        self.selector = GpuSelector.from_config(dict(cfg))
        self.display_name = cfg.get("display_name")
        self.display_unit = cfg.get("display_unit")
        self.mode = AllocationMode(cfg.get("allocation_mode", AllocationMode.FRACTIONAL))
        self.shares_per_device = Decimal(cfg.get("shares_per_device", "1"))
        self.quantum_size = Decimal(cfg.get("quantum_size", "0.05"))
        self.allocation_strategy = str(cfg.get("allocation_strategy", "fill")).lower()
        if self.allocation_strategy not in ("fill", "evenly"):
            raise ValueError(f"invalid allocation_strategy: {self.allocation_strategy}")
        self.hook_path = Path(cfg["hook_path"]) if cfg.get("hook_path") else default_hook_path()
        self.reserved_memory = int(cfg.get("reserved_memory", "0"))
        self.sm_limit = str(cfg.get("sm_limit", "true")).lower() in ("1", "true", "yes")
        self.spot_status_path = (
            Path(cfg["spot_status_path"])
            if cfg.get("spot_status_path")
            else agent_state_dir(self.local_config) / spotstatus.STATUS_FILE
        )
        self.spot_status_max_age = float(cfg.get("spot_status_max_age", spotstatus.DEFAULT_MAX_AGE))
        if self.shares_per_device <= 0 or self.quantum_size <= 0:
            raise ValueError("shares_per_device and quantum_size must be positive")

    async def cleanup(self) -> None:
        if self._nvml is not None:
            self._nvml.close()

    async def update_plugin_config(self, new_plugin_config: Mapping[str, Any]) -> None:
        pass  # SPEC: config is read at init only (restart the agent to apply)

    @property
    def _slot(self) -> SlotName:
        return self.slot_shares if self.mode == AllocationMode.FRACTIONAL else self.slot_device

    # ---- devices & slots ----

    async def list_devices(self) -> Collection[CUDAFracDevice]:
        if not self.enabled or self._nvml is None:
            return []
        if self._devices is None:
            gpus = await asyncio.to_thread(self._nvml.list_gpus)
            self._devices = [
                _to_device(g, self.key)
                for g in gpus
                if self.selector.matches(g.name, g.total_memory, g.uuid)
            ]
        return self._devices

    def _device_map(self) -> dict[str, CUDAFracDevice]:
        return {str(d.device_id): d for d in self._devices or []}

    async def available_slots(self) -> Mapping[SlotName, Decimal]:
        devices = await self.list_devices()
        per_device = self.shares_per_device if self.mode == AllocationMode.FRACTIONAL else Decimal(1)
        return {self._slot: per_device * len(devices)}

    async def create_alloc_map(self) -> AbstractAllocMap:
        devices = await self.list_devices()
        if self.mode == AllocationMode.DISCRETE:
            return DiscretePropertyAllocMap(
                device_slots={
                    d.device_id: DeviceSlotInfo(SlotTypes.COUNT, self.slot_device, Decimal(1))
                    for d in devices
                },
                exclusive_slot_types=self.exclusive_slot_types,
            )
        strategy = (
            AllocationStrategy.FILL
            if self.allocation_strategy == "fill"
            else AllocationStrategy.EVENLY
        )
        return FractionAllocMap(
            device_slots={
                d.device_id: DeviceSlotInfo(SlotTypes.COUNT, self.slot_shares, self.shares_per_device)
                for d in devices
            },
            exclusive_slot_types=self.exclusive_slot_types,
            allocation_strategy=strategy,
            quantum_size=self.quantum_size,
        )

    def _limits(self, device_alloc: Any) -> list:
        devices = self._device_map()
        amounts = devalloc.amounts_for_slot(device_alloc, str(self._slot))
        if self.mode == AllocationMode.DISCRETE:
            # A discrete allocation of 1 is a whole device: compute_limits yields no limits.
            amounts = {dev: Decimal(1) for dev, amt in amounts.items() if amt > 0}
            per_device = Decimal(1)
        else:
            per_device = self.shares_per_device
        return compute_limits(
            {d: a for d, a in amounts.items() if d in devices},
            shares_per_device=per_device,
            total_memory_by_device={d: int(dev.memory_size) for d, dev in devices.items()},
            uuid_by_device={d: dev.uuid for d, dev in devices.items()},
            reserved_memory=self.reserved_memory,
        )

    # ---- container creation ----

    async def generate_docker_args(self, docker: Any, device_alloc: Any) -> Mapping[str, Any]:
        if not self.enabled:
            return {}
        limits = self._limits(device_alloc)
        if not limits:
            return {}
        env = {"LABGPU_DEVICE_UUIDS": ",".join(lim.uuid for lim in limits)}
        if self.fraction_enforced:
            env.update(build_hami_environ(limits, sm_limit=self.sm_limit))
        args: dict[str, Any] = {"Env": [f"{k}={v}" for k, v in env.items()]}
        if not self.is_fake:
            args["HostConfig"] = {
                "DeviceRequests": [
                    {
                        "Driver": "nvidia",
                        "DeviceIDs": [lim.uuid for lim in limits],
                        "Capabilities": DEVICE_CAPABILITIES,
                    }
                ],
            }
        return args

    async def get_hooks(self, distro: str, arch: str) -> Sequence[Path]:
        if self.enabled and self.fraction_enforced:
            return [self.hook_path]
        return []

    async def generate_resource_data(self, device_alloc: Any) -> Mapping[str, str]:
        if not self.enabled:
            return {}
        limits = self._limits(device_alloc)
        return {
            "CUDA_GLOBAL_DEVICE_IDS": ",".join(f"{lim.local_index}:{lim.device_id}" for lim in limits),
            "CUDA_RESOURCE_VIRTUALIZED": "1" if any(not lim.is_whole for lim in limits) else "0",
        }

    async def get_attached_devices(self, device_alloc: Any) -> Sequence[DeviceModelInfo]:
        devices = self._device_map()
        attached: list[DeviceModelInfo] = []
        for lim in self._limits(device_alloc):
            dev = devices[lim.device_id]
            attached.append({
                "device_id": dev.device_id,
                "model_name": dev.model_name,
                "data": {
                    "smp": lim.sm_limit_percent or PROCESSING_UNITS,
                    "mem": BinarySize(lim.memory_limit_bytes or dev.memory_size),
                },
            })
        return attached

    async def restore_from_container(self, container: Any, alloc_map: AbstractAllocMap) -> None:
        if not self.enabled:
            return
        resource_spec = await get_resource_spec_from_container(container.backend_obj)
        if resource_spec is None:
            return
        allocated = devalloc.amounts_for_slot(
            resource_spec.allocations.get(self.key, {}), str(self._slot)
        )
        alloc_map.apply_allocation({
            self._slot: {DeviceId(d): amt for d, amt in allocated.items()},
        })

    async def get_docker_networks(self, device_alloc: Any) -> list[str]:
        return []

    async def generate_mounts(self, source_path: Path, device_alloc: Any) -> list[MountInfo]:
        return []

    # ---- metadata ----

    def get_version(self) -> str:
        return __version__

    async def extra_info(self) -> Mapping[str, Any]:
        if not self.enabled or self._nvml is None:
            return {"cuda_support": False}
        try:
            driver = await asyncio.to_thread(self._nvml.driver_version)
        except NvmlError:
            driver = "unknown"
        return {
            "cuda_support": True,
            "nvidia_version": driver,
            "labgpu_mode": str(self.mode),
            "labgpu_models": ",".join(sorted({d.model_name for d in self._devices or []})),
            "labgpu_fake": "true" if self.is_fake else "false",
            "fraction_enforced": "true" if self.fraction_enforced else "false",
        }

    def get_metadata(self) -> AcceleratorMetadata:
        models = sorted({d.model_name for d in self._devices or []})
        fractional = self.mode == AllocationMode.FRACTIONAL
        if self.display_name:
            name = self.display_name
        elif self.key == "cuda":
            name = "fGPU" if fractional else "GPU"
        elif len(models) == 1:
            name = models[0].removeprefix("NVIDIA ")
        else:
            name = str(self.key)
        described = ", ".join(models) or "none"
        return {
            "slot_name": str(self._slot),
            "human_readable_name": name,
            "description": (
                f"Fractional GPU (HAMi-core enforced): {described}"
                if fractional
                else f"CUDA GPU: {described}"
            ),
            "display_unit": self._display_unit(models, fractional),
            "number_format": {"binary": False, "round_length": 2 if fractional else 0},
            "display_icon": "gpu1",
        }

    def _display_unit(self, models: Sequence[str], fractional: bool) -> str:
        """
        The WebUI labels each accelerator type in its session launcher by display_unit, so
        per-model slots need distinct units (SPEC 1.2).
        """
        if self.display_unit:
            return self.display_unit
        if self.key == "cuda":
            return "fGPU" if fractional else "GPU"
        return short_model_name(models[0]) if len(models) == 1 else str(self.key).upper()

    # ---- statistics ----

    async def _snapshots(self) -> list:
        if not self.enabled or self._nvml is None:
            return []
        nvml = self._nvml
        indices = [int(d.device_id) for d in await self.list_devices()]
        try:
            return await asyncio.to_thread(lambda: [nvml.snapshot(i) for i in indices])
        except NvmlError as e:
            log.warning("NVML stats failed: %s", e)
            return []

    async def gather_node_measures(self, ctx: StatContext) -> Sequence[NodeMeasurement]:
        snaps = await self._snapshots()
        mem_per_dev = {
            DeviceId(str(s.info.index)): Measurement(
                Decimal(s.used_memory), Decimal(s.info.total_memory)
            )
            for s in snaps
        }
        util_per_dev = {
            DeviceId(str(s.info.index)): Measurement(Decimal(s.gpu_util), Decimal(100))
            for s in snaps
        }
        return [
            NodeMeasurement(
                MetricKey(f"{self.key}_mem"),
                MetricTypes.GAUGE,
                unit_hint="bytes",
                stats_filter=frozenset({"max"}),
                per_node=Measurement(
                    Decimal(sum(s.used_memory for s in snaps)),
                    Decimal(sum(s.info.total_memory for s in snaps)),
                ),
                per_device=mem_per_dev,
            ),
            NodeMeasurement(
                MetricKey(f"{self.key}_util"),
                MetricTypes.UTILIZATION,
                unit_hint="percent",
                stats_filter=frozenset({"avg", "max"}),
                per_node=Measurement(
                    Decimal(sum(s.gpu_util for s in snaps)), Decimal(len(snaps) * 100)
                ),
                per_device=util_per_dev,
            ),
        ]

    async def gather_container_measures(
        self, ctx: StatContext, container_ids: Sequence[str]
    ) -> Sequence[ContainerMeasurement]:
        """Attribute GPU usage per process (SPEC 1.9) so co-located containers never mix."""
        snaps = await self._snapshots()
        pids = [p.pid for s in snaps for p in s.processes]
        pid_to_cid = await asyncio.to_thread(self._nvml.container_of_pids, pids) if self._nvml else {}
        wanted = {cid[:12]: cid for cid in container_ids}

        mem: dict[str, int] = defaultdict(int)
        util: dict[str, int] = defaultdict(int)
        gpus_seen: dict[str, set[int]] = defaultdict(set)
        for s in snaps:
            for p in s.processes:
                full_cid = pid_to_cid.get(p.pid)
                cid = wanted.get(full_cid[:12]) if full_cid else None
                if cid is None:
                    continue
                mem[cid] += p.used_memory
                util[cid] += p.sm_util
                gpus_seen[cid].add(s.info.index)
        capacity_by_index = {s.info.index: s.info.total_memory for s in snaps}
        return [
            ContainerMeasurement(
                MetricKey(f"{self.key}_mem"),
                MetricTypes.USAGE,
                unit_hint="bytes",
                stats_filter=frozenset({"max"}),
                per_container={
                    cid: Measurement(
                        Decimal(used),
                        Decimal(sum(capacity_by_index[i] for i in gpus_seen[cid])),
                    )
                    for cid, used in mem.items()
                },
            ),
            ContainerMeasurement(
                MetricKey(f"{self.key}_util"),
                MetricTypes.USAGE,
                unit_hint="percent",
                stats_filter=frozenset({"avg", "max"}),
                per_container={
                    cid: Measurement(Decimal(u), Decimal(len(gpus_seen[cid]) * 100))
                    for cid, u in util.items()
                },
            ),
            *await self._lending_measures(container_ids),
        ]

    async def _lending_measures(self, container_ids: Sequence[str]) -> list[ContainerMeasurement]:
        """`<key>_lent` and `<key>_lent_since` per session (SPEC 1.13); nothing if status is unknown."""
        if not self.enabled or not container_ids:
            return []
        status = await asyncio.to_thread(
            spotstatus.read_status, self.spot_status_path, time.time(), self.spot_status_max_age
        )
        if status is None:
            return []
        own = {d.uuid for d in self._devices or []}
        figures = spotstatus.session_lending(await self._gpus_of(container_ids), own, status)
        if not figures:
            return []
        return [
            ContainerMeasurement(
                MetricKey(f"{self.key}_lent"),
                MetricTypes.GAUGE,
                unit_hint="count",
                stats_filter=frozenset({"max"}),
                per_container={
                    cid: Measurement(Decimal(f.lent), Decimal(f.total)) for cid, f in figures.items()
                },
            ),
            ContainerMeasurement(
                MetricKey(f"{self.key}_lent_since"),
                MetricTypes.GAUGE,
                unit_hint="count",
                stats_filter=frozenset(),
                per_container={
                    # capacity 1 lets readers undo the manager summing duplicate series
                    # after agent restarts (SPEC 1.13)
                    cid: Measurement(Decimal(int(f.since)), Decimal(1)) for cid, f in figures.items()
                },
            ),
        ]

    async def _gpus_of(self, container_ids: Sequence[str]) -> dict[str, list[str]]:
        """Container id -> GPU UUIDs it holds, from its env; each container is inspected once."""
        cache = self._container_gpus if self._container_gpus is not None else {}
        missing = [cid for cid in container_ids if cid not in cache]
        if missing:
            try:
                async with aiodocker.Docker() as docker:
                    for cid in missing:
                        try:
                            container = await docker.containers.get(cid)
                            env = ((await container.show()).get("Config") or {}).get("Env") or []
                        except DockerError:
                            env = []
                        cache[cid] = spotstatus.uuids_from_env(env)
            except DockerError as e:
                log.warning("[%s] cannot inspect containers for lending stats: %r", self.entry_name, e)
        self._container_gpus = {cid: cache[cid] for cid in container_ids if cid in cache}
        return self._container_gpus

    async def gather_process_measures(
        self, ctx: StatContext, pid_map: Mapping[int, str]
    ) -> Sequence[ProcessMeasurement]:
        return []


class CUDAFracPlugin(LabGpuPlugin):
    """`cuda_frac`: all GPUs (unless model_pattern narrows them) under the key `cuda`."""

    entry_name = "cuda_frac"
    default_key = "cuda"


class GpuSlotPlugin1(LabGpuPlugin):
    entry_name = "gpu_slot_1"


class GpuSlotPlugin2(LabGpuPlugin):
    entry_name = "gpu_slot_2"


class GpuSlotPlugin3(LabGpuPlugin):
    entry_name = "gpu_slot_3"


class GpuSlotPlugin4(LabGpuPlugin):
    entry_name = "gpu_slot_4"


def short_model_name(model: str) -> str:
    """"NVIDIA RTX PRO 6000 Blackwell Workstation Edition" -> "PRO6000"."""
    words = model.removeprefix("NVIDIA ").removeprefix("GeForce ").split()
    # "RTX PRO 6000" -> "PRO6000", "RTX A6000" -> "A6000", but "RTX 4050" -> "RTX4050".
    if len(words) > 1 and words[0] == "RTX" and (words[1] == "PRO" or words[1][0].isalpha()):
        words = words[1:]
    keep = []
    for w in words:
        keep.append(w)
        if any(c.isdigit() for c in w):
            break
    return "".join(keep)


def _to_device(g: GpuInfo, key: DeviceName) -> CUDAFracDevice:
    numa: int | None
    try:
        numa_raw = Path(f"/sys/bus/pci/devices/{_sysfs_bus_id(g.pci_bus_id)}/numa_node").read_text()
        numa = int(numa_raw.strip())
        numa = None if numa < 0 else numa
    except (OSError, ValueError):
        numa = None
    return CUDAFracDevice(
        model_name=g.name,
        uuid=g.uuid,
        device_id=DeviceId(str(g.index)),
        hw_location=g.pci_bus_id,
        numa_node=numa,
        memory_size=g.total_memory,
        processing_units=PROCESSING_UNITS,
        # The agent's affinity map groups devices by device_name; without it the name is
        # derived from the class and allocation finds no devices for the plugin key.
        **_device_name_kwarg(key),
    )


def _device_name_kwarg(key: DeviceName) -> dict[str, Any]:
    params = inspect.signature(AbstractComputeDevice.__init__).parameters
    return {"device_name": key} if "device_name" in params else {}


def _sysfs_bus_id(nvml_bus_id: str) -> str:
    # NVML reports "00000000:3B:00.0"; sysfs uses "0000:3b:00.0".
    domain, rest = nvml_bus_id.split(":", 1)
    return f"{domain[-4:]}:{rest}".lower()
