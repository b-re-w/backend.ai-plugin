"""
Spot launch mode as Backend.AI accelerator plugins (SPEC 2.12).

`gpu_spot_1..4` each offer `<key>.device` for one GPU model, e.g. `pro6000-spot.device`. A spot
session is a normal Backend.AI session that requests it from the WebUI session launcher. The slot
capacity is the number of GPUs of that model the idleness monitor currently judges lendable or
lent, re-read every time the agent refreshes its slots (30 s in 26.8), so the manager only
schedules spot sessions while there is room. This plugin picks the GPU a new spot session runs on
(a LENDABLE one) and caps its memory with HAMi-core; the other GPUs of the model are attached too,
because cuda-checkpoint can only move a process to a GPU it can see, but capped at 1 MiB so the
session cannot use them. Moving the session when the owner comes back is the monitor's job.

The monitor runs in a background thread of the agent itself, started by the first spot plugin
that initialises (SPEC 2.13); there is no separate service.

The owner-side plugin (`gpu_slot_N` or `cuda_frac`) keeps the GPUs themselves; this plugin claims
none of them and only exposes one pool device per model.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import logging
import time
from collections.abc import Collection, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from ai.backend.agent.resources import (
    AbstractAllocMap,
    AbstractComputePlugin,
    DeviceSlotInfo,
    DiscretePropertyAllocMap,
)
from ai.backend.agent.stats import (
    ContainerMeasurement,
    NodeMeasurement,
    ProcessMeasurement,
    StatContext,
)
from ai.backend.agent.types import MountInfo
from ai.backend.common.types import (
    AcceleratorMetadata,
    DeviceId,
    DeviceModelInfo,
    DeviceName,
    MountTypes,
    SlotName,
    SlotTypes,
)

from .. import __version__, devalloc, spotstatus
from ..nvml import FakeNvmlReader, GpuInfo, NvmlError, NvmlReader, open_reader
from ..fraction import MEMORY_SHARED_CACHE
from ..paths import agent_state_dir, default_cuda_checkpoint, default_hook_path
from ..spot.ckpt import CONTAINER_CUDA_CHECKPOINT
from ..selection import GpuSelector, validate_key
from ..sizes import MiB
from ..spot.config import Config as SpotConfigFile
from .plugin import (
    DEVICE_CAPABILITIES,
    PROCESSING_UNITS,
    CUDAFracDevice,
    PluginNotConfigured,
    _device_name_kwarg,
    get_resource_spec_from_container,
    short_model_name,
)

log = logging.getLogger("ai.backend.labgpu.accelerator.spot")

POOL_DEVICE_ID = DeviceId("spot")

# Env handed to spot containers; the monitor recognises spot sessions by it (SPEC 2.12).
ENV_SPOT = "LABGPU_SPOT"
ENV_SPOT_UUIDS = "LABGPU_SPOT_UUIDS"
ENV_SPOT_GPU = "LABGPU_SPOT_GPU"
BLOCKED_LIMIT = "1m"  # HAMi-core limit for GPUs the session must not use (0 would mean unlimited)
RESERVE_SECONDS = 120.0  # how long a GPU handed to a new session stays taken before the monitor reports it


class LabGpuSpotPlugin(AbstractComputePlugin):
    config_watch_enabled = False

    entry_name: str = ""
    key = DeviceName("spot")
    slot_types: Sequence[tuple[SlotName, SlotTypes]] = ()
    exclusive_slot_types: set[str] = set()
    display_name: str | None = None
    display_unit: str | None = None
    selector: GpuSelector = GpuSelector()
    enabled: bool = True
    spot_status_path: Path = Path("./var/lib/backend.ai/labgpu") / spotstatus.STATUS_FILE
    spot_status_max_age: float = spotstatus.DEFAULT_MAX_AGE

    _nvml: NvmlReader | FakeNvmlReader | None = None
    _gpus: list[GpuInfo] | None = None
    _monitor: Any = None  # the BackgroundMonitor this instance started, if any
    hook_path: Path = default_hook_path()
    cuda_checkpoint: Path = Path("cuda-checkpoint")
    _handed_out: dict[str, float] | None = None  # GPU uuid -> when a new session got it

    @property
    def slot(self) -> SlotName:
        return SlotName(f"{self.key}.device")

    @property
    def is_fake(self) -> bool:
        return bool(getattr(self._nvml, "is_fake", False))

    async def init(self, context: Any | None = None) -> None:
        cfg = self.plugin_config
        raw_key = cfg.get("key")
        if not raw_key:
            raise PluginNotConfigured(
                f"{self.entry_name}: set config/plugins/accelerator/{self.entry_name}/key to use it"
            )
        self.key = DeviceName(validate_key(str(raw_key)))
        self.selector = GpuSelector.from_config(dict(cfg))
        self.display_name = cfg.get("display_name")
        self.display_unit = cfg.get("display_unit")
        state_dir = agent_state_dir(self.local_config)
        self.spot_status_path = (
            Path(cfg["spot_status_path"]) if cfg.get("spot_status_path") else state_dir / spotstatus.STATUS_FILE
        )
        self.spot_status_max_age = float(cfg.get("spot_status_max_age", spotstatus.DEFAULT_MAX_AGE))
        self.hook_path = Path(cfg["hook_path"]) if cfg.get("hook_path") else default_hook_path()
        self._handed_out = {}
        self.slot_types = ((self.slot, SlotTypes.COUNT),)
        self.exclusive_slot_types = {str(self.slot)}
        try:
            self._nvml = open_reader()
            await self._list_gpus()
        except NvmlError as e:
            log.error("[%s] NVML unavailable (%s); disabled.", self.entry_name, e)
            self.enabled = False
            return
        if not self.hook_path.is_file():
            # SPEC 2.12: a spot session without a memory cap could starve the owner; offer none.
            log.error(
                "[%s] HAMi-core %s not found: spot memory cannot be capped, spot slots disabled.",
                self.entry_name,
                self.hook_path,
            )
            self.enabled = False
            return
        if str(cfg.get("monitor_enabled", "true")).lower() not in ("0", "false", "no"):
            self._start_monitor(cfg.get("monitor") or {}, state_dir)
        spot_section = (cfg.get("monitor") or {}).get("spot") or {}
        self.cuda_checkpoint = (
            Path(spot_section["cuda_checkpoint"])
            if spot_section.get("cuda_checkpoint")
            else default_cuda_checkpoint()
        )
        log.info(
            "[%s] labgpu %s spot: key=%s gpus=%s",
            self.entry_name,
            __version__,
            self.key,
            [g.uuid for g in self._gpus or []],
        )

    def _start_monitor(self, sections: Mapping[str, Any], state_dir: Path) -> None:
        """
        Run the spot monitor in this agent unless another spot plugin already does (SPEC 2.13).
        Its settings are this plugin's etcd `monitor/<section>/<key>`; state goes to `state_dir`.
        """
        from ..spot.daemon import BackgroundMonitor, Controller
        from ..spot.docker import DockerCli

        def make() -> Controller:
            cfg = SpotConfigFile.from_dict({k: dict(v) for k, v in sections.items()})
            if "state_dir" not in sections.get("controller", {}):
                cfg = replace(cfg, controller=replace(cfg.controller, state_dir=state_dir))
            return Controller(cfg, open_reader(), DockerCli())

        try:
            self._monitor = BackgroundMonitor.ensure_started(make)
        except Exception:
            log.exception("[%s] could not start the spot monitor; spot slots stay at 0", self.entry_name)

    async def _list_gpus(self) -> list[GpuInfo]:
        if self._gpus is None:
            if self._nvml is None:
                return []
            gpus = await asyncio.to_thread(self._nvml.list_gpus)
            self._gpus = [g for g in gpus if self.selector.matches(g.name, g.total_memory, g.uuid)]
        return self._gpus

    async def cleanup(self) -> None:
        if self._monitor is not None:
            await asyncio.to_thread(self._monitor.stop)
            self._monitor = None
        if self._nvml is not None:
            self._nvml.close()

    async def update_plugin_config(self, new_plugin_config: Mapping[str, Any]) -> None:
        pass

    # ---- devices & slots ----

    async def list_devices(self) -> Collection[CUDAFracDevice]:
        gpus = await self._list_gpus() if self.enabled else []
        if not gpus:
            return []
        first = gpus[0]
        return [
            CUDAFracDevice(
                model_name=first.name,
                uuid="",
                device_id=POOL_DEVICE_ID,
                hw_location="labgpu-spot-pool",
                numa_node=None,
                memory_size=max(g.total_memory for g in gpus),
                processing_units=PROCESSING_UNITS,
                **_device_name_kwarg(self.key),
            )
        ]

    async def available_slots(self) -> Mapping[SlotName, Decimal]:
        gpus = await self._list_gpus() if self.enabled else []
        status = await asyncio.to_thread(
            spotstatus.read_status, self.spot_status_path, time.time(), self.spot_status_max_age
        )
        return {self.slot: Decimal(spotstatus.spot_capacity([g.uuid for g in gpus], status))}

    async def create_alloc_map(self) -> AbstractAllocMap:
        # The agent builds allocation maps once at start, so the pool is sized for every GPU of
        # the model; the manager is held back by the live capacity in available_slots().
        gpus = await self._list_gpus() if self.enabled else []
        devices = (
            {POOL_DEVICE_ID: DeviceSlotInfo(SlotTypes.COUNT, self.slot, Decimal(len(gpus)))}
            if gpus
            else {}
        )
        return DiscretePropertyAllocMap(
            device_slots=devices, exclusive_slot_types=self.exclusive_slot_types
        )

    def _allocated(self, device_alloc: Any) -> Decimal:
        amounts = devalloc.amounts_for_slot(device_alloc, str(self.slot))
        return sum(amounts.values(), Decimal(0))

    # ---- container creation ----

    def _pick(self) -> tuple[str, int]:
        """The lendable GPU for a new spot session and its memory cap in bytes (SPEC 2.12)."""
        now = time.time()
        handed = {u: t for u, t in (self._handed_out or {}).items() if now - t < RESERVE_SECONDS}
        status = spotstatus.read_status(self.spot_status_path, now, self.spot_status_max_age)
        picked = spotstatus.pick_spot_gpu([g.uuid for g in self._gpus or []], status, handed)
        if picked is None:
            # The manager only schedules while there is room, so this is a race; the monitor
            # will move or evict the session. Cap it hard meanwhile.
            log.warning("[%s] no lendable GPU for a new spot session", self.entry_name)
            picked = ((self._gpus or [])[0].uuid, 0)
        handed[picked[0]] = now
        self._handed_out = handed
        return picked

    async def generate_docker_args(self, docker: Any, device_alloc: Any) -> Mapping[str, Any]:
        if not self.enabled or self._allocated(device_alloc) <= 0:
            return {}
        gpus = self._gpus or []
        chosen, lendable = await asyncio.to_thread(self._pick)
        # The chosen GPU comes first so it is the program's cuda:0; the others stay attached
        # for moving but get a 1 MiB cap (SPEC 2.12).
        order = [chosen] + [g.uuid for g in gpus if g.uuid != chosen]
        env = {
            ENV_SPOT: "1",
            ENV_SPOT_UUIDS: ",".join(g.uuid for g in gpus),
            ENV_SPOT_GPU: chosen,
            "CUDA_VISIBLE_DEVICES": ",".join(order),
            "CUDA_DEVICE_MEMORY_SHARED_CACHE": MEMORY_SHARED_CACHE,
        }
        for i, uuid in enumerate(order):
            env[f"CUDA_DEVICE_MEMORY_LIMIT_{i}"] = (
                f"{max(lendable // MiB, 1)}m" if uuid == chosen else BLOCKED_LIMIT
            )
        args: dict[str, Any] = {"Env": [f"{k}={v}" for k, v in env.items()]}
        if not self.is_fake:
            # NVML indices, not UUIDs: the stock cuda plugin looks DeviceIDs up by index when it
            # gathers container stats and fails on anything else.
            indices = [str(g.index) for g in gpus]
            args["HostConfig"] = {
                "DeviceRequests": [
                    {"Driver": "nvidia", "DeviceIDs": indices, "Capabilities": DEVICE_CAPABILITIES}
                ],
            }
        return args

    async def get_hooks(self, distro: str, arch: str) -> Sequence[Path]:
        return [self.hook_path] if self.enabled else []

    async def generate_resource_data(self, device_alloc: Any) -> Mapping[str, str]:
        return {}

    async def get_attached_devices(self, device_alloc: Any) -> Sequence[DeviceModelInfo]:
        if not self.enabled or self._allocated(device_alloc) <= 0:
            return []
        gpus = self._gpus or []
        return [
            {
                "device_id": POOL_DEVICE_ID,
                "model_name": gpus[0].name if gpus else "",
                "data": {"smp": PROCESSING_UNITS, "mem": max((g.total_memory for g in gpus), default=0)},
            }
        ]

    async def restore_from_container(self, container: Any, alloc_map: AbstractAllocMap) -> None:
        if not self.enabled:
            return
        resource_spec = await get_resource_spec_from_container(container.backend_obj)
        if resource_spec is None:
            return
        allocated = devalloc.amounts_for_slot(
            resource_spec.allocations.get(self.key, {}), str(self.slot)
        )
        alloc_map.apply_allocation({self.slot: {DeviceId(d): a for d, a in allocated.items()}})

    async def get_docker_networks(self, device_alloc: Any) -> list[str]:
        return []

    async def generate_mounts(self, source_path: Path, device_alloc: Any) -> list[MountInfo]:
        """cuda-checkpoint, read-only, for the monitor to run inside the container (SPEC 2.13)."""
        if not self.enabled or self._allocated(device_alloc) <= 0:
            return []
        tool = self.cuda_checkpoint
        if not tool.is_file():
            log.warning("[%s] %s not found: this spot session cannot be moved", self.entry_name, tool)
            return []
        return [MountInfo(MountTypes.BIND, tool, Path(CONTAINER_CUDA_CHECKPOINT))]

    # ---- metadata ----

    def get_version(self) -> str:
        return __version__

    async def extra_info(self) -> Mapping[str, Any]:
        return {"labgpu_spot": "true" if self.enabled else "false"}

    def get_metadata(self) -> AcceleratorMetadata:
        models = sorted({g.name for g in self._gpus or []})
        base = short_model_name(models[0]) if len(models) == 1 else str(self.key).upper()
        return {
            "slot_name": str(self.slot),
            "human_readable_name": self.display_name or f"{base} Spot",
            "description": f"Spot GPU (lent while the owner is idle): {', '.join(models) or 'none'}",
            "display_unit": self.display_unit or f"{base}-SPOT",
            "number_format": {"binary": False, "round_length": 0},
            "display_icon": "gpu1",
        }

    # ---- statistics: GPU usage of spot sessions is reported by the owner-side plugin ----

    async def gather_node_measures(self, ctx: StatContext) -> Sequence[NodeMeasurement]:
        return []

    async def gather_container_measures(
        self, ctx: StatContext, container_ids: Sequence[str]
    ) -> Sequence[ContainerMeasurement]:
        return []

    async def gather_process_measures(
        self, ctx: StatContext, pid_map: Mapping[int, str]
    ) -> Sequence[ProcessMeasurement]:
        return []


class SpotSlotPlugin1(LabGpuSpotPlugin):
    entry_name = "gpu_spot_1"


class SpotSlotPlugin2(LabGpuSpotPlugin):
    entry_name = "gpu_spot_2"


class SpotSlotPlugin3(LabGpuSpotPlugin):
    entry_name = "gpu_spot_3"


class SpotSlotPlugin4(LabGpuSpotPlugin):
    entry_name = "gpu_spot_4"
