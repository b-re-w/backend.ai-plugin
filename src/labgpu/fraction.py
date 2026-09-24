"""
Share-to-limit arithmetic for fractional GPU allocation (SPEC 1.6, 1.7).

Pure module: no Backend.AI, NVML, or Docker imports.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from .devalloc import device_sort_key
from .sizes import MiB

MEMORY_SHARED_CACHE = "/tmp/labgpu-vgpu.cache"


@dataclass(frozen=True)
class DeviceLimit:
    """The HAMi-core limits for one attached device, or none if it is attached whole."""

    local_index: int
    device_id: str
    uuid: str
    share: Decimal
    fraction: Decimal
    memory_limit_bytes: int | None
    sm_limit_percent: int | None

    @property
    def is_whole(self) -> bool:
        return self.memory_limit_bytes is None


def compute_limit(
    *,
    local_index: int,
    device_id: str,
    uuid: str,
    share: Decimal,
    shares_per_device: Decimal,
    total_memory: int,
    reserved_memory: int = 0,
) -> DeviceLimit:
    fraction = share / shares_per_device
    if fraction >= 1:
        return DeviceLimit(local_index, device_id, uuid, share, fraction, None, None)
    mem_bytes = int(fraction * total_memory) - reserved_memory
    mem_bytes = max(mem_bytes // MiB, 1) * MiB
    sm_pct = min(max(math.ceil(fraction * 100), 1), 100)
    return DeviceLimit(local_index, device_id, uuid, share, fraction, mem_bytes, sm_pct)


def compute_limits(
    shares_by_device: Mapping[str, Decimal],
    *,
    shares_per_device: Decimal,
    total_memory_by_device: Mapping[str, int],
    uuid_by_device: Mapping[str, str],
    reserved_memory: int = 0,
) -> list[DeviceLimit]:
    """Limits for every device with a non-zero share, in stable local-index order."""
    active = sorted((d for d, s in shares_by_device.items() if s > 0), key=device_sort_key)
    return [
        compute_limit(
            local_index=idx,
            device_id=dev_id,
            uuid=uuid_by_device[dev_id],
            share=shares_by_device[dev_id],
            shares_per_device=shares_per_device,
            total_memory=total_memory_by_device[dev_id],
            reserved_memory=reserved_memory,
        )
        for idx, dev_id in enumerate(active)
    ]


def build_hami_environ(limits: Sequence[DeviceLimit], *, sm_limit: bool) -> dict[str, str]:
    """
    HAMi-core environment variables. Indices are container-local because the container sees
    its attached GPUs renumbered from zero. HAMi-core takes a single global SM limit, so the
    largest per-device limit is used.
    """
    env: dict[str, str] = {}
    fractional = [lim for lim in limits if not lim.is_whole]
    for lim in fractional:
        assert lim.memory_limit_bytes is not None
        env[f"CUDA_DEVICE_MEMORY_LIMIT_{lim.local_index}"] = f"{lim.memory_limit_bytes // MiB}m"
    if fractional:
        env["CUDA_DEVICE_MEMORY_SHARED_CACHE"] = MEMORY_SHARED_CACHE
        if sm_limit:
            env["CUDA_DEVICE_SM_LIMIT"] = str(
                max(lim.sm_limit_percent or 100 for lim in fractional)
            )
    return env

