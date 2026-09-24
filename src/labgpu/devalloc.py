"""
Normalize the `device_alloc` argument across Backend.AI versions (SPEC 1.10).

25.x passes `Mapping[SlotName, Mapping[DeviceId, Decimal]]`; newer trees pass a
`DeviceAllocation` whose `units` carry `device_id` and per-slot `amounts`.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

type SlotAlloc = dict[str, dict[str, Decimal]]


def normalize(device_alloc: Any) -> SlotAlloc:
    result: SlotAlloc = {}
    units = getattr(device_alloc, "units", None)
    if units is not None:
        for unit in units:
            for slot_name, amount in unit.amounts.items():
                result.setdefault(str(slot_name), {})[str(unit.device_id)] = Decimal(amount)
        return result
    if isinstance(device_alloc, Mapping):
        for slot_name, per_device in device_alloc.items():
            result[str(slot_name)] = {
                str(dev_id): Decimal(amount) for dev_id, amount in per_device.items()
            }
        return result
    raise TypeError(f"unsupported device allocation: {type(device_alloc).__name__}")


def amounts_for_slot(device_alloc: Any, slot_name: str) -> dict[str, Decimal]:
    return normalize(device_alloc).get(slot_name, {})


def active_device_ids(device_alloc: Any) -> list[str]:
    """Devices holding a non-zero amount of any slot, sorted by numeric index when possible."""
    ids = {dev for per_dev in normalize(device_alloc).values() for dev, amt in per_dev.items() if amt > 0}
    return sorted(ids, key=device_sort_key)


def device_sort_key(device_id: str) -> tuple[int, int, str]:
    """Numeric device IDs in numeric order, then any others lexically."""
    return (0, int(device_id), "") if device_id.isdigit() else (1, 0, device_id)
