"""Human-readable binary sizes ("16g", "512m") to bytes."""

from __future__ import annotations

import re

MiB = 2**20
GiB = 2**30

_UNITS = {"": 1, "b": 1, "k": 2**10, "m": 2**20, "g": 2**30, "t": 2**40}
_rx_size = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(?:i?b)?\s*$", re.IGNORECASE)


def parse_size(value: str | int) -> int:
    """Parse "16g", "512MiB", "1.5G" or a plain byte count into bytes."""
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"negative size: {value}")
        return value
    m = _rx_size.match(value)
    if m is None:
        raise ValueError(f"invalid size: {value!r}")
    number, unit = m.groups()
    return int(float(number) * _UNITS[unit.lower()])


def format_mib(num_bytes: int) -> str:
    """Format bytes as HAMi-core's memory limit syntax ("20480m")."""
    return f"{num_bytes // MiB}m"
