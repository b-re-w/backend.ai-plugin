"""Which GPUs a plugin instance takes (SPEC 1.11), and the claim guard against overlaps."""

from __future__ import annotations

import fnmatch
import os
import re
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .sizes import parse_size

_rx_key = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class GpuClaimConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuSelector:
    patterns: tuple[str, ...] = ("*",)
    min_memory: int | None = None
    max_memory: int | None = None
    excluded_uuids: frozenset[str] = frozenset()

    @classmethod
    def from_config(cls, cfg: dict) -> GpuSelector:
        raw = str(cfg.get("model_pattern", "*"))
        patterns = tuple(p.strip() for p in raw.split(",") if p.strip()) or ("*",)
        raw_mask = cfg.get("device_mask")
        return cls(
            patterns=patterns,
            min_memory=parse_size(cfg["min_memory"]) if cfg.get("min_memory") else None,
            max_memory=parse_size(cfg["max_memory"]) if cfg.get("max_memory") else None,
            excluded_uuids=frozenset(m.strip() for m in raw_mask.split(",")) if raw_mask else frozenset(),
        )

    def matches(self, name: str, total_memory: int, uuid: str = "") -> bool:
        if uuid in self.excluded_uuids:
            return False
        if self.min_memory is not None and total_memory < self.min_memory:
            return False
        if self.max_memory is not None and total_memory > self.max_memory:
            return False
        return model_matches(name, self.patterns)


def model_matches(name: str, patterns: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(fnmatch.fnmatchcase(lowered, p.lower()) for p in patterns)


def validate_key(key: str) -> str:
    if not _rx_key.match(key) or key in ("cpu", "mem"):
        raise ValueError(f"invalid plugin key {key!r}: use lowercase letters, digits, and '-'")
    return key


def default_claim_dir() -> Path:
    env = os.environ.get("LABGPU_CLAIM_DIR")
    if env:
        return Path(env)
    run = Path("/run/labgpu/claims")
    try:
        run.mkdir(parents=True, exist_ok=True)
        if os.access(run, os.W_OK):
            return run
    except OSError:
        pass
    return Path(tempfile.gettempdir()) / "labgpu-claims"


def claim_gpus(uuids: Iterable[str], owner: str, claim_dir: Path | None = None) -> None:
    """
    Record that `owner` (a plugin entry-point name) takes these GPUs in this process.
    Raises if another plugin in the same process already took one of them.
    """
    directory = claim_dir or default_claim_dir()
    directory.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    uuids = list(uuids)
    for uuid in uuids:
        path = directory / uuid
        try:
            other_pid, _, other_owner = path.read_text().partition(":")
        except OSError:
            continue
        if other_pid == str(pid) and other_owner != owner:
            raise GpuClaimConflict(
                f"GPU {uuid} is already taken by plugin {other_owner!r}; narrow model_pattern "
                f"or memory bounds so that {owner!r} does not overlap it"
            )
    for uuid in uuids:
        (directory / uuid).write_text(f"{pid}:{owner}")
