"""
Opening URLs from GPU sessions in the user's browser (SPEC 1.14).

Pure module: builds the mounts and environment both labgpu plugins add to a session. The plugins
convert the mount tuples to Backend.AI MountInfo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BROWSER_DIR = Path(__file__).resolve().parent
CONTAINER_DIR_PREFIX = "/opt/labgpu/browser-"
CONTAINER_PROFILE_PREFIX = "/etc/profile.d/labgpu-browser-"


@dataclass(frozen=True)
class BrowserMount:
    source: Path
    target: Path


def browser_mounts(key: str) -> list[BrowserMount]:
    """Per plugin key, so a session holding several keys never mounts one target twice."""
    return [
        BrowserMount(BROWSER_DIR, Path(f"{CONTAINER_DIR_PREFIX}{key}")),
        BrowserMount(BROWSER_DIR / "profile.sh", Path(f"{CONTAINER_PROFILE_PREFIX}{key}.sh")),
    ]


def browser_environ(key: str) -> dict[str, str]:
    """$BROWSER reaches every process of the container, tmux servers included."""
    return {"BROWSER": f"{CONTAINER_DIR_PREFIX}{key}/xdg-open"}
