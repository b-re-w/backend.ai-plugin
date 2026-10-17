"""
Opening URLs from GPU sessions in the user's browser (SPEC 1.14).

Pure module: builds the mounts and environment both labgpu plugins add to a session. The plugins
create the request file and convert the mount tuples to Backend.AI MountInfo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BROWSER_DIR = Path(__file__).resolve().parent
CONTAINER_DIR_PREFIX = "/opt/labgpu/browser-"
CONTAINER_PROFILE_PREFIX = "/etc/profile.d/labgpu-browser-"
# The desktop app reads the first of these (by key) with Backend.AI's download_single.
CONTAINER_REQUEST_DIR = "/home/work/.labgpu-browser"
REQUEST_FILE_NAME = "browser-open-url"  # in the plugin's per-kernel config dir on the host


@dataclass(frozen=True)
class BrowserMount:
    source: Path
    target: Path


def request_file(config_dir: Path) -> Path:
    """Host file the session writes URLs to; the plugin creates it in its per-kernel config dir."""
    return config_dir / REQUEST_FILE_NAME


def browser_mounts(key: str, config_dir: Path) -> list[BrowserMount]:
    """Per plugin key, so a session holding several keys never mounts one target twice."""
    return [
        BrowserMount(BROWSER_DIR, Path(f"{CONTAINER_DIR_PREFIX}{key}")),
        BrowserMount(BROWSER_DIR / "profile.sh", Path(f"{CONTAINER_PROFILE_PREFIX}{key}.sh")),
        BrowserMount(request_file(config_dir), Path(f"{CONTAINER_REQUEST_DIR}/{key}")),
    ]


def browser_environ(key: str) -> dict[str, str]:
    """$BROWSER reaches every process of the container, tmux servers included."""
    return {"BROWSER": f"{CONTAINER_DIR_PREFIX}{key}/xdg-open"}
