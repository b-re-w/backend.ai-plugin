"""
Backend.AI glue for SPEC 1.14: creates the per-session request file and returns MountInfo.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ai.backend.agent.types import MountInfo
from ai.backend.common.types import MountTypes

from . import browser_mounts, request_file

log = logging.getLogger("ai.backend.labgpu.browser")


def browser_mount_infos(key: str, config_dir: Path) -> list[MountInfo]:
    """
    The opener, its profile script and the request file. Without the request file (cannot create
    it) only the opener is mounted, which then prints the URL instead.
    """
    mounts = browser_mounts(key, config_dir)
    try:
        f = request_file(config_dir)
        f.touch(exist_ok=True)
        f.chmod(0o666)  # the session user writes it, whatever uid the container runs as
    except OSError as e:
        log.warning("[%s] cannot create %s: URLs from this session are only printed: %s", key, f, e)
        mounts = [m for m in mounts if m.source != f]
    return [MountInfo(MountTypes.BIND, m.source, m.target) for m in mounts]
