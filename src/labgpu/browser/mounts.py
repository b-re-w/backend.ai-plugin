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
    The opener and its profile script, after creating the request file in the session's
    /home/work. Without the request file (cannot create it) the opener prints the URL instead.
    """
    f = request_file(config_dir, key)
    try:
        f.parent.mkdir(mode=0o777, exist_ok=True)
        f.parent.chmod(0o777)  # the session user may run as another uid than the agent
        f.touch(exist_ok=True)
        f.chmod(0o666)
    except OSError as e:
        log.warning("[%s] cannot create %s: URLs from this session are only printed: %s", key, f, e)
    return [MountInfo(MountTypes.BIND, m.source, m.target) for m in browser_mounts(key)]
