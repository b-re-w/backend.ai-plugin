"""
Default locations of the external tools the init scripts install into the plugin checkout's own
.venv (scripts/install_hami_core.sh, scripts/install_cuda_checkpoint.sh). Pure module.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

# src/labgpu/paths.py -> the checkout root (the package is installed editable from it).
REPO_DIR = Path(__file__).resolve().parents[2]
VENV_DIR = REPO_DIR / ".venv"


def default_hook_path() -> Path:
    """HAMi-core library the plugins inject (SPEC 1.2 `hook_path`)."""
    return VENV_DIR / "lib" / "libvgpu.so"


def default_cuda_checkpoint() -> Path:
    """<checkout>/.venv/bin/cuda-checkpoint, else whatever is on PATH (SPEC 2.2)."""
    local = VENV_DIR / "bin" / "cuda-checkpoint"
    if local.exists():
        return local
    found = shutil.which("cuda-checkpoint")
    return Path(found) if found else local


def agent_state_dir(local_config: Any) -> Path:
    """
    `<var-base-path>/labgpu` of the agent the plugin runs in (SPEC 2.13): Backend.AI keeps an
    agent's runtime state under its `[agent] var-base-path`, which the operator makes writable.
    """
    base: Any = None
    for getter in (
        lambda c: c["agent"]["var-base-path"],
        lambda c: c.agent.var_base_path,
        lambda c: c.agent_common.var_base_path,
    ):
        try:
            base = getter(local_config)
            break
        except (KeyError, AttributeError, TypeError):
            continue
    return Path(base or "./var/lib/backend.ai") / "labgpu"
