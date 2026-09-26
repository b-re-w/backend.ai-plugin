"""
Default locations of the external tools the init scripts install into the plugin checkout's own
.venv (scripts/install_hami_core.sh, scripts/install_cuda_checkpoint.sh). Pure module.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# src/labgpu/paths.py -> the checkout root (the package is installed editable from it).
REPO_DIR = Path(__file__).resolve().parents[2]
VENV_DIR = REPO_DIR / ".venv"


# Monitor state (status.json, parked.json) and its optional config, writable without root:
# the agent may run as an ordinary user (SPEC 2.13).
STATE_DIR = VENV_DIR / "labgpu"
CONFIG_PATH = STATE_DIR / "spot.toml"


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
