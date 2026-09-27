"""
Mounted read-only into spot containers at /opt/labgpu/python and put first on PYTHONPATH by the
spot plugin (SPEC 2.12). Python imports it at startup.

When the spot monitor cannot move this session to another free GPU, it lowers the GPU memory limit
and sends LABGPU_SPOT_OOM_SIGNAL: the handler below raises torch.OutOfMemoryError in the main
thread at once, the same error a program gets when the GPU runs out of memory. Nothing is killed.

Afterwards the image's own sitecustomize, if any, is still imported.
"""

import os
import signal
import sys

_MESSAGE = (
    "CUDA out of memory: this spot GPU was reclaimed by its owner and no other GPU of the same "
    "type is free (labgpu). Reason: {reason}"
)


def _labgpu_spot_oom(signum, frame):
    try:
        with open("/tmp/labgpu-spot-evicted") as f:
            reason = f.read().strip()
    except OSError:
        reason = "unknown"
    torch = sys.modules.get("torch")
    error = getattr(torch, "OutOfMemoryError", None) if torch is not None else None
    raise (error or MemoryError)(_MESSAGE.format(reason=reason))


def _install():
    try:
        signum = int(os.environ.get("LABGPU_SPOT_OOM_SIGNAL", "0"))
    except ValueError:
        return
    if signum <= 0:
        return
    try:
        signal.signal(signum, _labgpu_spot_oom)
    except (ValueError, OSError):
        pass  # not the main thread or an invalid signal: leave the program alone


def _chain():
    """Import the next sitecustomize on sys.path, the one this file shadows."""
    here = os.path.dirname(os.path.abspath(__file__))
    rest = [p for p in sys.path if os.path.abspath(p or ".") != here]
    import importlib.machinery
    import importlib.util

    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:  # never break the program's startup
        print(f"labgpu: the image's sitecustomize failed: {e!r}", file=sys.stderr)


_install()
_chain()
