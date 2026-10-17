"""SPEC 1.14: the xdg-open shim forwards URLs to the newest live VS Code window."""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

from labgpu.browser import BROWSER_DIR, browser_environ, browser_mounts

SHIM = BROWSER_DIR / "xdg-open"


def run_shim(home: Path, url: str) -> subprocess.CompletedProcess[str]:
    env = {"HOME": str(home), "PATH": os.environ["PATH"], "XDG_RUNTIME_DIR": str(home / "run")}
    return subprocess.run(["sh", str(SHIM), url], env=env, capture_output=True, text=True)


def fake_helper(home: Path, log: Path, live: set[str]) -> None:
    """A serve-web browser.sh that succeeds only for sockets in `live`, logging what it opened."""
    helper = home / ".vscode/cli/serve-web/abc/bin/helpers/browser.sh"
    helper.parent.mkdir(parents=True)
    accepted = " ".join(sorted(live))
    helper.write_text(
        "#!/bin/sh\n"
        f'for s in {accepted}; do [ "$VSCODE_IPC_HOOK_CLI" = "$s" ] && '
        f'echo "$VSCODE_IPC_HOOK_CLI $1" >> {log} && exit 0; done\nexit 1\n'
    )
    helper.chmod(0o755)


def make_socket(path: Path) -> socket.socket:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = socket.socket(socket.AF_UNIX)
    s.bind(str(path))
    return s


def test_opens_through_a_live_window_skipping_dead_ones(tmp_path):
    log = tmp_path / "opened"
    dead = tmp_path / "run/vscode-ipc-dead.sock"
    live = tmp_path / "run/vscode-ipc-live.sock"
    keep = [make_socket(live), make_socket(dead)]  # dead is newer, so it is tried first
    fake_helper(tmp_path, log, {str(live)})
    result = run_shim(tmp_path, "https://github.com/login/device")
    assert result.returncode == 0
    assert log.read_text().split() == [str(live), "https://github.com/login/device"]
    for s in keep:
        s.close()


def test_prints_the_url_without_a_window(tmp_path):
    fake_helper(tmp_path, tmp_path / "opened", set())
    result = run_shim(tmp_path, "https://example.com")
    assert result.returncode == 1
    assert "https://example.com" in result.stderr


def test_refuses_non_web_urls(tmp_path):
    result = run_shim(tmp_path, "file:///etc/passwd")
    assert result.returncode == 3


def test_mounts_and_environ_are_per_key():
    assert [str(m.target) for m in browser_mounts("pro6000")] == [
        "/opt/labgpu/browser-pro6000",
        "/etc/profile.d/labgpu-browser-pro6000.sh",
    ]
    assert browser_environ("pro6000") == {"BROWSER": "/opt/labgpu/browser-pro6000/xdg-open"}
