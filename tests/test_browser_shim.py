"""SPEC 1.14: the xdg-open shim queues URLs for the WebUI desktop app."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from labgpu.browser import BROWSER_DIR, browser_environ, browser_mounts

SHIM_TEXT = (BROWSER_DIR / "xdg-open").read_text()


def run_shim(tmp_path: Path, url: str) -> subprocess.CompletedProcess[str]:
    # The shim writes under /home/work/.labgpu-browser; point it at a test directory instead.
    shim = tmp_path / "xdg-open"
    shim.write_text(SHIM_TEXT.replace("/home/work/.labgpu-browser", str(tmp_path / "req")))
    env = {"PATH": os.environ["PATH"]}
    return subprocess.run(["sh", str(shim), url], env=env, capture_output=True, text=True)


def test_writes_the_url_to_the_first_request_file(tmp_path):
    req = tmp_path / "req"
    req.mkdir()
    for key in ("cuda-pro6000", "cuda-a6000"):
        (req / key).write_text("")
    result = run_shim(tmp_path, "https://github.com/login/device")
    assert result.returncode == 0
    stamp, url = (req / "cuda-a6000").read_text().split()
    assert url == "https://github.com/login/device" and stamp.isdigit()
    assert (req / "cuda-pro6000").read_text() == ""
    assert "https://github.com/login/device" in result.stderr


def test_prints_the_url_without_a_request_file(tmp_path):
    result = run_shim(tmp_path, "https://example.com")
    assert result.returncode == 1
    assert "https://example.com" in result.stderr


def test_refuses_non_web_urls(tmp_path):
    assert run_shim(tmp_path, "file:///etc/passwd").returncode == 3


def test_mounts_and_environ_are_per_key(tmp_path):
    assert [str(m.target) for m in browser_mounts("pro6000", tmp_path)] == [
        "/opt/labgpu/browser-pro6000",
        "/etc/profile.d/labgpu-browser-pro6000.sh",
        "/home/work/.labgpu-browser/pro6000",
    ]
    assert browser_environ("pro6000") == {"BROWSER": "/opt/labgpu/browser-pro6000/xdg-open"}
