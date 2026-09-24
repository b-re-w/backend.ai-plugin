from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from labgpu import devalloc, procmap
from labgpu.fraction import build_hami_environ, compute_limit, compute_limits
from labgpu.sizes import GiB, MiB, parse_size
from labgpu.spot.hostinfo import CpuSampler, parse_cpu_usage_usec

CID = "a" * 64


def test_parse_size():
    assert parse_size("16g") == 16 * GiB
    assert parse_size("512MiB") == 512 * MiB
    assert parse_size("1.5G") == int(1.5 * GiB)
    assert parse_size(123) == 123
    with pytest.raises(ValueError):
        parse_size("lots")


def test_whole_device_has_no_limits():
    lim = compute_limit(
        local_index=0, device_id="0", uuid="GPU-a", share=Decimal(1),
        shares_per_device=Decimal(1), total_memory=40 * GiB,
    )
    assert lim.is_whole and lim.sm_limit_percent is None


def test_half_device_limits():
    lim = compute_limit(
        local_index=0, device_id="0", uuid="GPU-a", share=Decimal("0.5"),
        shares_per_device=Decimal(1), total_memory=40 * GiB, reserved_memory=256 * MiB,
    )
    assert lim.memory_limit_bytes == 20 * GiB - 256 * MiB
    assert lim.sm_limit_percent == 50


def test_limits_with_custom_shares_per_device_and_rounding():
    lim = compute_limit(
        local_index=0, device_id="0", uuid="GPU-a", share=Decimal("0.33"),
        shares_per_device=Decimal(4), total_memory=10 * GiB,
    )
    assert lim.sm_limit_percent == 9  # ceil(8.25)
    assert lim.memory_limit_bytes % MiB == 0


def test_mixed_allocation_env_uses_local_indices():
    limits = compute_limits(
        {"3": Decimal("1"), "1": Decimal("0.25"), "2": Decimal("0")},
        shares_per_device=Decimal(1),
        total_memory_by_device={"1": 16 * GiB, "3": 16 * GiB, "2": 16 * GiB},
        uuid_by_device={"1": "GPU-1", "2": "GPU-2", "3": "GPU-3"},
    )
    assert [(lim.local_index, lim.device_id) for lim in limits] == [(0, "1"), (1, "3")]
    env = build_hami_environ(limits, sm_limit=True)
    assert env["CUDA_DEVICE_MEMORY_LIMIT_0"] == "4096m"
    assert "CUDA_DEVICE_MEMORY_LIMIT_1" not in env
    assert env["CUDA_DEVICE_SM_LIMIT"] == "25"
    assert build_hami_environ(limits[1:], sm_limit=True) == {}
    assert "CUDA_DEVICE_SM_LIMIT" not in build_hami_environ(limits, sm_limit=False)


def test_devalloc_accepts_both_shapes():
    legacy = {"cuda.shares": {"0": Decimal("0.5"), "1": Decimal(0)}}
    unit = SimpleNamespace(device_id="0", amounts={"cuda.shares": Decimal("0.5")})
    modern = SimpleNamespace(units=[unit])
    assert devalloc.amounts_for_slot(legacy, "cuda.shares")["0"] == Decimal("0.5")
    assert devalloc.amounts_for_slot(modern, "cuda.shares") == {"0": Decimal("0.5")}
    assert devalloc.active_device_ids(legacy) == ["0"]
    with pytest.raises(TypeError):
        devalloc.normalize(42)


@pytest.mark.parametrize(
    "text",
    [
        f"0::/system.slice/docker-{CID}.scope\n",
        f"12:memory:/docker/{CID}\n0::/\n",
        f"0::/kubepods/burstable/pod1/{CID}\n",
    ],
)
def test_parse_cgroup(text):
    assert procmap.parse_cgroup(text) == CID


def test_parse_cgroup_host_process():
    assert procmap.parse_cgroup("0::/user.slice/user-1000.slice/session-3.scope\n") is None


def test_cpu_stat():
    assert parse_cpu_usage_usec("usage_usec 1500\nuser_usec 1000\n") == 1500


def test_cpu_sampler(tmp_path: Path):
    proc = tmp_path / "proc" / "42"
    proc.mkdir(parents=True)
    (proc / "cgroup").write_text(f"0::/system.slice/docker-{CID}.scope\n")
    cg = tmp_path / "cg" / "system.slice" / f"docker-{CID}.scope"
    cg.mkdir(parents=True)
    sampler = CpuSampler(tmp_path / "proc", tmp_path / "cg")
    (cg / "cpu.stat").write_text("usage_usec 0\n")
    assert sampler.sample(CID, 42, 100.0) is None
    (cg / "cpu.stat").write_text("usage_usec 2000000\n")
    assert sampler.sample(CID, 42, 102.0) == pytest.approx(1.0)
    assert sampler.sample(CID, 999, 104.0) is None
