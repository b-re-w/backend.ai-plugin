"""
Exercise CUDAFracPlugin against the real Backend.AI agent classes with a fake NVML.

Skipped unless `ai.backend.agent` is importable (Linux + Backend.AI installed or on PYTHONPATH).
"""

import asyncio
import json
from decimal import Decimal

import pytest

pytest.importorskip("ai.backend.agent.resources")

from ai.backend.common.types import SlotName  # noqa: E402

from labgpu.accelerator.plugin import (  # noqa: E402
    AllocationMode,
    CUDAFracPlugin,
    GpuSlotPlugin1,
    GpuSlotPlugin2,
    PluginNotConfigured,
)
from labgpu.selection import GpuClaimConflict  # noqa: E402
from labgpu.nvml import GpuInfo, GpuProcess, GpuSnapshot  # noqa: E402
from labgpu.sizes import GiB  # noqa: E402

GPUS = [
    GpuInfo(0, "GPU-aaaa", "NVIDIA A100", 40 * GiB, "00000000:01:00.0"),
    GpuInfo(1, "GPU-bbbb", "NVIDIA A100", 40 * GiB, "00000000:02:00.0"),
]
CID_A = "a" * 64
CID_B = "b" * 64


class FakeNvml:
    def list_gpus(self):
        return GPUS

    def snapshot(self, index):
        procs = {
            0: (GpuProcess(10, 8 * GiB, 30), GpuProcess(20, 4 * GiB, 10)),
            1: (GpuProcess(30, 2 * GiB, 5),),
        }[index]
        return GpuSnapshot(GPUS[index], sum(p.used_memory for p in procs), 40, procs)

    def driver_version(self):
        return "550.54"

    def close(self):
        pass


def make_plugin(**cfg) -> CUDAFracPlugin:
    p = CUDAFracPlugin({k: str(v) for k, v in cfg.items()}, {})
    p._read_config(p.plugin_config)
    p._nvml = FakeNvml()
    p.enabled = True
    p.fraction_enforced = p.mode == AllocationMode.FRACTIONAL
    p._set_slot_types()
    asyncio.run(p.list_devices())
    return p


def env_of(args):
    return dict(e.split("=", 1) for e in args["Env"])


def test_plugin_is_concrete_and_reports_slots():
    p = make_plugin()
    assert asyncio.run(p.available_slots()) == {SlotName("cuda.shares"): Decimal(2)}
    meta = p.get_metadata()
    assert meta["slot_name"] == "cuda.shares" and meta["display_unit"] == "fGPU"
    from labgpu.accelerator.plugin import short_model_name
    assert short_model_name("NVIDIA RTX PRO 5000 Blackwell") == "PRO5000"
    assert short_model_name("NVIDIA GeForce RTX 4050 Laptop GPU") == "RTX4050"
    assert short_model_name("NVIDIA RTX A6000") == "A6000"


def test_fractional_allocation_end_to_end():
    p = make_plugin()
    alloc_map = asyncio.run(p.create_alloc_map())
    alloc = alloc_map.allocate({SlotName("cuda.shares"): Decimal("0.5")})
    args = asyncio.run(p.generate_docker_args(None, alloc))
    req = args["HostConfig"]["DeviceRequests"][0]
    assert req["Driver"] == "nvidia" and len(req["DeviceIDs"]) == 1
    assert req["DeviceIDs"][0].startswith("GPU-")
    env = env_of(args)
    assert env["CUDA_DEVICE_MEMORY_LIMIT_0"] == "20480m"
    assert env["CUDA_DEVICE_SM_LIMIT"] == "50"
    assert env["LABGPU_DEVICE_UUIDS"] == req["DeviceIDs"][0]
    data = asyncio.run(p.generate_resource_data(alloc))
    assert data["CUDA_RESOURCE_VIRTUALIZED"] == "1"
    attached = asyncio.run(p.get_attached_devices(alloc))
    assert attached[0]["data"]["mem"] == 20 * GiB
    assert asyncio.run(p.get_hooks("ubuntu22.04", "x86_64")) == [p.hook_path]


def test_fill_keeps_a_request_on_one_gpu_but_does_not_pack_sessions():
    p = make_plugin(allocation_strategy="fill")
    alloc_map = asyncio.run(p.create_alloc_map())
    whole = alloc_map.allocate({SlotName("cuda.shares"): Decimal("1")})
    assert len(whole[SlotName("cuda.shares")]) == 1
    # Upstream FILL starts from the most free GPU, so the next session goes elsewhere
    # (SPEC 4, open question on bin packing).
    other = alloc_map.allocate({SlotName("cuda.shares"): Decimal("0.25")})
    assert set(other[SlotName("cuda.shares")]).isdisjoint(whole[SlotName("cuda.shares")])


def test_one_and_a_half_gpus():
    p = make_plugin()
    alloc_map = asyncio.run(p.create_alloc_map())
    alloc = alloc_map.allocate({SlotName("cuda.shares"): Decimal("1.5")})
    env = env_of(asyncio.run(p.generate_docker_args(None, alloc)))
    assert len(env["LABGPU_DEVICE_UUIDS"].split(",")) == 2
    limited = [k for k in env if k.startswith("CUDA_DEVICE_MEMORY_LIMIT_")]
    assert len(limited) == 1


def test_discrete_mode_has_no_limits():
    p = make_plugin(allocation_mode="discrete")
    alloc_map = asyncio.run(p.create_alloc_map())
    alloc = alloc_map.allocate({SlotName("cuda.device"): Decimal(1)})
    env = env_of(asyncio.run(p.generate_docker_args(None, alloc)))
    assert not any(k.startswith("CUDA_DEVICE_") for k in env)
    assert p.get_metadata()["slot_name"] == "cuda.device"


def test_container_measures_are_per_process(monkeypatch):
    p = make_plugin()
    monkeypatch.setattr(
        FakeNvml, "container_of_pids", lambda self, pids: {10: CID_A, 20: CID_B, 30: CID_A},
        raising=False,
    )
    measures = asyncio.run(p.gather_container_measures(None, [CID_A, CID_B]))
    mem = {m.key: m for m in measures}["cuda_mem"].per_container
    util = {m.key: m for m in measures}["cuda_util"].per_container
    assert mem[CID_A].value == 10 * GiB and mem[CID_B].value == 4 * GiB
    assert util[CID_A].value == 35 and util[CID_A].capacity == 200
    assert util[CID_B].value == 10 and util[CID_B].capacity == 100


PRIMARY = {
    "driver": "fake",
    "gpus": [
        {"uuid": "GPU-p5000-72", "name": "NVIDIA RTX PRO 5000 Blackwell", "memory": "72g"},
        {"uuid": "GPU-p6000-1", "name": "NVIDIA RTX PRO 6000 Blackwell", "memory": "96g"},
        {"uuid": "GPU-p6000-2", "name": "NVIDIA RTX PRO 6000 Blackwell", "memory": "96g"},
        {"uuid": "GPU-a6000", "name": "NVIDIA RTX A6000", "memory": "48g"},
    ],
}


@pytest.fixture
def fake_primary(tmp_path, monkeypatch):
    path = tmp_path / "gpus.json"
    path.write_text(json.dumps(PRIMARY))
    monkeypatch.setenv("LABGPU_FAKE_NVML", str(path))
    monkeypatch.setenv("LABGPU_CLAIM_DIR", str(tmp_path / "claims"))
    hook = tmp_path / "libvgpu.so"
    hook.write_bytes(b"")
    return hook


def init_plugin(cls, hook, **cfg):
    p = cls({"hook_path": str(hook), **cfg}, {})
    asyncio.run(p.init())
    return p


def test_per_model_slots_through_real_init(fake_primary):
    p6000 = init_plugin(GpuSlotPlugin1, fake_primary, key="pro6000", model_pattern="*PRO 6000*")
    a6000 = init_plugin(GpuSlotPlugin2, fake_primary, key="a6000", model_pattern="*A6000*")
    assert p6000.is_fake and p6000.fraction_enforced
    assert asyncio.run(p6000.available_slots()) == {SlotName("pro6000.shares"): Decimal(2)}
    assert asyncio.run(a6000.available_slots()) == {SlotName("a6000.shares"): Decimal(1)}
    assert p6000.get_metadata()["human_readable_name"] == "RTX PRO 6000 Blackwell"
    # Distinct units: the WebUI names accelerator choices by display_unit.
    assert p6000.get_metadata()["display_unit"] == "PRO6000"
    assert a6000.get_metadata()["display_unit"] == "A6000"
    # The agent's affinity map looks devices up by device_name == plugin key.
    assert {d.device_name for d in asyncio.run(p6000.list_devices())} == {"pro6000"}
    alloc = asyncio.run(p6000.create_alloc_map()).allocate({SlotName("pro6000.shares"): Decimal("0.5")})
    args = asyncio.run(p6000.generate_docker_args(None, alloc))
    assert "HostConfig" not in args  # fake mode never attaches GPUs
    env = env_of(args)
    assert env["CUDA_DEVICE_MEMORY_LIMIT_0"] == "49152m"
    assert env["LABGPU_DEVICE_UUIDS"].startswith("GPU-p6000-")
    measures = asyncio.run(p6000.gather_node_measures(None))
    assert {str(m.key) for m in measures} == {"pro6000_mem", "pro6000_util"}


def test_overlapping_plugins_are_refused(fake_primary):
    init_plugin(GpuSlotPlugin1, fake_primary, key="pro6000", model_pattern="*PRO 6000*")
    with pytest.raises(GpuClaimConflict):
        init_plugin(CUDAFracPlugin, fake_primary)  # would take every GPU


def test_unconfigured_slot_plugin_is_skipped(fake_primary):
    with pytest.raises(PluginNotConfigured):
        init_plugin(GpuSlotPlugin2, fake_primary)


def test_memory_bounds_split_same_model(tmp_path, monkeypatch):
    data = {"gpus": [
        {"uuid": "GPU-l", "name": "NVIDIA RTX PRO 5000 Blackwell", "memory": "72g"},
        {"uuid": "GPU-s", "name": "NVIDIA RTX PRO 5000 Blackwell", "memory": "48g"},
    ]}
    path = tmp_path / "g.json"
    path.write_text(json.dumps(data))
    monkeypatch.setenv("LABGPU_FAKE_NVML", str(path))
    monkeypatch.setenv("LABGPU_CLAIM_DIR", str(tmp_path / "claims"))
    hook = tmp_path / "h.so"
    hook.write_bytes(b"")
    large = init_plugin(GpuSlotPlugin1, hook, key="pro5000l", model_pattern="*PRO 5000*", min_memory="60g")
    small = init_plugin(GpuSlotPlugin2, hook, key="pro5000", model_pattern="*PRO 5000*", max_memory="60g")
    assert [d.uuid for d in asyncio.run(large.list_devices())] == ["GPU-l"]
    assert [d.uuid for d in asyncio.run(small.list_devices())] == ["GPU-s"]


def test_lending_measures_per_session(fake_primary, tmp_path):
    import time
    status = tmp_path / "status.json"
    now = time.time()
    status.write_text(json.dumps({"updated_at": now, "gpus": [
        {"uuid": "GPU-p6000-1", "lent_job": 7, "lent_since": now - 600},
        {"uuid": "GPU-p6000-2", "lent_job": None, "lent_since": None},
    ]}))
    p = init_plugin(GpuSlotPlugin1, fake_primary, key="pro6000", model_pattern="*PRO 6000*",
                    spot_status_path=str(status))

    async def fake_gpus_of(container_ids):
        return {"c1": ["GPU-p6000-1"], "c2": ["GPU-p6000-2"], "c3": ["GPU-a6000"]}

    p._gpus_of = fake_gpus_of
    measures = {str(m.key): m for m in asyncio.run(p.gather_container_measures(None, ["c1", "c2", "c3"]))}
    lent = measures["pro6000_lent"].per_container
    assert (lent["c1"].value, lent["c1"].capacity) == (1, 1)
    assert (lent["c2"].value, lent["c2"].capacity) == (0, 1)
    assert "c3" not in lent  # the A6000 belongs to another plugin
    assert int(measures["pro6000_lent_since"].per_container["c1"].value) == int(now - 600)
    assert measures["pro6000_lent_since"].per_container["c1"].capacity == 1

    status.write_text(json.dumps({"updated_at": now - 3600, "gpus": []}))  # stale file
    keys = {str(m.key) for m in asyncio.run(p.gather_container_measures(None, ["c1"]))}
    assert "pro6000_lent" not in keys
