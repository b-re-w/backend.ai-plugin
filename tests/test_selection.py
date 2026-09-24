import json
import os

import pytest

from labgpu.nvml import FakeNvmlReader, NvmlError, open_reader
from labgpu.selection import GpuClaimConflict, GpuSelector, claim_gpus, model_matches, validate_key
from labgpu.sizes import GiB
from labgpu.spot.jobspec import JobSpec, JobSpecError
from labgpu.spot.model import GpuState, GpuVerdict, QueuedJob
from labgpu.spot.planner import plan

PRIMARY = {
    "driver": "fake",
    "gpus": [
        {"uuid": "GPU-p5000-72", "name": "NVIDIA RTX PRO 5000 Blackwell", "memory": "72g"},
        {"uuid": "GPU-p6000-1", "name": "NVIDIA RTX PRO 6000 Blackwell", "memory": "96g"},
        {"uuid": "GPU-p6000-2", "name": "NVIDIA RTX PRO 6000 Blackwell", "memory": "96g",
         "processes": [{"pid": 7, "mem": "4g", "sm": 30, "container": "c" * 64}]},
        {"uuid": "GPU-a6000", "name": "NVIDIA RTX A6000", "memory": "48g"},
    ],
}


def test_selector_by_model_and_memory():
    pro5000_large = GpuSelector.from_config({"model_pattern": "*PRO 5000*", "min_memory": "60g"})
    pro5000_small = GpuSelector.from_config({"model_pattern": "*pro 5000*", "max_memory": "60g"})
    assert pro5000_large.matches("NVIDIA RTX PRO 5000 Blackwell", 72 * GiB)
    assert not pro5000_large.matches("NVIDIA RTX PRO 5000 Blackwell", 48 * GiB)
    assert pro5000_small.matches("NVIDIA RTX PRO 5000 Blackwell", 48 * GiB)
    assert not pro5000_small.matches("NVIDIA RTX PRO 6000 Blackwell", 48 * GiB)
    both = GpuSelector.from_config({"model_pattern": "*A6000*, *PRO 6000*", "device_mask": "GPU-x"})
    assert both.matches("NVIDIA RTX A6000", GiB) and both.matches("NVIDIA RTX PRO 6000", GiB)
    assert not both.matches("NVIDIA RTX A6000", GiB, "GPU-x")
    assert GpuSelector().matches("anything", 1)


def test_validate_key():
    assert validate_key("pro6000") == "pro6000"
    for bad in ("PRO6000", "a.b", "", "cpu", "-x"):
        with pytest.raises(ValueError):
            validate_key(bad)


def test_claims_detect_overlap_within_process(tmp_path):
    claim_gpus(["GPU-a", "GPU-b"], "gpu_slot_1", tmp_path)
    claim_gpus(["GPU-a"], "gpu_slot_1", tmp_path)  # re-claim by the same plugin is fine
    with pytest.raises(GpuClaimConflict):
        claim_gpus(["GPU-b"], "gpu_slot_2", tmp_path)
    # A file left by a previous agent process is stale and gets overwritten.
    (tmp_path / "GPU-c").write_text("999999999:gpu_slot_3")
    claim_gpus(["GPU-c"], "gpu_slot_2", tmp_path)
    assert (tmp_path / "GPU-c").read_text() == f"{os.getpid()}:gpu_slot_2"


def test_fake_reader(tmp_path, monkeypatch):
    path = tmp_path / "gpus.json"
    path.write_text(json.dumps(PRIMARY))
    reader = FakeNvmlReader(path)
    gpus = reader.list_gpus()
    assert [g.uuid for g in gpus][:2] == ["GPU-p5000-72", "GPU-p6000-1"]
    snap = reader.snapshot(2)
    assert snap.used_memory == 4 * GiB and snap.processes[0].sm_util == 30
    assert reader.container_of_pids([7, 8]) == {7: "c" * 64, 8: None}
    # The file is re-read on every call so tests can change it live.
    data = json.loads(path.read_text())
    data["gpus"][0]["fail"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(NvmlError):
        reader.snapshot(0)
    monkeypatch.setenv("LABGPU_FAKE_NVML", str(path))
    assert open_reader().is_fake


def test_planner_respects_gpu_models():
    def v(uuid, model):
        return GpuVerdict(uuid, GpuState.LENDABLE, True, False, (), 40 * GiB, 0, model)

    verdicts = {"A": v("A", "NVIDIA RTX A6000"), "B": v("B", "NVIDIA RTX PRO 6000 Blackwell")}
    queue = [QueuedJob(1, 0, 0.0, 0, GiB, ("*PRO 6000*",))]
    p = plan(verdicts, [], queue, ram_budget=100 * GiB, enforced=True)
    assert [(launch.job_id, launch.gpu_uuid) for launch in p.launches] == [(1, "B")]


def test_jobspec_gpu_models():
    spec = JobSpec.from_dict({"name": "n", "image": "i", "command": ["x"], "gpu_models": ["*A6000*"]})
    assert JobSpec.from_json(spec.to_json()).gpu_models == ("*A6000*",)
    assert model_matches("NVIDIA RTX A6000", spec.gpu_models)
    with pytest.raises(JobSpecError):
        JobSpec.from_dict({"name": "n", "image": "i", "command": ["x"], "gpu_models": "*A6000*"})
