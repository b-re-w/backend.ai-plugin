from pathlib import Path

import pytest

from labgpu.nvml import GpuInfo, GpuProcess, GpuSnapshot
from labgpu.sizes import GiB, MiB
from labgpu.spot.config import Config, IdleConfig, ReclaimConfig, SpotConfig
from labgpu.spot.detector import GpuTracker
from labgpu.spot.docker import OwnerContainer, SpotContainer, build_run_args, parse_gpu_refs
from labgpu.spot.jobs import JobState, JobStore
from labgpu.spot.jobspec import JobSpec, JobSpecError
from labgpu.spot.model import (
    ClassifiedProcess,
    GpuObservation,
    GpuState,
    GpuVerdict,
    ProcKind,
    QueuedJob,
    RunningSpot,
)
from labgpu.spot.observer import observe, resolve_gpu_refs
from labgpu.spot.planner import plan

IDLE = IdleConfig(idle_minutes=10, owner_util_threshold=5, owner_mem_delta_mib=512,
                  owner_cpu_threshold=0.5, unclaimed_grace_seconds=60)
RECLAIM = ReclaimConfig(grace_seconds=30, mem_reserve_mib=2048)
OWNER = "o" * 64
SPOT = "s" * 64


def obs(*, util=0, owner_mem=4 * GiB, used=None, owners=(OWNER,), cpu=0.0, unknown=False, ok=True,
        spot_mem=0):
    procs = []
    if owners:
        procs.append(ClassifiedProcess(1, ProcKind.OWNER, OWNER, owner_mem, util))
    if spot_mem:
        procs.append(ClassifiedProcess(2, ProcKind.SPOT, SPOT, spot_mem, 90))
    if unknown:
        procs.append(ClassifiedProcess(3, ProcKind.UNKNOWN, None, 100 * MiB, 0))
    total_used = used if used is not None else sum(p.used_memory for p in procs)
    return GpuObservation("GPU-a", 0, ok, 40 * GiB, total_used, tuple(procs),
                          frozenset(owners), cpu, None if ok else "boom")


def step(tracker, o, t, **kw):
    return tracker.update(o, t, IDLE, RECLAIM, **{"lent": False, **kw})


# ---- detector ----

def test_idle_owner_becomes_lendable_after_idle_minutes():
    tr = GpuTracker(obs(), 0)
    assert step(tr, obs(), 60).state is GpuState.IDLE
    v = step(tr, obs(), 600)
    assert v.state is GpuState.LENDABLE and v.lendable
    assert v.lendable_memory == 40 * GiB - 4 * GiB - 2 * GiB


def test_activity_resets_idle_timer():
    tr = GpuTracker(obs(), 0)
    assert step(tr, obs(util=50), 300).state is GpuState.BUSY
    assert step(tr, obs(), 600).state is GpuState.IDLE
    assert step(tr, obs(), 900).state is GpuState.LENDABLE


@pytest.mark.parametrize(
    "active_obs",
    [
        obs(util=6),
        obs(owner_mem=5 * GiB),
        obs(cpu=2.0),
        obs(unknown=True),
    ],
)
def test_each_activity_rule_reclaims_a_lent_gpu(active_obs):
    tr = GpuTracker(obs(), 0)
    step(tr, obs(), 600)
    v = step(tr, active_obs, 610, lent=True)
    assert v.must_reclaim and v.state is GpuState.RECLAIMING and v.reasons


def test_new_owner_container_reclaims_unclaimed_gpu():
    tr = GpuTracker(obs(owners=()), 0)
    assert step(tr, obs(owners=()), 61).lendable
    v = step(tr, obs(owners=(OWNER,), owner_mem=0), 70, lent=True)
    assert v.must_reclaim and "new owner" in v.reasons[0]


def test_memory_baseline_follows_release_down():
    tr = GpuTracker(obs(owner_mem=10 * GiB), 0)
    step(tr, obs(owner_mem=2 * GiB), 10)
    # Growing from the lowered baseline by more than the delta counts as activity.
    assert step(tr, obs(owner_mem=3 * GiB), 20).state is GpuState.BUSY


def test_low_free_memory_reclaims():
    tr = GpuTracker(obs(), 0)
    v = step(tr, obs(used=40 * GiB - 512 * MiB), 700, lent=True)
    assert v.must_reclaim


def test_lent_and_quiet_stays_lent_and_spot_usage_is_ignored():
    tr = GpuTracker(obs(), 0)
    v = step(tr, obs(spot_mem=20 * GiB), 700, lent=True)
    assert v.state is GpuState.LENT and not v.must_reclaim


def test_unknown_observation_reclaims_and_resets():
    tr = GpuTracker(obs(), 0)
    v = step(tr, obs(ok=False), 700, lent=True)
    assert v.state is GpuState.UNKNOWN and v.must_reclaim
    assert step(tr, obs(), 701).state is GpuState.IDLE  # timer restarted


def test_blocked_gpu_is_not_lent_and_is_reclaimed():
    tr = GpuTracker(obs(), 0)
    assert not step(tr, obs(), 700, blocked="node paused").lendable
    assert step(tr, obs(), 701, lent=True, blocked="node paused").must_reclaim


def test_already_reclaiming_does_not_reissue():
    tr = GpuTracker(obs(), 0)
    v = step(tr, obs(util=90), 700, lent=True, reclaiming=True)
    assert v.state is GpuState.RECLAIMING and not v.must_reclaim


# ---- planner ----

def verdict(uuid, lendable=True, mem=20 * GiB, must=False):
    state = GpuState.LENDABLE if lendable else GpuState.BUSY
    return GpuVerdict(uuid, state, lendable, must, ("why",) if must else (), mem if lendable else 0)


def qjob(i, prio=0, gpu_mem=0, ram=GiB, t=None):
    return QueuedJob(i, prio, t if t is not None else float(i), gpu_mem, ram)


def test_plan_reclaims_and_backfills():
    verdicts = {"A": verdict("A", mem=8 * GiB), "B": verdict("B", lendable=False, must=True)}
    running = [RunningSpot(9, "B", "c9")]
    queue = [qjob(1, gpu_mem=16 * GiB), qjob(2, gpu_mem=4 * GiB)]
    p = plan(verdicts, running, queue, ram_budget=100 * GiB, enforced=True)
    assert [r.job_id for r in p.reclaims] == [9]
    assert [(launch.job_id, launch.gpu_uuid) for launch in p.launches] == [(2, "A")]
    assert p.launches[0].gpu_mem_limit == 8 * GiB


def test_plan_priority_ram_budget_and_enforcement():
    verdicts = {"A": verdict("A"), "B": verdict("B")}
    queue = [qjob(1, ram=10 * GiB), qjob(2, prio=5, ram=10 * GiB)]
    p = plan(verdicts, [], queue, ram_budget=15 * GiB, enforced=True)
    assert [launch.job_id for launch in p.launches] == [2]
    assert plan(verdicts, [], queue, ram_budget=100 * GiB, enforced=False).launches == ()


def test_plan_skips_occupied_gpu_and_reclaims_vanished_gpu():
    verdicts = {"A": verdict("A")}
    running = [RunningSpot(1, "A", "c1"), RunningSpot(2, "GONE", "c2")]
    p = plan(verdicts, running, [qjob(3)], ram_budget=100 * GiB, enforced=True)
    assert p.launches == ()
    assert [r.job_id for r in p.reclaims] == [2]


# ---- job spec & docker args ----

def spec_dict(**kw):
    base = {"name": "t", "image": "img:1", "command": ["python", "train.py"],
            "gpu_mem": "8g", "mounts": [{"src": "/vfroot/u/proj", "dst": "/home/work/proj"}]}
    return {**base, **kw}


def test_jobspec_validation():
    spec = JobSpec.from_dict(spec_dict())
    assert spec.gpu_mem == 8 * GiB
    spec.validate_mounts([Path("/vfroot")], resolve=False)
    with pytest.raises(JobSpecError):
        JobSpec.from_dict(spec_dict(mounts=[{"src": "/vfroot/../etc", "dst": "/x"}])).validate_mounts(
            [Path("/vfroot")], resolve=False)
    with pytest.raises(JobSpecError):
        JobSpec.from_dict(spec_dict(command="python train.py"))
    with pytest.raises(JobSpecError):
        JobSpec.from_dict(spec_dict(surprise=1))
    assert JobSpec.from_json(spec.to_json()) == spec


def test_build_run_args():
    spec = JobSpec.from_dict(spec_dict(env={"A": "1"}, entrypoint=""))
    cfg = SpotConfig(hook_path=Path("/opt/labgpu/lib/libvgpu.so"))
    argv = build_run_args(job_id=7, attempt=2, spec=spec, uid=1000, gid=1000, gpu_uuid="GPU-a",
                          gpu_mem_limit=10 * GiB, ram=16 * GiB, cfg=cfg, enforce=True)
    joined = " ".join(argv)
    assert "--gpus device=GPU-a" in joined
    assert "--oom-score-adj 1000" in joined
    assert "--user 1000:1000" in joined
    assert "CUDA_DEVICE_MEMORY_LIMIT_0=10240m" in joined
    assert "LD_PRELOAD=/opt/labgpu/libvgpu.so" in joined
    assert "--name labgpu-spot-7-2" in joined
    assert argv[-3:] == ["img:1", "python", "train.py"]
    assert "LD_PRELOAD" not in " ".join(build_run_args(
        job_id=7, attempt=2, spec=spec, uid=1000, gid=1000, gpu_uuid="GPU-a",
        gpu_mem_limit=GiB, ram=GiB, cfg=cfg, enforce=False))
    with pytest.raises(ValueError):
        build_run_args(job_id=7, attempt=1, spec=spec, uid=0, gid=0, gpu_uuid="GPU-a",
                       gpu_mem_limit=GiB, ram=GiB, cfg=cfg, enforce=True)


def test_parse_gpu_refs_precedence():
    assert parse_gpu_refs({"Config": {"Env": ["LABGPU_DEVICE_UUIDS=GPU-a,GPU-b"]}}) == ("GPU-a", "GPU-b")
    assert parse_gpu_refs({"HostConfig": {"DeviceRequests": [
        {"Driver": "nvidia", "DeviceIDs": ["0", "2"]}]}}) == ("0", "2")
    assert parse_gpu_refs({"Config": {"Env": ["NVIDIA_VISIBLE_DEVICES=all"]}}) == ("all",)
    assert parse_gpu_refs({"Config": {"Env": ["NVIDIA_VISIBLE_DEVICES=void"]}}) == ()


# ---- observer ----

GPUS = [GpuInfo(0, "GPU-a", "A100", 40 * GiB, "00000000:01:00.0"),
        GpuInfo(1, "GPU-b", "A100", 40 * GiB, "00000000:02:00.0")]


def test_resolve_and_classify():
    assert resolve_gpu_refs(["1"], GPUS) == {"GPU-b"}
    assert resolve_gpu_refs(["all"], GPUS) == {"GPU-a", "GPU-b"}
    snaps = {
        0: GpuSnapshot(GPUS[0], 30 * GiB, 50, (GpuProcess(10, 4 * GiB, 1), GpuProcess(20, 20 * GiB, 80))),
        1: GpuSnapshot(GPUS[1], 1 * GiB, 0, (GpuProcess(30, GiB, 0),)),
    }
    owners = [OwnerContainer(OWNER, 100, ("GPU-a",))]
    spots = [SpotContainer(SPOT, "labgpu-spot-1-1", 1, "GPU-a", True, None)]
    pid_map = {10: OWNER, 20: SPOT, 30: None}
    result = observe(GPUS, snaps.__getitem__, owners, spots, lambda pids: pid_map, {OWNER: 0.1})
    a, b = result
    assert [p.kind for p in a.processes] == [ProcKind.OWNER, ProcKind.SPOT]
    assert a.owner_memory == 4 * GiB and a.owner_cpu_cores == 0.1
    assert b.has_unknown and not b.owner_containers


# ---- job store ----

def test_job_lifecycle(tmp_path):
    clock = iter(range(1000)).__next__
    store = JobStore(tmp_path / "db", clock=lambda: float(clock()))
    spec = JobSpec.from_dict(spec_dict())
    jid = store.submit(spec, 1000, 1000)
    assert store.queued(GiB)[0].gpu_mem == 8 * GiB
    assert store.mark_running(jid, "GPU-a", "c1") == 1
    store.mark_preempting(jid, "owner back")
    assert store.active()[0].reclaiming
    assert store.finish(jid, 143, max_attempts=2) is JobState.QUEUED
    rec = store.get(jid)
    assert rec.preemptions == 1 and rec.gpu_uuid is None
    store.mark_running(jid, "GPU-b", "c2")
    store.mark_preempting(jid, "owner back")
    assert store.finish(jid, 143, max_attempts=2) is JobState.FAILED
    jid2 = store.submit(spec, 1000, 1000)
    store.mark_running(jid2, "GPU-a", "c3")
    assert store.finish(jid2, 0, max_attempts=2) is JobState.SUCCEEDED
    jid3 = store.submit(spec, 1000, 1000)
    store.mark_running(jid3, "GPU-a", "c4")
    assert store.finish(jid3, None, max_attempts=5, vanished=True) is JobState.QUEUED


def test_cancel_running_job_keeps_gpu_occupied_until_container_exits(tmp_path):
    store = JobStore(tmp_path / "db")
    jid = store.submit(JobSpec.from_dict(spec_dict()), 1000, 1000)
    store.mark_running(jid, "GPU-a", "c1")
    assert store.cancel(jid) == "c1"
    assert [(a.job_id, a.reclaiming) for a in store.active()] == [(jid, True)]
    assert store.finish(jid, 137, max_attempts=3) is JobState.CANCELLED
    assert store.active() == []


def test_pause_settings(tmp_path):
    store = JobStore(tmp_path / "db")
    store.set_paused(True, "GPU-a")
    store.set_paused(True)
    assert store.paused() == (True, frozenset({"GPU-a"}))
    store.set_paused(False, "GPU-a")
    store.set_paused(False)
    assert store.paused() == (False, frozenset())


def test_config_from_dict():
    cfg = Config.from_dict({"idle": {"idle_minutes": 5}, "spot": {"default_ram": "8g",
                            "allowed_mount_roots": ["/data"]}})
    assert cfg.idle.idle_seconds == 300
    assert cfg.spot.default_ram == 8 * GiB
    assert cfg.spot.allowed_mount_roots == (Path("/data"),)
    with pytest.raises(ValueError):
        Config.from_dict({"idle": {"typo": 1}})
