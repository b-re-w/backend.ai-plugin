import pytest

from labgpu.nvml import GpuInfo, GpuProcess, GpuSnapshot
from labgpu.sizes import GiB, MiB
from labgpu.spot.config import Config, IdleConfig, ReclaimConfig
from labgpu.spot.detector import GpuTracker
from labgpu.spot.docker import OwnerContainer, parse_gpu_refs
from labgpu.spot.model import ClassifiedProcess, GpuObservation, GpuState, ProcKind
from labgpu.spot.observer import observe, resolve_gpu_refs

IDLE = IdleConfig(idle_minutes=10, owner_util_threshold=5, owner_mem_delta_mib=512,
                  owner_cpu_threshold=0.5, unclaimed_grace_seconds=60)
RECLAIM = ReclaimConfig(mem_reserve_mib=2048)
OWNER = "o" * 64


def obs(*, util=0, owner_mem=4 * GiB, used=None, owners=(OWNER,), cpu=0.0, unknown=False, ok=True):
    procs = []
    if owners:
        procs.append(ClassifiedProcess(1, ProcKind.OWNER, OWNER, owner_mem, util))
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


def test_lent_and_quiet_stays_lent():
    tr = GpuTracker(obs(), 0)
    v = step(tr, obs(), 700, lent=True)
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


# ---- docker ----

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
    pid_map = {10: OWNER, 20: None, 30: None}
    a, b = observe(GPUS, snaps.__getitem__, owners, lambda pids: pid_map, {OWNER: 0.1})
    assert [p.kind for p in a.processes] == [ProcKind.OWNER, ProcKind.UNKNOWN]
    assert a.owner_memory == 4 * GiB and a.owner_cpu_cores == 0.1
    assert b.has_unknown and not b.owner_containers


def test_config_from_dict():
    cfg = Config.from_dict({"idle": {"idle_minutes": 5}})
    assert cfg.idle.idle_seconds == 300
    with pytest.raises(ValueError):
        Config.from_dict({"idle": {"typo": 1}})
    with pytest.raises(ValueError):
        Config.from_dict({"spot": {}})


# ---- ignored host processes (SPEC 2.1) ----

def test_host_xorg_is_ignored_but_not_inside_a_container():
    snaps = {
        0: GpuSnapshot(GPUS[0], 5 * GiB, 1, (
            GpuProcess(10, 4 * GiB, 0),                 # owner
            GpuProcess(11, 200 * MiB, 3, name="Xorg"),  # host display server
        )),
        1: GpuSnapshot(GPUS[1], GiB, 0, (
            GpuProcess(30, GiB, 0, name="Xorg"),        # renamed process inside some container
            GpuProcess(31, 100 * MiB, 0, name="python"),  # unlisted host process
        )),
    }
    owners = [OwnerContainer(OWNER, 100, ("GPU-a",))]
    pid_map = {10: OWNER, 11: None, 30: "c" * 64, 31: None}
    a, b = observe(GPUS, snaps.__getitem__, owners, lambda pids: pid_map, {},
                   ignored_names=("Xorg",))
    assert [p.kind for p in a.processes] == [ProcKind.OWNER, ProcKind.IGNORED]
    assert not a.has_unknown
    assert [p.kind for p in b.processes] == [ProcKind.UNKNOWN, ProcKind.UNKNOWN]


def test_ignored_processes_config():
    assert Config().idle.ignored_processes == ("Xorg",)
    cfg = Config.from_dict({"idle": {"ignored_processes": ["Xorg", "gnome-shell"]}})
    assert cfg.idle.ignored_processes == ("Xorg", "gnome-shell")
    assert Config.from_dict({"idle": {"ignored_processes": []}}).idle.ignored_processes == ()


def test_process_name(tmp_path):
    from labgpu.procmap import process_name
    (tmp_path / "42").mkdir()
    (tmp_path / "42" / "comm").write_text("Xorg\n")
    assert process_name(42, tmp_path) == "Xorg"
    assert process_name(43, tmp_path) == ""
