import json
from pathlib import Path

from labgpu.nvml import GpuInfo, GpuProcess, GpuSnapshot
from labgpu.sizes import GiB
from labgpu.spot.ckpt import device_map, driver_major
from labgpu.spot.config import Config, ControllerConfig, IdleConfig, SpotConfig
from labgpu.spot.daemon import Controller
from labgpu.spot.docker import OwnerContainer, SpotContainer, split_sessions
from labgpu.spot.model import GpuState, GpuVerdict, ProcKind
from labgpu.spot.observer import observe
from labgpu.spot.placement import Evict, Move, Park, ParkedSpot, Restore, RunningSpot, plan
from labgpu.spotstatus import parse_status, pick_spot_gpu, spot_capacity

P6K = "NVIDIA RTX PRO 6000"
A6K = "NVIDIA RTX A6000"


def v(uuid, state, model=P6K, must=False, mem=80 * GiB, reasons=()):
    return GpuVerdict(uuid, state, state is GpuState.LENDABLE, must, reasons,
                      mem if state is GpuState.LENDABLE else 0, 0, model)


# ---- placement ----

def test_spot_stays_on_quiet_lent_gpu():
    verdicts = {"A": v("A", GpuState.LENT), "B": v("B", GpuState.LENDABLE)}
    running = [RunningSpot("s1", frozenset({"A"}), ("A", "B"), 0)]
    assert plan(verdicts, running, [], now=10, can_checkpoint=True, park_seconds=300) == []


def test_owner_back_moves_spot_to_same_model_gpu():
    verdicts = {
        "A": v("A", GpuState.RECLAIMING, must=True, reasons=("owner SM util 90% > 5%",)),
        "B": v("B", GpuState.LENDABLE, model=A6K),  # other model: never a target
        "C": v("C", GpuState.LENDABLE, mem=10 * GiB),
        "D": v("D", GpuState.LENDABLE, mem=50 * GiB),
    }
    running = [RunningSpot("s1", frozenset({"A"}), ("A", "B", "C", "D"), 0)]
    [action] = plan(verdicts, running, [], now=10, can_checkpoint=True, park_seconds=300)
    assert action == Move("s1", "A", "D", "owner SM util 90% > 5%")


def test_no_target_parks_then_evicts_after_timeout():
    verdicts = {"A": v("A", GpuState.RECLAIMING, must=True), "B": v("B", GpuState.BUSY)}
    running = [RunningSpot("s1", frozenset({"A"}), ("A", "B"), 0)]
    [park] = plan(verdicts, running, [], now=10, can_checkpoint=True, park_seconds=300)
    assert isinstance(park, Park) and park.src == "A"
    parked = [ParkedSpot("s1", "A", ("A", "B"), 10)]
    assert plan(verdicts, [], parked, now=100, can_checkpoint=True, park_seconds=300) == []
    [ev] = plan(verdicts, [], parked, now=310, can_checkpoint=True, park_seconds=300)
    assert isinstance(ev, Evict) and ev.gpu is None


def test_parked_spot_returns_to_its_own_gpu_when_it_frees_up():
    verdicts = {"A": v("A", GpuState.LENDABLE), "B": v("B", GpuState.BUSY)}
    parked = [ParkedSpot("s1", "A", ("A", "B"), 10)]
    assert plan(verdicts, [], parked, now=20, can_checkpoint=True, park_seconds=300) == [
        Restore("s1", "A", "A")
    ]


def test_without_cuda_checkpoint_evicts_right_away():
    verdicts = {"A": v("A", GpuState.RECLAIMING, must=True)}
    running = [RunningSpot("s1", frozenset({"A"}), ("A",), 0)]
    [ev] = plan(verdicts, running, [], now=10, can_checkpoint=False, park_seconds=300)
    assert ev == Evict("s1", "A", "GPU is RECLAIMING")


def test_parked_spot_gets_the_free_gpu_first():
    verdicts = {
        "A": v("A", GpuState.RECLAIMING, must=True),
        "B": v("B", GpuState.LENDABLE),
    }
    running = [RunningSpot("s2", frozenset({"A"}), ("A", "B"), 5)]
    parked = [ParkedSpot("s1", "C", ("A", "B", "C"), 1)]
    verdicts["C"] = v("C", GpuState.BUSY)
    actions = plan(verdicts, running, parked, now=10, can_checkpoint=True, park_seconds=300)
    assert Restore("s1", "C", "B") in actions
    assert any(isinstance(a, Park) and a.container_id == "s2" for a in actions)


def test_second_spot_on_the_same_gpu_leaves_and_busy_is_skipped():
    verdicts = {"A": v("A", GpuState.LENT), "B": v("B", GpuState.LENDABLE)}
    running = [
        RunningSpot("old", frozenset({"A"}), ("A", "B"), 0),
        RunningSpot("new", frozenset({"A"}), ("A", "B"), 5),
    ]
    [mv] = plan(verdicts, running, [], now=10, can_checkpoint=True, park_seconds=300)
    assert mv == Move("new", "A", "B", "another spot session is on this GPU")
    assert plan(verdicts, running, [], now=10, can_checkpoint=True, park_seconds=300,
                busy=frozenset({"new"})) == []


def test_multi_gpu_spot_is_evicted():
    verdicts = {"A": v("A", GpuState.LENT), "B": v("B", GpuState.LENT)}
    running = [RunningSpot("s1", frozenset({"A", "B"}), ("A", "B"), 0)]
    [ev] = plan(verdicts, running, [], now=1, can_checkpoint=True, park_seconds=300)
    assert isinstance(ev, Evict)


def test_device_map_is_a_swap_over_every_visible_gpu():
    assert device_map("A", "C", ["A", "B", "C"]) == "A=C,B=B,C=A"
    assert driver_major("580.178.04") == 580 and driver_major("x") == 0


# ---- docker / observer ----

def test_split_sessions_and_spot_classification():
    spot = {"Id": "s" * 64, "State": {"Pid": 9},
            "Config": {"Env": ["LABGPU_SPOT=1", "LABGPU_SPOT_UUIDS=GPU-a,GPU-b"]}}
    owner = {"Id": "o" * 64, "State": {"Pid": 8}, "Config": {"Env": ["LABGPU_DEVICE_UUIDS=GPU-a"]}}
    owners, spots = split_sessions([spot, owner])
    assert [o.id for o in owners] == ["o" * 64]
    assert spots == [SpotContainer("s" * 64, 9, ("GPU-a", "GPU-b"))]

    gpus = [GpuInfo(0, "GPU-a", P6K, 96 * GiB, "0:1"), GpuInfo(1, "GPU-c", P6K, 96 * GiB, "0:2")]
    snaps = {
        0: GpuSnapshot(gpus[0], 30 * GiB, 0, (GpuProcess(1, 4 * GiB, 0), GpuProcess(2, 20 * GiB, 90))),
        1: GpuSnapshot(gpus[1], 5 * GiB, 0, (GpuProcess(3, 5 * GiB, 0),)),
    }
    pid_map = {1: "o" * 64, 2: "s" * 64, 3: "s" * 64}  # pid 3: spot on a GPU it was not given
    a, c = observe(gpus, snaps.__getitem__, owners, lambda pids: pid_map, {}, spots=spots)
    assert [p.kind for p in a.processes] == [ProcKind.OWNER, ProcKind.SPOT]
    assert a.spot_containers == {"s" * 64} and a.owner_util == 0
    assert a.free_memory_without_spot == 96 * GiB - 30 * GiB + 20 * GiB
    assert [p.kind for p in c.processes] == [ProcKind.UNKNOWN]


# ---- status / plugin capacity ----

def test_spot_capacity_counts_lendable_and_lent():
    text = json.dumps({"updated_at": 100, "gpus": [
        {"uuid": "A", "state": "LENDABLE"}, {"uuid": "B", "state": "LENT"},
        {"uuid": "C", "state": "BUSY"}, {"uuid": "X", "state": "LENDABLE"},
    ]})
    status = parse_status(text, 110)
    assert spot_capacity(["A", "B", "C", "D"], status) == 2  # X is another model's GPU
    assert spot_capacity(["A"], None) == 0  # unknown status: no room
    off = json.dumps({"updated_at": 100, "spot_enabled": False, "gpus": [{"uuid": "A", "state": "LENDABLE"}]})
    assert spot_capacity(["A"], parse_status(off, 110)) == 0


# ---- controller with fakes ----

class FakeGpus:
    def __init__(self, gpus, procs):
        self.gpus, self.procs = gpus, procs  # procs: index -> list[(pid, mem, util, cid)]

    def list_gpus(self):
        return self.gpus

    def snapshot(self, index):
        ps = self.procs.get(index, [])
        return GpuSnapshot(self.gpus[index], sum(p[1] for p in ps), 0,
                           tuple(GpuProcess(p[0], p[1], p[2]) for p in ps))

    def container_of_pids(self, pids):
        cids = {p[0]: p[3] for ps in self.procs.values() for p in ps}
        return {pid: cids.get(pid) for pid in pids}

    def driver_version(self):
        return "580.178.04"


class FakeSessions:
    def __init__(self, owners, spots):
        self.owners, self.spots = owners, spots

    def sessions(self):
        return self.owners, self.spots


class FakeCkpt:
    def __init__(self):
        self.calls = []

    def available(self, driver):
        return None

    def move(self, cid, pids, src, dst, visible):
        self.calls.append(("move", tuple(pids), src, dst))

    def park(self, cid, pids):
        self.calls.append(("park", tuple(pids)))

    def restore(self, cid, pids, src, dst, visible):
        self.calls.append(("restore", tuple(pids), src, dst))


class Inline:
    def submit(self, fn):
        fn()

    def shutdown(self, wait=True):
        pass


def make_controller(tmp_path: Path, procs):
    gpus = [GpuInfo(i, f"GPU-{i}", P6K, 96 * GiB, f"0:{i}") for i in range(3)]
    owner = OwnerContainer("o" * 64, 0, ("GPU-0",))
    spot = SpotContainer("s" * 64, 0, ("GPU-0", "GPU-1", "GPU-2"))
    cfg = Config(
        controller=ControllerConfig(state_dir=tmp_path),
        idle=IdleConfig(idle_minutes=1, owner_cpu_threshold=0),
        spot=SpotConfig(park_seconds=60),
    )
    fake = FakeGpus(gpus, procs)
    ck = FakeCkpt()
    c = Controller(cfg, fake, FakeSessions([owner], [spot]), checkpointer=ck, workers=Inline())
    c.start()
    return c, fake, ck


def test_controller_moves_spot_when_owner_returns(tmp_path):
    OWNER, SPOT = "o" * 64, "s" * 64
    procs = {0: [(10, 4 * GiB, 0, OWNER)]}
    c, fake, ck = make_controller(tmp_path, procs)
    for t in (0, 30, 61, 70):  # GPU-0 owner idle; GPU-1/2 unclaimed
        c.tick(t)
    assert c.last_verdicts["GPU-0"].state is GpuState.LENDABLE
    procs[0].append((20, 30 * GiB, 90, SPOT))  # spot lands on GPU-0
    c.tick(80)
    assert c.last_verdicts["GPU-0"].state is GpuState.LENT and ck.calls == []
    procs[0][0] = (10, 4 * GiB, 80, OWNER)  # owner is back
    c.tick(90)
    assert ck.calls == [("move", (20,), "GPU-0", "GPU-1")] or ck.calls == [("move", (20,), "GPU-0", "GPU-2")]
    c.write_status(tmp_path / "status.json", 90)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["can_move"] is True and status["gpus"][0]["lent_job"] == SPOT[:12]


def test_controller_parks_then_restores_and_persists(tmp_path):
    OWNER, SPOT = "o" * 64, "s" * 64
    procs = {0: [(10, 4 * GiB, 0, OWNER)], 1: [(11, GiB, 50, None)], 2: [(12, GiB, 50, None)]}
    c, fake, ck = make_controller(tmp_path, procs)  # GPU-1/2 busy with unknown processes
    for t in (0, 61, 70):
        c.tick(t)
    procs[0].append((20, 30 * GiB, 90, SPOT))
    procs[0][0] = (10, 4 * GiB, 80, OWNER)
    c.tick(80)
    assert ck.calls == [("park", (20,))]
    procs[0].pop()  # checkpointed: no longer on the GPU
    c.tick(85)  # records the park
    assert "s" * 64 in c.parked and (tmp_path / "parked.json").exists()
    procs[2] = []  # GPU-2 frees up
    for t in (90, 160):
        c.tick(t)
    assert ("restore", (20,), "GPU-0", "GPU-2") in ck.calls
    c.tick(165)
    assert c.parked == {}


def test_background_monitor_runs_once_per_process(tmp_path):
    from labgpu.spot.daemon import BackgroundMonitor

    procs = {0: []}
    made = []

    def make():
        c, _, _ = make_controller(tmp_path, procs)
        c.cfg = Config(controller=ControllerConfig(state_dir=tmp_path, poll_interval=0.1))
        made.append(c)
        return c

    first = BackgroundMonitor.ensure_started(make)
    try:
        assert first is not None
        assert BackgroundMonitor.ensure_started(make) is None  # a second spot plugin reuses it
        assert len(made) == 1
        import time
        for _ in range(50):
            if (tmp_path / "status.json").exists():
                break
            time.sleep(0.1)
        assert json.loads((tmp_path / "status.json").read_text())["gpus"]
    finally:
        first.stop()
    assert BackgroundMonitor._running is None


def test_pick_spot_gpu_takes_the_roomiest_free_lendable_gpu():
    text = json.dumps({"updated_at": 100, "gpus": [
        {"uuid": "A", "state": "LENDABLE", "lendable_memory": 10},
        {"uuid": "B", "state": "LENDABLE", "lendable_memory": 50},
        {"uuid": "C", "state": "LENT", "lent_job": "x", "lendable_memory": 0},
        {"uuid": "D", "state": "BUSY"},
    ]})
    status = parse_status(text, 110)
    assert pick_spot_gpu(["A", "B", "C", "D"], status) == ("B", 50)
    assert pick_spot_gpu(["A", "B"], status, taken={"B"}) == ("A", 10)
    assert pick_spot_gpu(["C", "D"], status) is None
    assert pick_spot_gpu(["A"], None) is None


def test_state_dir_follows_the_agent_var_base_path():
    from types import SimpleNamespace

    from labgpu.paths import agent_state_dir

    assert agent_state_dir({"agent": {"var-base-path": "/srv/bai/var"}}) == Path("/srv/bai/var/labgpu")
    obj = SimpleNamespace(agent=SimpleNamespace(var_base_path=Path("/x")))
    assert agent_state_dir(obj) == Path("/x/labgpu")
    assert agent_state_dir(None) == Path("./var/lib/backend.ai/labgpu")


def test_monitor_config_from_etcd_strings():
    cfg = Config.from_dict({
        "idle": {"idle_minutes": "10", "ignored_processes": "Xorg, gnome-shell"},
        "spot": {"enabled": "false", "park_seconds": "120", "cuda_checkpoint": "/opt/cc"},
        "reclaim": {"mem_reserve_mib": "4096"},
    })
    assert cfg.idle.idle_seconds == 600
    assert cfg.idle.ignored_processes == ("Xorg", "gnome-shell")
    assert cfg.spot.enabled is False and cfg.spot.park_seconds == 120.0
    assert cfg.spot.cuda_checkpoint == Path("/opt/cc") and cfg.reclaim.mem_reserve_mib == 4096


def test_cuda_checkpoint_runs_inside_the_container(tmp_path):
    from labgpu.spot.ckpt import CONTAINER_CUDA_CHECKPOINT, CudaCheckpoint, container_pid

    (tmp_path / "4242").mkdir()
    (tmp_path / "4242" / "status").write_text("Name: python\nNSpid: 4242 17\n")
    assert container_pid(4242, tmp_path) == 17
    calls = []
    ck = CudaCheckpoint(Path("/x"), run=lambda cid, *argv: calls.append((cid, argv)) or "",
                        pid_in_container=lambda p: p - 4225)
    ck.move("c1", [4242], "GPU-a", "GPU-b", ["GPU-a", "GPU-b"])
    actions = [argv[argv.index("--action") + 1] for _, argv in calls]
    assert actions == ["lock", "checkpoint", "restore", "unlock"]
    cid, argv = calls[2]
    assert cid == "c1" and CONTAINER_CUDA_CHECKPOINT in argv and argv[:5] == (
        "env", "-u", "LD_PRELOAD", "-u", "CUDA_VISIBLE_DEVICES")
    assert argv[argv.index("--pid") + 1] == "17"
    assert argv[argv.index("--device-map") + 1] == "GPU-a=GPU-b,GPU-b=GPU-a"


def test_cli_status_accepts_state_dir_after_the_command(tmp_path, capsys):
    from labgpu.spot.cli import main

    (tmp_path / "status.json").write_text(json.dumps({"updated_at": 0, "gpus": []}))
    assert main(["status", "--state-dir", str(tmp_path)]) == 0
    assert main(["--state-dir", str(tmp_path), "status"]) == 0
