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
from labgpu.spot.placement import Move, Oom, Park, ParkedSpot, Restore, RunningSpot, plan
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
    assert plan(verdicts, running, [], now=10, can_checkpoint=True, oom_grace=10) == []


def test_owner_back_moves_spot_to_same_model_gpu():
    verdicts = {
        "A": v("A", GpuState.RECLAIMING, must=True, reasons=("owner SM util 90% > 5%",)),
        "B": v("B", GpuState.LENDABLE, model=A6K),  # other model: never a target
        "C": v("C", GpuState.LENDABLE, mem=10 * GiB),
        "D": v("D", GpuState.LENDABLE, mem=50 * GiB),
    }
    running = [RunningSpot("s1", frozenset({"A"}), ("A", "B", "C", "D"), 0)]
    [action] = plan(verdicts, running, [], now=10, can_checkpoint=True, oom_grace=10)
    assert action == Move("s1", "A", "D", "owner SM util 90% > 5%")


def test_no_target_raises_oom_then_parks_if_still_on_the_gpu():
    verdicts = {"A": v("A", GpuState.RECLAIMING, must=True), "B": v("B", GpuState.BUSY)}
    running = [RunningSpot("s1", frozenset({"A"}), ("A", "B"), 0)]
    [oom] = plan(verdicts, running, [], now=10, can_checkpoint=True, oom_grace=10)
    assert oom == Oom("s1", "A", "GPU is RECLAIMING")
    assert plan(verdicts, running, [], now=15, can_checkpoint=True, oom_grace=10, oomed={"s1": 10}) == []
    [park] = plan(verdicts, running, [], now=21, can_checkpoint=True, oom_grace=10, oomed={"s1": 10})
    assert isinstance(park, Park) and park.src == "A"
    parked = [ParkedSpot("s1", "A", ("A", "B"), 21)]
    assert plan(verdicts, [], parked, now=500, can_checkpoint=True, oom_grace=10) == []  # waits, never killed


def test_program_without_the_oom_handler_is_parked_at_once():
    verdicts = {"A": v("A", GpuState.RECLAIMING, must=True), "B": v("B", GpuState.BUSY)}
    running = [RunningSpot("s1", frozenset({"A"}), ("A", "B"), 0, oom_ready=False)]
    [park] = plan(verdicts, running, [], now=10, can_checkpoint=True, oom_grace=10)
    assert isinstance(park, Park)


def test_parked_spot_returns_to_its_own_gpu_when_it_frees_up():
    verdicts = {"A": v("A", GpuState.LENDABLE), "B": v("B", GpuState.BUSY)}
    parked = [ParkedSpot("s1", "A", ("A", "B"), 10)]
    assert plan(verdicts, [], parked, now=20, can_checkpoint=True, oom_grace=10) == [
        Restore("s1", "A", "A")
    ]


def test_without_cuda_checkpoint_only_the_oom_error_is_possible():
    verdicts = {"A": v("A", GpuState.RECLAIMING, must=True)}
    running = [RunningSpot("s1", frozenset({"A"}), ("A",), 0, oom_ready=False)]
    [oom] = plan(verdicts, running, [], now=10, can_checkpoint=False, oom_grace=10)
    assert oom == Oom("s1", "A", "GPU is RECLAIMING")


def test_parked_spot_gets_the_free_gpu_first():
    verdicts = {
        "A": v("A", GpuState.RECLAIMING, must=True),
        "B": v("B", GpuState.LENDABLE),
    }
    running = [RunningSpot("s2", frozenset({"A"}), ("A", "B"), 5)]
    parked = [ParkedSpot("s1", "C", ("A", "B", "C"), 1)]
    verdicts["C"] = v("C", GpuState.BUSY)
    actions = plan(verdicts, running, parked, now=10, can_checkpoint=True, oom_grace=10)
    assert Restore("s1", "C", "B") in actions
    assert any(isinstance(a, Oom) and a.container_id == "s2" for a in actions)


def test_second_spot_on_the_same_gpu_leaves_and_busy_is_skipped():
    verdicts = {"A": v("A", GpuState.LENT), "B": v("B", GpuState.LENDABLE)}
    running = [
        RunningSpot("old", frozenset({"A"}), ("A", "B"), 0),
        RunningSpot("new", frozenset({"A"}), ("A", "B"), 5),
    ]
    [mv] = plan(verdicts, running, [], now=10, can_checkpoint=True, oom_grace=10)
    assert mv == Move("new", "A", "B", "another spot session is on this GPU")
    assert plan(verdicts, running, [], now=10, can_checkpoint=True, oom_grace=10,
                busy=frozenset({"new"})) == []


def test_multi_gpu_spot_gets_the_oom_error():
    verdicts = {"A": v("A", GpuState.LENT), "B": v("B", GpuState.LENT)}
    running = [RunningSpot("s1", frozenset({"A", "B"}), ("A", "B"), 0)]
    [oom] = plan(verdicts, running, [], now=1, can_checkpoint=True, oom_grace=10)
    assert isinstance(oom, Oom)


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

    def place(self, cid, pid, order, size):
        self.calls.append(("place", pid, tuple(order), size))

    def raise_oom(self, cid, pids, order, reason):
        self.calls.append(("oom", tuple(pids), tuple(order)))


class Inline:
    def submit(self, fn):
        fn()

    def shutdown(self, wait=True):
        pass


def make_controller(tmp_path: Path, procs, handler=True):
    gpus = [GpuInfo(i, f"GPU-{i}", P6K, 96 * GiB, f"0:{i}") for i in range(3)]
    owner = OwnerContainer("o" * 64, 0, ("GPU-0",))
    spot = SpotContainer("s" * 64, 0, ("GPU-0", "GPU-1", "GPU-2"))
    cfg = Config(
        controller=ControllerConfig(state_dir=tmp_path),
        idle=IdleConfig(idle_minutes=1, owner_cpu_threshold=0),
        spot=SpotConfig(oom_grace_seconds=10),
    )
    fake = FakeGpus(gpus, procs)
    ck = FakeCkpt()
    c = Controller(cfg, fake, FakeSessions([owner], [spot]), checkpointer=ck, workers=Inline(),
                   handler_check=lambda pid: handler)
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
    assert ck.calls[0] in (("move", (20,), "GPU-0", "GPU-1"), ("move", (20,), "GPU-0", "GPU-2"))
    dst = ck.calls[0][3]
    # The new GPU becomes cuda:0 with its lendable memory; GPU-0 (back to its owner) gets 1 MiB.
    assert ck.calls[1][:2] == ("place", 20) and ck.calls[1][2][0] == dst and ck.calls[1][2][1] == "GPU-0"
    assert ck.calls[1][3] > 80 * GiB
    c.write_status(tmp_path / "status.json", 90)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["can_move"] is True and status["gpus"][0]["lent_job"] == SPOT[:12]


def test_controller_parks_then_restores_and_persists(tmp_path):
    OWNER, SPOT = "o" * 64, "s" * 64
    procs = {0: [(10, 4 * GiB, 0, OWNER)], 1: [(11, GiB, 50, None)], 2: [(12, GiB, 50, None)]}
    c, fake, ck = make_controller(tmp_path, procs, handler=False)  # GPU-1/2 busy with strangers
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


def test_controller_raises_oom_then_parks_a_program_that_keeps_the_gpu(tmp_path):
    OWNER, SPOT = "o" * 64, "s" * 64
    procs = {0: [(10, 4 * GiB, 0, OWNER)], 1: [(11, GiB, 50, None)], 2: [(12, GiB, 50, None)]}
    c, fake, ck = make_controller(tmp_path, procs, handler=True)
    for t in (0, 61, 70):
        c.tick(t)
    procs[0].append((20, 30 * GiB, 90, SPOT))
    procs[0][0] = (10, 4 * GiB, 80, OWNER)
    c.tick(80)
    oom = ("oom", (20,), ("GPU-0", "GPU-1", "GPU-2"))
    assert ck.calls == [oom]  # torch.OutOfMemoryError right away, never killed
    c.tick(85)
    assert ck.calls == [oom]  # gives the program oom_grace to react
    c.tick(91)
    assert ck.calls[-1] == ("park", (20,))  # still holding the owner's GPU: off the GPU


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
        "spot": {"enabled": "false", "oom_grace_seconds": "5", "cuda_checkpoint": "/opt/cc"},
        "reclaim": {"mem_reserve_mib": "4096"},
    })
    assert cfg.idle.idle_seconds == 600
    assert cfg.idle.ignored_processes == ("Xorg", "gnome-shell")
    assert cfg.spot.enabled is False and cfg.spot.oom_grace_seconds == 5.0
    assert cfg.spot.cuda_checkpoint == Path("/opt/cc") and cfg.reclaim.mem_reserve_mib == 4096


def test_cuda_checkpoint_runs_inside_the_container_as_the_owner(tmp_path):
    from labgpu.spot.ckpt import CONTAINER_CUDA_CHECKPOINT, CudaCheckpoint, ProcIdentity, identify

    (tmp_path / "4242").mkdir()
    (tmp_path / "4242" / "status").write_text(
        "Name: python\nUid: 1100 1100 1100 1100\nGid: 1200 1200 1200 1200\nNSpid: 4242 17\n")
    assert identify(4242, tmp_path) == ProcIdentity(17, "1100:1200")
    calls = []
    ck = CudaCheckpoint(Path("/x"), run=lambda cid, *argv, user: calls.append((cid, user, argv)) or "",
                        identify=lambda p: ProcIdentity(17, "1100:1200"))
    ck.move("c1", [4242], "GPU-a", "GPU-b", ["GPU-a", "GPU-b"])
    actions = [argv[argv.index("--action") + 1] for _, _, argv in calls]
    assert actions == ["lock", "checkpoint", "restore", "unlock"]
    cid, user, argv = calls[2]
    assert cid == "c1" and user == "1100:1200" and CONTAINER_CUDA_CHECKPOINT in argv
    assert argv[:5] == ("env", "-u", "LD_PRELOAD", "-u", "CUDA_VISIBLE_DEVICES")
    assert argv[argv.index("--pid") + 1] == "17"
    assert argv[argv.index("--device-map") + 1] == "GPU-a=GPU-b,GPU-b=GPU-a"


def test_cli_status_accepts_state_dir_after_the_command(tmp_path, capsys):
    from labgpu.spot.cli import main

    (tmp_path / "status.json").write_text(json.dumps({"updated_at": 0, "gpus": []}))
    assert main(["status", "--state-dir", str(tmp_path)]) == 0
    assert main(["--state-dir", str(tmp_path), "status"]) == 0


def test_oom_lowers_limits_and_signals_as_owner(monkeypatch):
    from labgpu.spot import ckpt

    monkeypatch.setattr(ckpt, "_exists", lambda pid: True)
    calls = []
    ckpt.raise_oom("c1", [4242], order=["GPU-b", "GPU-a"], reason="no free GPU",
                   run=lambda cid, *argv, user: calls.append((user, argv)) or "",
                   identify=lambda p: ckpt.ProcIdentity(17, "1100:1200"))
    [(user, argv)] = calls
    assert user == "1100:1200" and argv[:2] == ("sh", "-c")
    marker, code, limits, pids, sig, env_text, env_file = argv[4:11]
    assert marker == "no free GPU" and "set_current_device_memory_limit" in code
    assert eval(limits) == [(0, 1 << 20), (1, 1 << 20)] and eval(pids) == [17]
    assert int(sig) == ckpt.OOM_SIGNAL and "KILL" not in code
    assert "CUDA_DEVICE_MEMORY_LIMIT_0=1m" in env_text and env_file == ckpt.SPOT_ENV_FILE


def test_place_blocks_every_gpu_but_the_current_one():
    from labgpu.spot import ckpt

    calls = []
    ckpt.place("c1", 4242, order=["GPU-b", "GPU-a", "GPU-c"], size=10 << 30,
               run=lambda cid, *argv, user: calls.append(argv) or "",
               identify=lambda p: ckpt.ProcIdentity(17, "1100:1200"))
    [argv] = calls
    limits, env_text = eval(argv[6]), argv[9]
    assert limits == [(0, 10 << 30), (1, 1 << 20), (2, 1 << 20)]
    assert env_text.splitlines() == [
        "CUDA_VISIBLE_DEVICES=GPU-b", "LABGPU_SPOT_GPU=GPU-b",
        "CUDA_DEVICE_MEMORY_LIMIT_0=10240m", "CUDA_DEVICE_MEMORY_LIMIT_1=1m", "CUDA_DEVICE_MEMORY_LIMIT_2=1m",
    ]


def test_mixed_processes_each_get_their_own_treatment(tmp_path):
    OWNER, SPOT = "o" * 64, "s" * 64
    procs = {0: [(10, 4 * GiB, 0, OWNER)], 1: [(11, GiB, 50, None)], 2: [(12, GiB, 50, None)]}
    c, fake, ck = make_controller(tmp_path, procs)
    c.handler_check = lambda pid: pid == 20  # 20 installed the handler, 21 did not
    for t in (0, 61, 70):
        c.tick(t)
    procs[0] += [(20, 20 * GiB, 90, SPOT), (21, 10 * GiB, 90, SPOT)]
    procs[0][0] = (10, 4 * GiB, 80, OWNER)
    c.tick(80)
    # 21 is taken off the GPU at once; 20 gets the error and its grace period.
    assert ck.calls == [("park", (21,)), ("oom", (20,), ("GPU-0", "GPU-1", "GPU-2"))]
    procs[0].remove((21, 10 * GiB, 90, SPOT))
    c.tick(85)
    assert c.parked[SPOT].pids == (21,)
    c.tick(91)  # 20 still holds the GPU after the grace: parked too, same record
    assert ck.calls[-1] == ("park", (20,))
    procs[0].remove((20, 20 * GiB, 90, SPOT))
    c.tick(92)
    assert set(c.parked[SPOT].pids) == {20, 21}


def test_has_handler_reads_sigcgt(tmp_path):
    from labgpu.spot.ckpt import has_handler

    (tmp_path / "7").mkdir()
    (tmp_path / "7" / "status").write_text("SigCgt: %016x" % (1 << 43))
    assert has_handler(7, 44, tmp_path) and not has_handler(7, 10, tmp_path)
    assert not has_handler(8, 44, tmp_path)


def test_parked_spot_without_room_stays_parked(tmp_path):
    OWNER, SPOT = "o" * 64, "s" * 64
    procs = {0: [(10, 4 * GiB, 0, OWNER)], 1: [(11, GiB, 50, None)], 2: [(12, GiB, 50, None)]}
    c, fake, ck = make_controller(tmp_path, procs, handler=False)
    for t in (0, 61, 70):
        c.tick(t)
    procs[0].append((20, 30 * GiB, 90, SPOT))
    procs[0][0] = (10, 4 * GiB, 80, OWNER)
    c.tick(80)
    procs[0].pop()
    for t in (85, 400, 900):
        c.tick(t)
    assert SPOT in c.parked and ck.calls == [("park", (20,))]  # waits off the GPU, never killed


def test_sitecustomize_follows_the_monitor_env_file(tmp_path, monkeypatch):
    import importlib.util

    env = tmp_path / "env"
    env.write_text("CUDA_VISIBLE_DEVICES=GPU-b,GPU-a\nCUDA_DEVICE_MEMORY_LIMIT_0=10m\nPATH=/evil\n")
    spec = importlib.util.spec_from_file_location(
        "labgpu_site", Path(__file__).parents[1] / "src/labgpu/spot/inject/sitecustomize.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    monkeypatch.setenv("LABGPU_SPOT_OOM_SIGNAL", "0")
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_ENV_FILE", str(env))
    path_before = __import__("os").environ["PATH"]
    module._follow_moves()
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-b,GPU-a"
    assert os.environ["CUDA_DEVICE_MEMORY_LIMIT_0"] == "10m" and os.environ["PATH"] == path_before


def test_failed_park_still_sends_the_oom_error(tmp_path):
    from labgpu.spot.ckpt import CheckpointError

    OWNER, SPOT = "o" * 64, "s" * 64
    procs = {0: [(10, 4 * GiB, 0, OWNER)], 1: [(11, GiB, 50, None)], 2: [(12, GiB, 50, None)]}
    c, fake, ck = make_controller(tmp_path, procs)
    c.handler_check = lambda pid: pid == 20

    def broken_park(cid, pids):
        ck.calls.append(("park-failed", tuple(pids)))
        raise CheckpointError("process exited")

    ck.park = broken_park
    for t in (0, 61, 70):
        c.tick(t)
    procs[0] += [(20, 20 * GiB, 90, SPOT), (21, 10 * GiB, 90, SPOT)]
    procs[0][0] = (10, 4 * GiB, 80, OWNER)
    c.tick(80)
    assert ck.calls == [("park-failed", (21,)), ("oom", (20,), ("GPU-0", "GPU-1", "GPU-2"))]
    assert c.oomed[SPOT] == 80 and SPOT not in c.parked
