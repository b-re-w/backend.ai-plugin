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
from labgpu.spot.placement import Move, Oom, Park, ParkedSpot, Resize, Restore, RunningSpot, plan
from labgpu.spotstatus import GpuLending, parse_status, pick_spot_gpu, roomiest_spot_gpu, spot_capacity, unreported

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
    assert mv == Move("new", "A", "B", "spot sessions on this GPU exceed one GPU in total")
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
    assert spots == [SpotContainer("s" * 64, 9, ("GPU-a", "GPU-b"))]  # no LABGPU_SPOT_GPU: gpu ""

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

    def gate(self, cid, allow, deny):
        self.calls.append(("gate", tuple(allow), tuple(deny)))

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
    spot = SpotContainer("s" * 64, 0, ("GPU-0", "GPU-1", "GPU-2"), gpu="GPU-0")
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


def test_pick_spot_gpu_best_fit_by_share():
    text = json.dumps({"updated_at": 100, "gpus": [
        {"uuid": "A", "state": "LENDABLE", "spot_capacity": 10},
        {"uuid": "B", "state": "LENDABLE", "spot_capacity": 50},
        {"uuid": "C", "state": "LENT", "lent_job": "x", "spot_share": 0.5, "spot_capacity": 40},
        {"uuid": "D", "state": "BUSY"},
        {"uuid": "E", "state": "LENT", "lent_job": "y"},  # older monitor: fully taken
    ]})
    status = parse_status(text, 110)
    # A whole GPU: the empty one with more capacity.
    assert pick_spot_gpu(["A", "B", "C", "D", "E"], status, 1.0) == ("B", 50)
    # Half a GPU: C is already half full, so it is filled first and A, B stay whole.
    assert pick_spot_gpu(["A", "B", "C", "D", "E"], status, 0.5) == ("C", 40)
    # Shares just handed out count as taken.
    assert pick_spot_gpu(["A", "B", "C"], status, 0.5, taken={"C": 0.5}) == ("B", 50)
    assert pick_spot_gpu(["C", "D", "E"], status, 0.75) is None
    assert pick_spot_gpu(["A"], None) is None
    # Fragmented: nothing fits 0.75, the roomiest GPU is offered with what it has left.
    assert roomiest_spot_gpu(["C", "D", "E"], status) == ("C", 40, 0.5)


def test_several_fractional_spots_share_one_gpu():
    verdicts = {"A": v("A", GpuState.LENT), "B": v("B", GpuState.LENDABLE)}
    running = [
        RunningSpot("a", frozenset({"A"}), ("A", "B"), 0, share=0.5),
        RunningSpot("b", frozenset({"A"}), ("A", "B"), 1, share=0.25),
        RunningSpot("c", frozenset({"A"}), ("A", "B"), 2, share=0.25),
    ]
    assert plan(verdicts, running, [], now=10, can_checkpoint=True, oom_grace=10) == []
    # A fourth one would push A over one GPU: the latest arrival leaves.
    running.append(RunningSpot("d", frozenset({"A"}), ("A", "B"), 3, share=0.25))
    [mv] = plan(verdicts, running, [], now=10, can_checkpoint=True, oom_grace=10)
    assert mv == Move("d", "A", "B", "spot sessions on this GPU exceed one GPU in total")


def test_reclaim_moves_each_fractional_spot_where_its_share_fits():
    verdicts = {
        "A": v("A", GpuState.RECLAIMING, must=True, reasons=("owner back",)),
        "B": v("B", GpuState.LENT),  # 0.5 used by another spot already
        "C": v("C", GpuState.LENDABLE),
    }
    running = [
        RunningSpot("x", frozenset({"B"}), ("A", "B", "C"), 0, share=0.5),
        RunningSpot("a1", frozenset({"A"}), ("A", "B", "C"), 1, share=0.5),
        RunningSpot("a2", frozenset({"A"}), ("A", "B", "C"), 2, share=0.5),
    ]
    actions = plan(verdicts, running, [], now=10, can_checkpoint=True, oom_grace=10)
    # a1 fills B's free half (best fit); B is then full, so a2 gets the empty C; x stays on B.
    assert Move("a1", "A", "B", "owner back") in actions
    assert Move("a2", "A", "C", "owner back") in actions
    assert len(actions) == 2


def test_parked_fractional_spot_restores_into_a_partly_used_gpu():
    verdicts = {"A": v("A", GpuState.BUSY), "B": v("B", GpuState.LENT)}
    running = [RunningSpot("x", frozenset({"B"}), ("A", "B"), 0, share=0.5)]
    parked = [ParkedSpot("p", "A", ("A", "B"), 5, share=0.5)]
    assert plan(verdicts, running, parked, now=10, can_checkpoint=True, oom_grace=10) == [Restore("p", "A", "B")]
    too_big = [ParkedSpot("p", "A", ("A", "B"), 5, share=0.75)]
    assert plan(verdicts, running, too_big, now=10, can_checkpoint=True, oom_grace=10) == []


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
    assert ck.calls == [("oom", (20,), ("GPU-0", "GPU-1", "GPU-2")), ("park", (21,))]
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
    assert ck.calls == [("oom", (20,), ("GPU-0", "GPU-1", "GPU-2")), ("park-failed", (21,))]
    assert c.oomed[SPOT] == 80 and SPOT not in c.parked


def test_process_on_a_gpu_it_was_not_given_is_held_off_it(tmp_path):
    OWNER, SPOT = "o" * 64, "s" * 64
    procs = {0: [(10, 4 * GiB, 0, OWNER)]}
    c, fake, ck = make_controller(tmp_path, procs)  # the session was given GPU-0
    for t in (0, 61, 70):
        c.tick(t)
    procs[0].append((20, 20 * GiB, 90, SPOT))  # its training on GPU-0: fine
    procs[1] = [(30, 10 * GiB, 90, SPOT)]  # CUDA_VISIBLE_DEVICES changed: 10 GiB on GPU-1
    c.tick(80)
    assert ck.calls == [("park", (30,))]  # right away, no grace, the rest untouched
    procs[1] = []
    c.tick(85)
    assert c.held[SPOT].pids == (30,) and SPOT not in c.parked
    for t in (90, 400):
        c.tick(t)
    assert not any(call[0] == "restore" for call in ck.calls)  # never restored automatically


def test_moves_update_the_assigned_gpu(tmp_path):
    OWNER, SPOT = "o" * 64, "s" * 64
    procs = {0: [(10, 4 * GiB, 0, OWNER)]}
    c, fake, ck = make_controller(tmp_path, procs)
    for t in (0, 30, 61, 70):
        c.tick(t)
    procs[0].append((20, 30 * GiB, 90, SPOT))
    c.tick(80)
    procs[0][0] = (10, 4 * GiB, 80, OWNER)
    c.tick(90)
    dst = ck.calls[0][3]
    procs[0].pop()
    procs[int(dst[-1])] = [(20, 30 * GiB, 90, SPOT)]
    c.tick(95)
    assert c.assigned[SPOT] == dst and not any(call[0] == "park" for call in ck.calls)


@__import__("pytest").mark.skipif(__import__("os").name == "nt", reason="':' in a directory name")
def test_device_minor_from_proc(tmp_path):
    from labgpu.spot.ckpt import device_minor

    d = tmp_path / "driver" / "nvidia" / "gpus" / "0000:3b:00.0"
    d.mkdir(parents=True)
    (d / "information").write_text("Model: NVIDIA RTX PRO 6000" + chr(10) + "Device Minor: 2" + chr(10))
    assert device_minor("00000000:3B:00.0", tmp_path) == 2
    assert device_minor("00000000:AA:00.0", tmp_path) is None


def test_device_files_are_gated_and_follow_moves(tmp_path):
    OWNER, SPOT = "o" * 64, "s" * 64
    procs = {0: [(10, 4 * GiB, 0, OWNER)]}
    c, fake, ck = make_controller(tmp_path, procs)
    c.minors = {"GPU-0": 0, "GPU-1": 1, "GPU-2": 2}
    c.tick(0)
    assert ck.calls == [("gate", (0,), (1, 2))]  # from the first sight: only its GPU opens
    c.tick(1)
    assert len(ck.calls) == 1  # once
    for t in (30, 61, 70):
        c.tick(t)
    procs[0].append((20, 30 * GiB, 90, SPOT))
    c.tick(80)
    procs[0][0] = (10, 4 * GiB, 80, OWNER)
    ck.calls.clear()
    c.tick(90)
    move = ck.calls[1]
    dst = int(move[3][-1])
    # Every device open while cuda-checkpoint works, then only the new GPU.
    assert ck.calls[0] == ("gate", (0, 1, 2), ()) and move[0] == "move"
    assert ck.calls[2] == ("gate", (dst,), tuple(m for m in (0, 1, 2) if m != dst))
    assert ck.calls[3][0] == "place"
    spot_proc = next(x for x in procs[0] if x[3] == SPOT)
    procs[0].remove(spot_proc)
    procs[dst] = [spot_proc]  # it now runs on the new GPU
    c.tick(95)
    assert c.gated[SPOT] == f"GPU-{dst}" and not any(x[0] == "gate" for x in ck.calls[4:])


def test_parked_session_gets_no_device_and_failed_restore_closes_the_target(tmp_path):
    from labgpu.spot.ckpt import CheckpointError

    OWNER, SPOT = "o" * 64, "s" * 64
    procs = {0: [(10, 4 * GiB, 0, OWNER)], 1: [(11, GiB, 50, None)], 2: [(12, GiB, 50, None)]}
    c, fake, ck = make_controller(tmp_path, procs, handler=False)
    c.minors = {"GPU-0": 0, "GPU-1": 1, "GPU-2": 2}
    for t in (0, 61, 70):
        c.tick(t)
    procs[0].append((20, 30 * GiB, 90, SPOT))
    procs[0][0] = (10, 4 * GiB, 80, OWNER)
    ck.calls.clear()
    c.tick(80)  # parked, and in the same step every device closed
    assert ck.calls == [("gate", (0, 1, 2), ()), ("park", (20,)), ("gate", (), (0, 1, 2))]
    procs[0].pop()
    ck.calls.clear()
    c.tick(85)
    assert not any(x[0] == "gate" for x in ck.calls)  # nothing reopened

    def broken_restore(cid, pids, src, dst, visible):
        ck.calls.append(("restore-failed", dst))
        raise CheckpointError("rc=1: restore failed")

    ck.restore = broken_restore
    procs[2] = []  # GPU-2 frees up
    ck.calls.clear()
    for t in (90, 160):
        c.tick(t)
    i = ck.calls.index(("restore-failed", "GPU-2"))
    assert ck.calls[i - 1] == ("gate", (0, 1, 2), ())  # all open for cuda-checkpoint
    assert ck.calls[i + 1] == ("gate", (), (0, 1, 2))  # still parked: all closed again


# ---- consolidation (SPEC 2.12 rule 3-1) ----

def test_big_share_on_a_fragmented_gpu_makes_room_instead_of_oom():
    # A and B each hold a 0.5 spot; a whole-GPU spot z landed on A (fragmented placement).
    verdicts = {"A": v("A", GpuState.LENT), "B": v("B", GpuState.LENT)}
    running = [
        RunningSpot("x", frozenset({"A"}), ("A", "B"), 0, share=0.5),
        RunningSpot("y", frozenset({"B"}), ("A", "B"), 1, share=0.5),
        RunningSpot("z", frozenset({"A"}), ("A", "B"), 5, share=1.0),
    ]
    actions = plan(verdicts, running, [], now=10, can_checkpoint=True, oom_grace=10)
    # x leaves A for B's free half; z waits on A (no error) and fits there once x is gone.
    assert actions == [Move("x", "A", "B", "making room for z")]
    after = [
        RunningSpot("x", frozenset({"B"}), ("A", "B"), 11, share=0.5),
        RunningSpot("y", frozenset({"B"}), ("A", "B"), 1, share=0.5),
        RunningSpot("z", frozenset({"A"}), ("A", "B"), 5, share=1.0),
    ]
    assert plan(verdicts, after, [], now=20, can_checkpoint=True, oom_grace=10) == []


def test_owner_back_consolidates_before_raising_oom():
    # Owner returns to A (z, 1.0). B and C each hold a 0.5 spot: no GPU fits z until one moves.
    verdicts = {
        "A": v("A", GpuState.RECLAIMING, must=True, reasons=("owner back",)),
        "B": v("B", GpuState.LENT),
        "C": v("C", GpuState.LENT),
    }
    running = [
        RunningSpot("b", frozenset({"B"}), ("A", "B", "C"), 0, share=0.5),
        RunningSpot("c", frozenset({"C"}), ("A", "B", "C"), 1, share=0.5),
        RunningSpot("z", frozenset({"A"}), ("A", "B", "C"), 2, share=1.0),
    ]
    actions = plan(verdicts, running, [], now=10, can_checkpoint=True, oom_grace=10)
    assert len(actions) == 1 and isinstance(actions[0], Move)
    assert actions[0].reason == "making room for z" and {actions[0].src, actions[0].dst} == {"B", "C"}
    assert not any(isinstance(a, Oom) for a in actions)


def test_no_consolidation_when_nothing_is_waiting():
    # Spread out but everyone fits: nothing moves.
    verdicts = {"A": v("A", GpuState.LENT), "B": v("B", GpuState.LENT)}
    running = [
        RunningSpot("x", frozenset({"A"}), ("A", "B"), 0, share=0.5),
        RunningSpot("y", frozenset({"B"}), ("A", "B"), 1, share=0.5),
    ]
    assert plan(verdicts, running, [], now=10, can_checkpoint=True, oom_grace=10) == []


def test_oom_when_room_cannot_be_made():
    verdicts = {
        "A": v("A", GpuState.RECLAIMING, must=True, reasons=("owner back",)),
        "B": v("B", GpuState.LENT),
    }
    running = [
        RunningSpot("b", frozenset({"B"}), ("A", "B"), 0, share=0.75),
        RunningSpot("z", frozenset({"A"}), ("A", "B"), 1, share=0.5),
    ]
    [oom] = plan(verdicts, running, [], now=10, can_checkpoint=True, oom_grace=10)
    assert oom == Oom("z", "A", "owner back")


def test_short_session_gets_its_full_limit_once_it_fits():
    verdicts = {"A": v("A", GpuState.LENT)}
    running = [RunningSpot("z", frozenset({"A"}), ("A",), 5, share=1.0, short=True)]
    assert plan(verdicts, running, [], now=20, can_checkpoint=True, oom_grace=10) == [Resize("z", "A")]
    running = [RunningSpot("z", frozenset({"A"}), ("A",), 5, share=1.0)]
    assert plan(verdicts, running, [], now=20, can_checkpoint=True, oom_grace=10) == []


def test_idle_spot_holds_its_share_but_never_pushes_a_running_one():
    verdicts = {
        "A": v("A", GpuState.LENT, must=True, reasons=("owner back",)),
        "B": v("B", GpuState.LENDABLE),
        "C": v("C", GpuState.LENDABLE),
    }
    running = [RunningSpot("s", frozenset({"A"}), ("A", "B", "C"), 0, share=0.75)]
    # B is the fuller fit, but an idle 0.5 spot was given it: the move goes to C.
    [mv] = plan(verdicts, running, [], now=10, can_checkpoint=True, oom_grace=10, idle={"B": 0.5})
    assert (mv.container_id, mv.dst) == ("s", "C")
    # An idle spot over the top of a running one on a quiet GPU moves nobody.
    verdicts["A"] = v("A", GpuState.LENT)
    assert plan(verdicts, running, [], now=10, can_checkpoint=True, oom_grace=10, idle={"A": 0.5}) == []


def test_status_counts_a_spot_with_no_gpu_process_on_its_gpu(tmp_path):
    OWNER = "o" * 64
    c, fake, ck = make_controller(tmp_path, {0: [(10, 4 * GiB, 0, OWNER)]})
    c.tick(0)
    c.write_status(tmp_path / "status.json", 0)
    gpus = {g["uuid"]: g for g in json.loads((tmp_path / "status.json").read_text())["gpus"]}
    assert gpus["GPU-0"]["spots"] == [{"container": "s" * 12, "share": 1.0, "seen": 0, "idle": True}]
    assert gpus["GPU-0"]["spot_share"] == 1.0 and gpus["GPU-1"]["spot_share"] == 0
    assert parse_status((tmp_path / "status.json").read_text(), 0)["GPU-0"].spot_share == 1.0


def test_a_handed_share_the_monitor_lists_is_not_counted_twice():
    status = {
        "A": GpuLending(False, None, "LENDABLE", spot_share=0.5, spot_capacity=40 * GiB, spots=((0.5, 105.0),)),
        "B": GpuLending(False, None, "LENDABLE", spot_capacity=40 * GiB),
    }
    # Handed out at 100, listed at 105: counted once, so a second 0.5 still goes to A.
    assert unreported([("A", 100.0, 0.5)], status) == {}
    assert pick_spot_gpu(["A", "B"], status, 0.5, unreported([("A", 100.0, 0.5)], status))[0] == "A"
    # Not listed yet (or an older spot was listed before the hand-out): still taken.
    assert unreported([("A", 110.0, 0.5)], status) == {"A": 0.5}
    assert unreported([("B", 100.0, 0.5)], status) == {"B": 0.5}
    # One listed spot accounts for one hand-out only.
    assert unreported([("A", 100.0, 0.5), ("A", 101.0, 0.5)], status) == {"A": 0.5}


def test_status_lists_when_each_spot_was_first_seen(tmp_path):
    c, fake, ck = make_controller(tmp_path, {0: [(10, 4 * GiB, 0, "o" * 64)]})
    c.tick(7)
    c.tick(9)
    c.write_status(tmp_path / "status.json", 9)
    st = parse_status((tmp_path / "status.json").read_text(), 9)
    assert st["GPU-0"].spots == ((1.0, 7.0),)
