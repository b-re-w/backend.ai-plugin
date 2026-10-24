"""Shared spot pool: GPU first come, first served; creation beyond GPU count; never evict idle spots (SPEC 2.16)."""

import json
import re
from pathlib import Path

import pytest

from labgpu.spot.config import Config, ControllerConfig, IdleConfig, PoolConfig, SpotConfig
from labgpu.spot.daemon import Controller
from labgpu.spot.docker import SpotContainer, parse_spot
from labgpu.spot.hostinfo import parse_meminfo_gib
from labgpu.spot.model import GpuState
from labgpu.spot.placement import IdleSpot, Move, Oom, Park, Rehome, RunningSpot, plan
from labgpu.spot.pool import UNLIMITED_SLOTS, Pool, memory_pressure, resolve
from labgpu.spotstatus import GpuLending, pick_idle_gpu, pool_slots
from labgpu.sizes import GiB

from test_spot_move import FakeCkpt, FakeGpus, FakeSessions, Inline, P6K, OwnerContainer, GpuInfo, v

KW = dict(now=10, can_checkpoint=True, oom_grace=10)


# ---- placement ----

def test_idle_spots_hold_no_share_in_pool_mode():
    verdicts = {
        "A": v("A", GpuState.LENT, must=True, reasons=("owner back",)),
        "B": v("B", GpuState.LENDABLE),
        "C": v("C", GpuState.LENDABLE),
    }
    running = [RunningSpot("s", frozenset({"A"}), ("A", "B", "C"), 0, share=0.75)]
    idle = [IdleSpot("i", "B", ("A", "B", "C"), share=0.5)]
    # Old mode: B is held for the idle spot, so the fuller fit is out and the move goes to C.
    [mv] = plan(verdicts, running, [], idle_spots=idle, **KW)
    assert mv.dst == "C"
    # Pool mode: the idle spot holds nothing, so the running one takes B (the fullest fit, tie to
    # the first); the idle spot would now have no room there and is given C before it starts.
    mv, re = plan(verdicts, running, [], idle_spots=idle, idle_holds_share=False, **KW)
    assert isinstance(mv, Move) and mv.dst == "B" and re == Rehome("i", "B", "C")


def test_late_starter_waits_instead_of_pushing_out_the_first():
    verdicts = {"A": v("A", GpuState.LENT), "B": v("B", GpuState.BUSY)}
    first = RunningSpot("first", frozenset({"A"}), ("A", "B"), since=0, share=0.6)
    late = RunningSpot("late", frozenset({"A"}), ("A", "B"), since=5, share=0.6, oom_ready=False)
    [act] = plan(verdicts, [first, late], [], idle_holds_share=False, **KW)
    assert isinstance(act, Park) and act.container_id == "late"  # no target: it waits parked


def test_late_starter_moves_to_a_gpu_with_room():
    verdicts = {"A": v("A", GpuState.LENT), "B": v("B", GpuState.LENDABLE)}
    first = RunningSpot("first", frozenset({"A"}), ("A", "B"), since=0, share=0.6)
    late = RunningSpot("late", frozenset({"A"}), ("A", "B"), since=5, share=0.6)
    [act] = plan(verdicts, [first, late], [], idle_holds_share=False, **KW)
    assert act == Move("late", "A", "B", "spot sessions on this GPU exceed one GPU in total")


def test_late_starter_with_the_oom_handler_gets_the_error_not_a_kill():
    verdicts = {"A": v("A", GpuState.LENT), "B": v("B", GpuState.BUSY)}
    first = RunningSpot("first", frozenset({"A"}), ("A", "B"), since=0, share=0.6)
    late = RunningSpot("late", frozenset({"A"}), ("A", "B"), since=5, share=0.6)
    [act] = plan(verdicts, [first, late], [], idle_holds_share=False, **KW)
    assert isinstance(act, Oom) and act.container_id == "late"


def test_idle_spot_on_a_full_gpu_is_rehomed_only_in_pool_mode():
    verdicts = {"A": v("A", GpuState.LENT), "B": v("B", GpuState.LENDABLE)}
    running = [RunningSpot("r", frozenset({"A"}), ("A", "B"), 0, share=0.8)]
    idle = [IdleSpot("i", "A", ("A", "B"), share=0.5)]
    assert plan(verdicts, running, [], idle_spots=idle, **KW) == []  # old mode: it holds A's 0.2... stays
    assert plan(verdicts, running, [], idle_spots=idle, idle_holds_share=False, **KW) == [Rehome("i", "A", "B")]
    # Nowhere better: it stays where it is, untouched.
    verdicts["B"] = v("B", GpuState.BUSY)
    assert plan(verdicts, running, [], idle_spots=idle, idle_holds_share=False, **KW) == []


def test_idle_spots_never_get_anything_but_a_rehome():
    """Owner back, pool crowded, no GPU left: a spot with no GPU process is not touched (SPEC 2.16.4)."""
    verdicts = {"A": v("A", GpuState.RECLAIMING, must=True), "B": v("B", GpuState.BUSY)}
    idle = [IdleSpot("i", "A", ("A", "B")), IdleSpot("j", "B", ("A", "B"), limited=True)]
    for holds in (True, False):
        assert plan(verdicts, [], [], idle_spots=idle, idle_holds_share=holds, **KW) == []


# ---- nothing is ever stopped ----

def test_no_code_path_stops_kills_or_removes_a_container():
    src = Path(__file__).resolve().parent.parent / "src" / "labgpu"
    pattern = re.compile(r"""["'](stop|kill|rm|rmi|restart|pause|prune)["']""")
    offenders = [
        f"{p.relative_to(src)}:{n}: {line.strip()}"
        for p in [*(src / "spot").glob("*.py"), src / "accelerator" / "spot_plugin.py"]
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if pattern.search(line)
    ]
    assert offenders == []


# ---- pool settings ----

def test_pool_is_off_by_default_and_when_values_are_bad():
    assert resolve(PoolConfig(), 32, 128.0) == (None, [])
    pool, notes = resolve(PoolConfig(enabled=True), 32, 128.0)
    assert pool == Pool(8, 32.0, 16, 8.0) and notes == [] and pool.slots == 16
    for bad in (
        PoolConfig(enabled=True, cpu_cores=0),
        PoolConfig(enabled=True, overcommit_max_sessions=-1),
        PoolConfig(enabled=True, preset_name="small"),
    ):
        pool, notes = resolve(bad, 32, 128.0)
        assert pool is None and notes[0][0] == "error"
    assert resolve(PoolConfig(enabled=True), None, 128.0)[0] is None  # node size unreadable


def test_pool_is_clamped_to_the_node_and_unlimited_sessions_need_a_number():
    pool, notes = resolve(PoolConfig(enabled=True, cpu_cores=64, mem_gib=512, overcommit_max_sessions=0), 16, 64.0)
    assert (pool.cpu_cores, pool.mem_gib) == (16, 64.0) and {n[0] for n in notes} == {"warning"}
    assert pool.slots == UNLIMITED_SLOTS


def test_pool_settings_come_from_etcd_strings():
    cfg = Config.from_dict({"pool": {"enabled": "true", "cpu_cores": "4", "mem_gib": "16",
                                     "overcommit_max_sessions": "0", "mem_floor_gib": "2.5"}})
    assert cfg.pool == PoolConfig(True, 4, 16.0, "", 0, 2.5)
    assert Config().pool.enabled is False  # opt-in
    with pytest.raises(ValueError):
        Config.from_dict({"pool": {"typo": 1}})


def test_memory_pressure_is_fail_safe():
    pool = Pool(8, 32.0, 16, 8.0)
    assert memory_pressure(pool, 7.9) and not memory_pressure(pool, 8.0) and memory_pressure(pool, None)
    assert parse_meminfo_gib("MemTotal: 1048576 kB\nMemAvailable:  524288 kB\n", "MemAvailable") == 0.5


# ---- slot figures ----

def lend(state, share=0.0, cap=80 * GiB):
    return GpuLending(False, None, state, spot_share=share, spot_capacity=cap)


def test_pool_slots_do_not_depend_on_gpu_count_or_state():
    st = {"A": lend("BUSY"), "B": lend("BUSY")}  # every GPU is the owners'
    assert pool_slots(["A", "B"], st, 16) == 16  # still creatable: it only waits when it uses the GPU
    assert pool_slots(["A", "B"], None, 16) == 0  # stale or no status: none (fail safe)
    assert pool_slots(["A", "B"], {"A": lend(""), "B": lend("")}, 16) == 0  # spot off in the monitor
    assert pool_slots(["X"], st, 16) == 0  # no GPU of this model on the node
    assert pool_slots([], st, 16) == 0


def test_pick_idle_gpu_takes_the_roomiest_and_reserves_nothing():
    st = {"A": lend("LENT", 0.7), "B": lend("LENT", 0.2), "C": lend("BUSY")}
    assert pick_idle_gpu(["A", "B", "C"], st, 0.5) == ("B", 80 * GiB, 0.5)
    assert pick_idle_gpu(["A", "B", "C"], st, 0.5) == ("B", 80 * GiB, 0.5)  # asking again changes nothing
    st = {"A": lend("LENT", 0.9)}
    assert pick_idle_gpu(["A"], st, 0.5) == ("A", 80 * GiB, pytest.approx(0.1))  # capped at the room left
    assert pick_idle_gpu(["C"], {"C": lend("BUSY")}, 0.5) is None
    assert pick_idle_gpu(["A"], None, 0.5) is None


# ---- controller ----

def pool_controller(tmp_path, procs, *, mem=lambda: 100.0, pool=True, spots=1):
    gpus = [GpuInfo(i, f"GPU-{i}", P6K, 96 * GiB, f"0:{i}") for i in range(3)]
    owner = OwnerContainer("o" * 64, 0, ("GPU-0",))
    spot_list = [
        SpotContainer(
            chr(ord("s") + k) * 64, 0, ("GPU-0", "GPU-1", "GPU-2"), gpu="GPU-0", user="1000:1000",
            share=0.5, cpus=4, mem=8 * GiB,
        )
        for k in range(spots)
    ]
    cfg = Config(
        controller=ControllerConfig(state_dir=tmp_path),
        idle=IdleConfig(idle_minutes=1, owner_cpu_threshold=0),
        spot=SpotConfig(oom_grace_seconds=10),
        pool=PoolConfig(enabled=pool, mem_floor_gib=8.0),
    )
    ck = FakeCkpt()
    c = Controller(cfg, FakeGpus(gpus, procs), FakeSessions([owner], spot_list), checkpointer=ck,
                   workers=Inline(), handler_check=lambda pid: True, pid_check=lambda pid, cid: True,
                   node_info=lambda: (32, 128.0), mem_available=mem)
    c.start()
    return c, ck


def test_status_idle_spots_hold_no_share_and_report_the_pool(tmp_path):
    OWNER = "o" * 64
    c, ck = pool_controller(tmp_path, {0: [(10, 4 * GiB, 0, OWNER)]}, spots=2)
    for t in (0, 61, 70):
        c.tick(t)
    c.write_status(tmp_path / "status.json", 70)
    st = json.loads((tmp_path / "status.json").read_text())
    gpu0 = st["gpus"][0]
    assert gpu0["spot_share"] == 0 and all(s.get("idle") for s in gpu0["spots"])
    assert st["pool"]["enabled"] and st["pool"]["sessions"] == 2 and st["pool"]["max_sessions"] == 16
    assert st["pool"]["requested_cpu_cores"] == 8 and st["pool"]["requested_mem_gib"] == 16.0
    assert {s["state"] for s in st["spot_sessions"].values()} == {"idle"}
    off, _ = pool_controller(tmp_path, {0: [(10, 4 * GiB, 0, OWNER)]}, pool=False)
    for t in (0, 61, 70):
        off.tick(t)
    off.write_status(tmp_path / "status2.json", 70)
    st = json.loads((tmp_path / "status2.json").read_text())
    assert st["pool"] == {"enabled": False} and st["gpus"][0]["spot_share"] == 0.5  # old behaviour


def test_owner_return_touches_no_idle_spot(tmp_path):
    OWNER = "o" * 64
    procs = {0: [(10, 4 * GiB, 0, OWNER)]}
    c, ck = pool_controller(tmp_path, procs, spots=2)
    for t in (0, 61, 70):
        c.tick(t)
    ck.calls.clear()
    procs[0][0] = (10, 4 * GiB, 90, OWNER)  # owner active again
    for t in (75, 80, 500, 5000):
        c.tick(t)
    # At most device-file changes and rehomes; no move, park, error, or anything else to a session.
    assert {call[0] for call in ck.calls} <= {"gate", "place_as"}
    assert set(c.spot_info) == {"s" * 64, "t" * 64} and not c.parked and not c.busy


def test_memory_pressure_stops_new_grants_but_stops_nothing(tmp_path):
    OWNER = "o" * 64
    mem = [100.0]
    procs = {0: [(10, 4 * GiB, 0, OWNER)]}
    c, ck = pool_controller(tmp_path, procs, mem=lambda: mem[0])
    for t in (0, 61, 70):
        c.tick(t)
    procs[0][0] = (10, 4 * GiB, 90, OWNER)  # GPU-0 no longer hosts the idle spot: it would be rehomed
    mem[0] = 1.0
    ck.calls.clear()
    c.tick(75)
    assert c.mem_pressure and not any(call[0] == "place_as" for call in ck.calls)
    mem[0] = 100.0
    c.tick(80)
    assert not c.mem_pressure


def test_parse_spot_counts_requested_cpu_and_memory():
    s = parse_spot({"Id": "x", "Config": {"Env": []}, "HostConfig": {"CpusetCpus": "0-3,8", "Memory": 4 * GiB}})
    assert (s.cpus, s.mem) == (5, 4 * GiB)
    s = parse_spot({"Id": "x", "Config": {"Env": []}, "HostConfig": {}})
    assert (s.cpus, s.mem) == (0, 0)
