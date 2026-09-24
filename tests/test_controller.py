"""End-to-end controller scenarios with fake NVML and Docker."""

from dataclasses import replace
from pathlib import Path

from labgpu.nvml import GpuInfo, GpuProcess, GpuSnapshot, NvmlError
from labgpu.sizes import GiB
from labgpu.spot.config import Config, ControllerConfig, IdleConfig, SpotConfig
from labgpu.spot.daemon import Controller
from labgpu.spot.docker import DockerError, OwnerContainer, SpotContainer
from labgpu.spot.jobs import JobState, JobStore
from labgpu.spot.jobspec import JobSpec
from labgpu.spot.model import GpuState

OWNER = "o" * 64
GPU = GpuInfo(0, "GPU-a", "A100", 40 * GiB, "00000000:01:00.0")


class FakeGpus:
    def __init__(self):
        self.procs: list[GpuProcess] = [GpuProcess(10, 4 * GiB, 0)]
        self.fail = False

    def list_gpus(self):
        return [GPU]

    def snapshot(self, index):
        if self.fail:
            raise NvmlError("Xid 79")
        used = sum(p.used_memory for p in self.procs)
        return GpuSnapshot(GPU, used, 0, tuple(self.procs))


class FakeDocker:
    def __init__(self):
        self.owners = [OwnerContainer(OWNER, 100, ("GPU-a",))]
        self.spots: dict[str, SpotContainer] = {}
        self.stopped: list[str] = []
        self.removed: list[str] = []
        self.runs: list[list[str]] = []
        self.down = False

    def owner_containers(self):
        if self.down:
            raise DockerError("daemon down")
        return list(self.owners)

    def spot_containers(self):
        if self.down:
            raise DockerError("daemon down")
        return list(self.spots.values())

    def run(self, argv):
        cid = f"{len(self.runs):064x}"
        self.runs.append(list(argv))
        job_id = int(argv[argv.index("--label", argv.index("--label") + 1) + 1].split("=")[1])
        self.spots[cid] = SpotContainer(cid, f"spot{job_id}", job_id, "GPU-a", True, None)
        return cid

    def stop_async(self, container, grace_seconds):
        self.stopped.append(container)

    def exit(self, container, code):
        self.spots[container] = replace(self.spots[container], running=False, exit_code=code)

    def save_logs_and_remove(self, container, log_path):
        self.removed.append(container)
        self.spots.pop(container, None)

    def wait_stops(self, timeout):
        pass


def make(tmp_path: Path, **spot_kw):
    cfg = Config(
        controller=ControllerConfig(state_dir=tmp_path, kill_switch_file=tmp_path / "off"),
        idle=IdleConfig(idle_minutes=10, owner_cpu_threshold=0),
        spot=SpotConfig(default_ram=4 * GiB, host_ram_reserve=8 * GiB, **spot_kw),
    )
    store = JobStore(tmp_path / "spot.db")
    gpus, docker = FakeGpus(), FakeDocker()
    pid_map = {10: OWNER, 11: OWNER}
    ctl = Controller(
        cfg, store, gpus, docker,
        pid_mapper=lambda pids: {p: pid_map.get(p) or next(
            (c for c in docker.spots if c.startswith("0")), None) for p in pids},
        ram_available=lambda: 64 * GiB,
        hook_present=lambda: True,
    )
    spec = JobSpec.from_dict({"name": "j", "image": "img", "command": ["run"], "gpu_mem": "8g"})
    jid = store.submit(spec, 1000, 1000)
    return ctl, store, gpus, docker, jid


def test_lend_reclaim_requeue_and_relend(tmp_path):
    ctl, store, gpus, docker, jid = make(tmp_path)

    ctl.tick(0)
    assert docker.runs == []  # just started: every GPU counts as recently used
    ctl.tick(599)
    assert docker.runs == []
    ctl.tick(600)
    assert len(docker.runs) == 1 and store.get(jid).state is JobState.RUNNING
    spot_cid = next(iter(docker.spots))

    # The spot's own GPU usage must not look like owner activity.
    gpus.procs.append(GpuProcess(20, 20 * GiB, 95))
    ctl.tick(605)
    assert docker.stopped == [] and ctl.last_verdicts["GPU-a"].state is GpuState.LENT

    # Owner comes back.
    gpus.procs[0] = GpuProcess(10, 4 * GiB, 70)
    ctl.tick(610)
    assert docker.stopped == [spot_cid]
    assert store.get(jid).state is JobState.PREEMPTING

    # Container exits after SIGTERM → job requeued, logs archived.
    docker.exit(spot_cid, 143)
    gpus.procs = [GpuProcess(10, 4 * GiB, 70)]
    ctl.tick(615)
    rec = store.get(jid)
    assert rec.state is JobState.QUEUED and rec.preemptions == 1
    assert docker.removed == [spot_cid]

    # Owner idles again for idle_minutes → lent again as attempt 2.
    gpus.procs = [GpuProcess(10, 4 * GiB, 0)]
    ctl.tick(1214)
    assert len(docker.runs) == 1
    ctl.tick(1215)
    assert len(docker.runs) == 2 and "labgpu-spot-1-2" in docker.runs[1]


def test_job_completion(tmp_path):
    ctl, store, gpus, docker, jid = make(tmp_path)
    ctl.tick(0)
    ctl.tick(600)
    docker.exit(next(iter(docker.spots)), 0)
    ctl.tick(605)
    assert store.get(jid).state is JobState.SUCCEEDED


def test_nvml_failure_reclaims(tmp_path):
    ctl, store, gpus, docker, jid = make(tmp_path)
    ctl.tick(0)
    ctl.tick(600)
    gpus.fail = True
    ctl.tick(605)
    assert len(docker.stopped) == 1
    assert ctl.last_verdicts["GPU-a"].state is GpuState.UNKNOWN


def test_docker_failure_reclaims_and_does_not_lend(tmp_path):
    ctl, store, gpus, docker, jid = make(tmp_path)
    ctl.tick(0)
    ctl.tick(600)
    docker.down = True
    ctl.tick(605)
    assert len(docker.stopped) == 1
    assert store.get(jid).state is JobState.PREEMPTING


def test_no_lending_without_hami(tmp_path):
    ctl, store, gpus, docker, jid = make(tmp_path)
    ctl.hook_present = lambda: False
    ctl.tick(0)
    ctl.tick(600)
    assert docker.runs == []


def test_pause_and_kill_switch(tmp_path):
    ctl, store, gpus, docker, jid = make(tmp_path)
    ctl.tick(0)
    ctl.tick(600)
    assert len(docker.runs) == 1
    (tmp_path / "off").touch()
    ctl.tick(605)
    assert len(docker.stopped) == 1


def test_unmanaged_spot_container_is_removed(tmp_path):
    ctl, store, gpus, docker, jid = make(tmp_path)
    docker.spots["x" * 64] = SpotContainer("x" * 64, "stray", 99, "GPU-a", True, None)
    ctl.tick(0)
    assert docker.removed == ["x" * 64]


def test_restart_resumes_running_job(tmp_path):
    ctl, store, gpus, docker, jid = make(tmp_path)
    ctl.tick(0)
    ctl.tick(600)
    # A new controller over the same store and containers: keeps the job, starts BUSY-timed.
    ctl2 = Controller(ctl.cfg, store, gpus, docker, pid_mapper=ctl.pid_mapper,
                      ram_available=ctl.ram_available, hook_present=lambda: True)
    ctl2.tick(700)
    assert store.get(jid).state is JobState.RUNNING
    assert ctl2.last_verdicts["GPU-a"].state is GpuState.LENT
    assert docker.stopped == []


def test_launch_failure_marks_job_failed(tmp_path):
    ctl, store, gpus, docker, jid = make(tmp_path)

    def broken_run(argv):
        raise DockerError("No such image: img")

    docker.run = broken_run
    ctl.tick(0)
    ctl.tick(600)
    rec = store.get(jid)
    assert rec.state is JobState.FAILED and "No such image" in rec.reason
