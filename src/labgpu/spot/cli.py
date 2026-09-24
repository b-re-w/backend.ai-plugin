"""`labgpu-spot` command line (SPEC 2.9)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, Config
from .jobs import JobStore
from .jobspec import JobSpec, JobSpecError


def _store(cfg: Config) -> JobStore:
    cfg.controller.state_dir.mkdir(parents=True, exist_ok=True)
    return JobStore(cfg.controller.state_dir / "spot.db")


def _parse_as(value: str) -> tuple[int, int]:
    uid, _, gid = value.partition(":")
    return int(uid), int(gid or uid)


def cmd_daemon(cfg: Config, args: argparse.Namespace) -> int:
    from ..nvml import open_reader
    from .daemon import Controller, run_forever
    from .docker import DockerCli

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    nvml = open_reader()
    if nvml.is_fake:
        logging.getLogger("ai.backend.labgpu.spot").warning(
            "FAKE NVML mode: GPUs are simulated and spot containers get no GPU attached"
        )
    controller = Controller(cfg, _store(cfg), nvml, DockerCli())
    if not controller.hook_present():
        logging.getLogger("ai.backend.labgpu.spot").error(
            "HAMi-core %s missing: spot GPU memory cannot be limited; %s",
            cfg.spot.hook_path,
            "lending anyway (allow_unenforced)" if cfg.spot.allow_unenforced else "not lending",
        )
    try:
        run_forever(controller)
    finally:
        nvml.close()
    return 0


def cmd_submit(cfg: Config, args: argparse.Namespace) -> int:
    try:
        spec = JobSpec.from_toml(Path(args.jobfile))
        spec.validate_mounts(cfg.spot.allowed_mount_roots)
    except (JobSpecError, OSError) as e:
        print(f"invalid job: {e}", file=sys.stderr)
        return 2
    if args.as_user:
        uid, gid = _parse_as(args.as_user)
    else:
        uid, gid = os.getuid(), os.getgid()  # type: ignore[attr-defined]
    if uid == 0:
        print("refusing to run a spot job as root; pass --as uid:gid", file=sys.stderr)
        return 2
    job_id = _store(cfg).submit(spec.resolved(), uid, gid)
    print(job_id)
    return 0


def cmd_ls(cfg: Config, args: argparse.Namespace) -> int:
    rows = _store(cfg).list(include_finished=args.all)
    print(f"{'ID':>5} {'STATE':<11} {'TRY':>3} {'PRE':>3} {'PRI':>3} {'GPU':<14} {'NAME':<24} REASON")
    for r in rows:
        gpu = (r.gpu_uuid or "-")[:14]
        print(
            f"{r.id:>5} {r.state:<11} {r.attempts:>3} {r.preemptions:>3} {r.priority:>3} "
            f"{gpu:<14} {r.name[:24]:<24} {r.reason or ''}"
        )
    return 0


def cmd_cancel(cfg: Config, args: argparse.Namespace) -> int:
    try:
        container = _store(cfg).cancel(args.job_id)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"cancelled {args.job_id}" + (" (the controller will stop its container)" if container else ""))
    return 0


def cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    path = cfg.controller.state_dir / "status.json"
    try:
        status = json.loads(path.read_text())
    except OSError:
        print("no status yet: is the daemon running?", file=sys.stderr)
        return 1
    age = time.time() - status["updated_at"]
    node_paused, gpus_paused = _store(cfg).paused()
    print(f"updated {age:.0f}s ago" + ("  [NODE PAUSED]" if node_paused else ""))
    for g in status["gpus"]:
        flag = " [paused]" if g["uuid"] in gpus_paused else ""
        print(
            f"{g['uuid']}  {g['state']:<10} idle {g['idle_for'] / 60:6.1f}m  "
            f"lendable {g['lendable_memory'] >> 20:>7}MiB{flag}  {'; '.join(g['reasons'])}"
        )
    return 0


def cmd_pause(cfg: Config, args: argparse.Namespace, paused: bool) -> int:
    _store(cfg).set_paused(paused, args.gpu)
    target = f"GPU {args.gpu}" if args.gpu else "node"
    print(f"{target} {'paused' if paused else 'resumed'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="labgpu-spot", description="Spot GPU lending controller")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("daemon", help="run the controller")
    p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("submit", help="queue a spot job")
    p.add_argument("jobfile")
    p.add_argument("--as", dest="as_user", metavar="UID:GID")
    p = sub.add_parser("ls", help="list jobs")
    p.add_argument("--all", action="store_true", help="include finished jobs")
    p = sub.add_parser("cancel", help="cancel a job")
    p.add_argument("job_id", type=int)
    sub.add_parser("status", help="show per-GPU lending state")
    for name in ("pause", "resume"):
        p = sub.add_parser(name, help=f"{name} lending on this node or one GPU")
        p.add_argument("--gpu", metavar="UUID")

    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    match args.command:
        case "daemon":
            return cmd_daemon(cfg, args)
        case "submit":
            return cmd_submit(cfg, args)
        case "ls":
            return cmd_ls(cfg, args)
        case "cancel":
            return cmd_cancel(cfg, args)
        case "status":
            return cmd_status(cfg, args)
        case "pause":
            return cmd_pause(cfg, args, True)
        case "resume":
            return cmd_pause(cfg, args, False)
    return 2


if __name__ == "__main__":
    sys.exit(main())
