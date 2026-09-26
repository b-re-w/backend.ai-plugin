"""`labgpu-spot` command line (SPEC 2.9)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .config import Config


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
        logging.getLogger("ai.backend.labgpu.spot").warning("FAKE NVML mode: GPUs are simulated")
    try:
        run_forever(Controller(cfg, nvml, DockerCli()))
    finally:
        nvml.close()
    return 0


def cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    path = cfg.controller.state_dir / "status.json"
    try:
        status = json.loads(path.read_text())
    except OSError:
        print("no status yet: is the daemon running?", file=sys.stderr)
        return 1
    print(f"updated {time.time() - status['updated_at']:.0f}s ago")
    for g in status["gpus"]:
        print(
            f"{g['uuid']}  {g['state']:<10} idle {g['idle_for'] / 60:6.1f}m  "
            f"lendable {g['lendable_memory'] >> 20:>7}MiB  spot {g.get('lent_job') or '-':<12}  "
            f"{'; '.join(g['reasons'])}"
        )
    for p in status.get("parked", []):
        print(f"parked {p['container']} from {p['from']} for {time.time() - p['since']:.0f}s")
    for cid, op in status.get("busy", {}).items():
        print(f"in progress: {op} {cid}")
    if not status.get("can_move", True):
        print("spot sessions cannot be moved here (see the daemon log); they are evicted instead")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="labgpu-spot", description="GPU idleness monitor and spot session mover")
    parser.add_argument("-c", "--config", type=Path, default=None, help="TOML with the SPEC 2.2 sections")
    parser.add_argument(
        "--state-dir", type=Path, default=None, help="the agent's <var-base-path>/labgpu (SPEC 2.13)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("daemon", help="run the monitor")
    p.add_argument("-v", "--verbose", action="store_true")
    sub.add_parser("status", help="show per-GPU idleness")
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    if args.state_dir is not None:
        from dataclasses import replace

        cfg = replace(cfg, controller=replace(cfg.controller, state_dir=args.state_dir))
    match args.command:
        case "daemon":
            return cmd_daemon(cfg, args)
        case "status":
            return cmd_status(cfg, args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
