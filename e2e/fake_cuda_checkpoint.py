#!/usr/bin/env python3
"""
Stand-in for NVIDIA cuda-checkpoint in the fake-NVML E2E (never on a real node).

It edits the fake NVML file named by $LABGPU_FAKE_NVML the way the real tool changes what NVML
sees: `checkpoint` takes the process off its GPU (kept in <file>.parked), `restore` puts it back,
on the GPU given by --device-map if any. lock/unlock are no-ops.
"""

import argparse
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--action", choices=["lock", "checkpoint", "restore", "unlock"])
parser.add_argument("--pid", type=int)
parser.add_argument("--device-map", default="")
parser.add_argument("--timeout")
args = parser.parse_args()

path = Path(os.environ["LABGPU_FAKE_NVML"])
parked_path = path.with_suffix(".parked")
data = json.loads(path.read_text())
parked = json.loads(parked_path.read_text()) if parked_path.exists() else {}

if args.action == "checkpoint":
    for g in data["gpus"]:
        for p in list(g.get("processes", [])):
            if p["pid"] == args.pid:
                g["processes"].remove(p)
                parked[str(args.pid)] = {"gpu": g["uuid"], "proc": p}
                break
    if str(args.pid) not in parked:
        sys.exit(f"pid {args.pid} is not on any GPU")
elif args.action == "restore":
    entry = parked.pop(str(args.pid), None)
    if entry is None:
        sys.exit(f"pid {args.pid} is not checkpointed")
    mapping = dict(pair.split("=", 1) for pair in args.device_map.split(",") if pair)
    dst = mapping.get(entry["gpu"], entry["gpu"])
    next(g for g in data["gpus"] if g["uuid"] == dst).setdefault("processes", []).append(entry["proc"])
path.write_text(json.dumps(data, indent=1))
parked_path.write_text(json.dumps(parked))
print(f"fake cuda-checkpoint {args.action} {args.pid} {args.device_map}")
