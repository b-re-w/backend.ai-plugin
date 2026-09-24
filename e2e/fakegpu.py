"""Edit the live fake-NVML file: attach owner processes to GPUs, set utilization, add strangers."""

import json
import subprocess
import sys
from pathlib import Path

path = Path(sys.argv[1])
cmd = sys.argv[2]
data = json.loads(path.read_text())
by_uuid = {g["uuid"]: g for g in data["gpus"]}


def owners() -> dict[str, str]:
    """GPU UUID -> Backend.AI kernel container id, read from the containers' env."""
    ids = subprocess.run(
        ["docker", "ps", "-q", "--no-trunc", "--filter", "label=ai.backend.kernel-id"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    result = {}
    for cid in ids:
        env = subprocess.run(
            ["docker", "inspect", cid, "--format", "{{range .Config.Env}}{{println .}}{{end}}"],
            capture_output=True, text=True, check=True,
        ).stdout
        for line in env.splitlines():
            if line.startswith("LABGPU_DEVICE_UUIDS="):
                for uuid in line.split("=", 1)[1].split(","):
                    result[uuid] = cid
    return result


match cmd:
    case "attach-owners":  # every owned GPU gets one idle owner process holding 4 GiB
        for i, (uuid, cid) in enumerate(sorted(owners().items())):
            by_uuid[uuid]["processes"] = [{"pid": 1000 + i, "mem": "4g", "sm": 0, "container": cid}]
    case "owner-util":  # owner-util <uuid> <sm%>
        by_uuid[sys.argv[3]]["processes"][0]["sm"] = int(sys.argv[4])
    case "stranger":  # stranger <uuid>: a host process nobody owns
        by_uuid[sys.argv[3]].setdefault("processes", []).append({"pid": 9999, "mem": "1g", "sm": 0})
    case "clear-stranger":
        for g in data["gpus"]:
            g["processes"] = [p for p in g.get("processes", []) if p["pid"] != 9999]
path.write_text(json.dumps(data, indent=1))
print(cmd, {g["uuid"]: [(p["pid"], p.get("sm"), (p.get("container") or "-")[:12]) for p in g.get("processes", [])]
             for g in data["gpus"]})
