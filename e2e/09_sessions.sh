D=$(dirname "$0"); . $D/env.sh
A="$PY $D/api.py"
q() { $A "$@" 2>&1 | grep -v "RPC authentication\|Deprecat\|warnings.warn" | tail -3; }
q start v1-p6000 pro6000.shares 0.5
q start v2-p6000 pro6000.shares 0.5
q start v3-a6000 a6000.shares 1
q start v4-p5000l   pro5000l.shares 0.25
q start v5-toobig a6000.shares 1
echo "--- containers"
for c in $(docker ps -q --filter label=ai.backend.kernel-id); do
  docker inspect $c --format '{{index .Config.Labels "ai.backend.kernel-id"}}' | cut -c1-8 | tr '\n' ' '
  docker inspect $c --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E "^(LABGPU_DEVICE_UUIDS|CUDA_DEVICE_|LD_PRELOAD)" | tr '\n' ' '
  echo
done
$A slots 2>&1 | grep -A2 "^agent"
