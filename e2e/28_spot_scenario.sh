# Spot scenario after `27_spot.sh setup` (fake NVML, fake cuda-checkpoint; SPEC 2.12):
#  1. two spot sessions fit (capacity 2), a third waits (PENDING)
#  2. both start computing on the same GPU: the later one is moved to the other PRO 6000
#  3. someone else uses the first GPU and there is no free one: that spot is parked (off the GPU)
#  4. the GPU frees up: the parked spot is restored there
#  5. busy again for longer than park_seconds: the spot is evicted
D=$(dirname "$0"); . $D/env.sh; R=$W/run; SP=$R/spot
export LABGPU_FAKE_NVML=$R/labgpu/gpus.json
A="$PY $D/api.py"
q() { $A "$@" 2>&1 | grep -v "RPC authentication\|Deprecat\|warnings.warn" | tail -2; }
st() { echo "--- $1"; bash $D/27_spot.sh status 2>&1 | grep -E "^GPU-fake-p6000|^parked|^in progress|occupied|pro6000-spot" | cut -c1-160; }
fg() { $PY $D/fakegpu.py $LABGPU_FAKE_NVML "$@" >/dev/null; }
P1=GPU-fake-p6000-1; P2=GPU-fake-p6000-2

echo "== 1. capacity"
q start spot-a pro6000-spot.device 1
q start spot-b pro6000-spot.device 1
( q start spot-c pro6000-spot.device 1 ) &   # waits: no room left
sleep 20
for c in $(docker ps -q --filter label=ai.backend.kernel-id); do
  docker inspect $c --format '{{.Name}} {{range .Config.Env}}{{println .}}{{end}}' | grep -E "^/|^LABGPU_SPOT" | tr '\n' ' '; echo
done

echo "== 2. both land on $P1 (as CUDA device 0 would)"
fg spot-on 0 $P1; fg spot-on 1 $P1
sleep 12; st "after placement"
grep -E "spot [0-9a-f]{12}: (move|park|restore|evict)" $SP/daemon.log | tail -3 | cut -c1-200

echo "== 3. a stranger on $P1, $P2 taken: park"
fg stranger $P1
sleep 12; st "after owner activity"

echo "== 4. $P1 frees up: restore"
fg clear-stranger
sleep 25; st "after the GPU is free again"

echo "== 5. busy for longer than park_seconds: evict"
fg stranger $P1
sleep 60; st "after park_seconds"
grep -E "spot [0-9a-f]{12}: (move|park|restore|evict)" $SP/daemon.log | tail -8 | cut -c1-200
fg clear-stranger
