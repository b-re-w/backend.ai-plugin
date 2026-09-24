#!/bin/bash
# Live spot-lending scenario against the running Backend.AI owner sessions (fake NVML).
D=$(dirname "$0"); . $D/env.sh; R=$W/run; SP=$R/spot
export LABGPU_FAKE_NVML=$R/labgpu/gpus.json
SPOT="$(dirname $PY)/labgpu-spot -c $SP/spot.toml"
FG="$PY $D/fakegpu.py $LABGPU_FAKE_NVML"
step() { echo; echo "=== $*"; }
show() {
  $SPOT status 2>&1 | sed 's/^/  /'
  $SPOT ls --all 2>&1 | sed 's/^/  /'
  docker ps --filter label=labgpu.spot=1 --format '  spot: {{.Names}} {{.Status}}'
}
docker ps -aq --filter label=labgpu.spot=1 | xargs -r docker rm -f >/dev/null
rm -rf $SP && mkdir -p $SP/jobs $SP/vf/job1
cat > $SP/spot.toml <<T
[controller]
poll_interval = 2
state_dir = "$SP/state"
kill_switch_file = "$SP/spot.disabled"
[idle]
idle_minutes = 0.5
unclaimed_grace_seconds = 10
[reclaim]
grace_seconds = 5
mem_reserve_mib = 2048
[spot]
hook_path = "$R/labgpu/libvgpu.so"
default_ram = "256m"
host_ram_reserve = "1g"
allowed_mount_roots = ["$SP/vf"]
T
LOOP='trap "echo got SIGTERM, saving checkpoint; exit 143" TERM; i=0; while true; do i=$((i+1)); echo tick $i; sleep 1; done'
cat > $SP/jobs/job1.toml <<T
name = "p6000-only"
image = "stable/labgpu-test:1.0"
entrypoint = ""
command = ["sh", "-c", '$LOOP']
gpu_models = ["*PRO 6000*"]
gpu_mem = "10g"
[[mounts]]
src = "$SP/vf/job1"
dst = "/ckpt"
T
cat > $SP/jobs/job2.toml <<T
name = "needs-60g"
image = "stable/labgpu-test:1.0"
entrypoint = ""
command = ["sh", "-c", '$LOOP']
gpu_models = ["*A6000*"]
gpu_mem = "60g"
T
cat > $SP/jobs/job3.toml <<T
name = "any-gpu"
image = "stable/labgpu-test:1.0"
entrypoint = ""
command = ["sh", "-c", '$LOOP']
T
cat > $SP/jobs/bad.toml <<T
name = "escape"
image = "stable/labgpu-test:1.0"
entrypoint = ""
command = ["true"]
[[mounts]]
src = "/etc"
dst = "/x"
T

step "1. owners hold 4 GiB each, idle; submit jobs"
$FG attach-owners
$FG xorg-everywhere   # like the lab servers: must not block lending (SPEC 2.1 ignored processes)
for j in job1 job2 job3; do echo "  submit $j -> $($SPOT submit $SP/jobs/$j.toml)"; done
echo "  submit bad (mount outside roots) -> $($SPOT submit $SP/jobs/bad.toml 2>&1)"

step "2. start controller; nothing is lent during the first idle window"
nohup setsid $SPOT daemon > $SP/daemon.log 2>&1 < /dev/null &
sleep 12; show

step "3. after idle_minutes (30s): lend"
sleep 26; show
echo "  run args of job1:"; docker inspect $(docker ps -q --filter label=labgpu.job-id=1) \
  --format '    user={{.Config.User}} mem={{.HostConfig.Memory}} cpushares={{.HostConfig.CpuShares}} oom={{.HostConfig.OomScoreAdj}} mounts={{range .Mounts}}{{.Source}}->{{.Destination}} {{end}}' 2>/dev/null
docker inspect $(docker ps -q --filter label=labgpu.job-id=1) --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | grep -E "CUDA_DEVICE|LD_PRELOAD|LABGPU" | sed 's/^/    /'

step "4. owner of the GPU running job1 comes back (SM 60%) -> reclaim, requeue, re-lend elsewhere"
G1=$(docker ps --filter label=labgpu.job-id=1 --format '{{.Label "labgpu.gpu-uuid"}}')
echo "  job1 was on $G1"; $FG owner-util $G1 60
sleep 14; show
C=$(docker ps -aq --filter label=labgpu.job-id=1 | tail -1)
ls $SP/state/logs/ 2>/dev/null | sed 's/^/  log: /'
tail -n 2 $SP/state/logs/1.* 2>/dev/null | sed 's/^/    /'

step "5. unknown host process appears on the GPU running job3 -> reclaim"
G3=$(docker ps --filter label=labgpu.job-id=3 --format '{{.Label "labgpu.gpu-uuid"}}')
echo "  job3 was on $G3"; $FG stranger $G3
sleep 10; show

step "6. pause the node -> everything reclaimed, nothing lent"
$FG clear-stranger >/dev/null
$SPOT pause; sleep 12; show
$SPOT resume

step "7. SIGTERM the controller -> it reclaims what it lent before exiting"
sleep 40; docker ps --filter label=labgpu.spot=1 --format '  before stop: {{.Names}}'
pkill -TERM -f "labgpu-spot -c $SP/spot.toml daemon"; sleep 12
docker ps --filter label=labgpu.spot=1 --format '  still running: {{.Names}}'
echo "  (no 'still running' lines = all reclaimed)"
$SPOT ls --all | sed 's/^/  /'

step "controller log (decisions)"
grep -E "LEND|RECLAIM|ended|LAUNCH|unmanaged" $SP/daemon.log | cut -c1-230
