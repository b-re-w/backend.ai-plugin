D=$(dirname "$0"); . $D/env.sh; R=$W/run; SP=$R/spot-real; unset LABGPU_FAKE_NVML
SPOT="$(dirname $PY)/labgpu-spot -c $SP/spot.toml"
echo "=== sessions see the real GPU, attached by UUID"
for c in $(docker ps -q --filter label=ai.backend.kernel-id); do
  docker inspect $c --format '  {{.Name}} DeviceRequests={{json .HostConfig.DeviceRequests}}' | cut -c1-200
  docker exec -e LD_PRELOAD= $c nvidia-smi -L 2>&1 | sed 's/^/    /'
done
$PY $D/api.py destroy real-g2 | grep destroy; sleep 5
docker ps -aq --filter label=labgpu.spot=1 | xargs -r docker rm -f >/dev/null
rm -rf $SP && mkdir -p $SP
cat > $SP/spot.toml <<T
[controller]
poll_interval = 2
state_dir = "$SP/state"
kill_switch_file = "$SP/off"
[idle]
idle_minutes = 0.5
owner_cpu_threshold = 0
[reclaim]
grace_seconds = 5
mem_reserve_mib = 512
[spot]
hook_path = "$R/labgpu/hami/libvgpu.so"
default_ram = "256m"
host_ram_reserve = "1g"
allowed_mount_roots = ["$SP"]
T
cat > $SP/job.toml <<'T'
name = "real-gpu-spot"
image = "stable/labgpu-test:1.0"
entrypoint = ""
command = ["sh", "-c", "LD_PRELOAD= nvidia-smi -L; trap 'echo got SIGTERM; exit 143' TERM; while true; do sleep 1; done"]
gpu_models = ["*RTX 4050*"]
T
echo "=== submit + start controller (real NVML)"; $SPOT submit $SP/job.toml
nohup setsid $SPOT daemon > $SP/daemon.log 2>&1 < /dev/null &
sleep 42; $SPOT status; $SPOT ls
J=$(docker ps -q --filter label=labgpu.spot=1)
docker inspect $J --format '  spot DeviceRequests={{json .HostConfig.DeviceRequests}}' 2>/dev/null | cut -c1-200
docker logs $J 2>&1 | grep -v "^$" | sed 's/^/  spot log: /' | head -3
echo "=== a new Backend.AI session takes the other half of the GPU -> new owner -> reclaim"
$PY $D/api.py start real-g3 rtx4050.shares 0.5 2>&1 | grep status
sleep 12; $SPOT status; $SPOT ls
grep -E "LEND|RECLAIM|ended" $SP/daemon.log | cut -c1-220
pkill -TERM -f "labgpu-spot -c $SP/spot.toml daemon"; sleep 3
