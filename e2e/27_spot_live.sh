# Keep a spot controller running (fake NVML) so the WebUI session detail can show live lending.
D=$(dirname "$0"); . $D/env.sh; R=$W/run; SP=$R/spot-live
export LABGPU_FAKE_NVML=$R/labgpu/gpus.json
SPOT="$(dirname $PY)/labgpu-spot -c $SP/spot.toml"
BAI="$PY -m ai.backend.cli mgr -f $R/manager.toml"

# The accelerator plugins read the controller's status file (SPEC 1.13).
for n in 1 2 3 4; do (cd $R && $BAI etcd put config/plugins/accelerator/gpu_slot_$n/spot_status_path $SP/state/status.json 2>/dev/null); done
bash $D/08_agent.sh 35 | sed 's/\x1b\[[0-9;]*m//g' | grep "Resource slots" | cut -c1-120

pkill -TERM -f "labgpu-spot -c $SP/spot.toml daemon" 2>/dev/null; sleep 2
docker ps -aq --filter label=labgpu.spot=1 | xargs -r docker rm -f >/dev/null
rm -rf $SP && mkdir -p $SP/vf
cat > $SP/spot.toml <<T
[controller]
poll_interval = 3
state_dir = "$SP/state"
kill_switch_file = "$SP/spot.disabled"
[idle]
idle_minutes = 0.5
unclaimed_grace_seconds = 10
[reclaim]
grace_seconds = 5
[spot]
hook_path = "$R/labgpu/libvgpu.so"
default_ram = "256m"
host_ram_reserve = "1g"
allowed_mount_roots = ["$SP/vf"]
T
for n in 1 2; do cat > $SP/job$n.toml <<T
name = "demo-$n"
image = "stable/labgpu-test:1.0"
entrypoint = ""
command = ["sh", "-c", "trap 'exit 143' TERM; while true; do sleep 1; done"]
gpu_models = ["*PRO 6000*"]
T
$SPOT submit $SP/job$n.toml; done
$PY $D/fakegpu.py $LABGPU_FAKE_NVML attach-owners >/dev/null
$PY $D/fakegpu.py $LABGPU_FAKE_NVML xorg-everywhere >/dev/null
nohup setsid $SPOT daemon > $SP/daemon.log 2>&1 < /dev/null &
sleep 45
$SPOT status; $SPOT ls
