# Spot launch mode end to end with fake NVML and a fake cuda-checkpoint (SPEC 2.12).
#
#   bash 27_spot.sh setup     # etcd + agent config for gpu_spot_1 (pro6000-spot), start the monitor
#   bash 27_spot.sh status    # monitor status and the agent's spot capacity
#   bash 27_spot.sh stop      # stop the monitor
#
# Moving is exercised by e2e/fake_cuda_checkpoint.py, which moves processes inside the fake NVML
# file; nothing here touches a real GPU.
D=$(dirname "$0"); . $D/env.sh; R=$W/run; SP=$R/spot
export LABGPU_FAKE_NVML=$R/labgpu/gpus.json
SPOT="$(dirname $PY)/labgpu-spot -c $SP/spot.toml"
BAI="$PY -m ai.backend.cli mgr -f $R/manager.toml"
put() { (cd $R && $BAI etcd put "$1" "$2" 2>/dev/null); }

stop_monitor() { pkill -TERM -f "labgpu-spot -c $SP/spot.toml daemon" 2>/dev/null; sleep 2; true; }

case "${1:-status}" in
setup)
  stop_monitor
  rm -rf $SP && mkdir -p $SP/state
  rm -f $R/labgpu/gpus.parked
  cat > $SP/cuda-checkpoint <<S
#!/bin/sh
exec $PY $D/fake_cuda_checkpoint.py "\$@"
S
  chmod +x $SP/cuda-checkpoint
  cat > $SP/spot.toml <<T
[controller]
poll_interval = 3
state_dir = "$SP/state"
[idle]
idle_minutes = 0.5
unclaimed_grace_seconds = 10
[spot]
cuda_checkpoint = "$SP/cuda-checkpoint"
park_seconds = 40
evict_grace_seconds = 3
T
  $PY $D/fakegpu.py $LABGPU_FAKE_NVML driver 580.178.04-fake >/dev/null
  put config/plugins/accelerator/gpu_spot_1/key pro6000-spot
  put config/plugins/accelerator/gpu_spot_1/model_pattern "*PRO 6000*"
  put config/plugins/accelerator/gpu_spot_1/spot_status_path $SP/state/status.json
  put config/plugins/accelerator/gpu_spot_1/monitor_enabled false   # this E2E runs the monitor itself, with the fake cuda-checkpoint
  for n in 1 2 3 4; do put config/plugins/accelerator/gpu_slot_$n/spot_status_path $SP/state/status.json; done
  put config/resource_slots/pro6000-spot.device count
  $PY $D/slot_types.py $W/bai/fixtures/manager/example-resource-slot-types.json \
    "pro6000-spot:PRO 6000 Spot:PRO6000-SPOT:device" > $SP/slot-types.json
  (cd $R && $BAI fixture populate $SP/slot-types.json 2>&1 | tail -1)
  # The agent only allocates device keys listed in allocation-order.
  sed -i 's/^allocation-order = \["pro6000", /allocation-order = ["pro6000", "pro6000-spot", /' $R/agent.toml
  grep -n "^allocation-order" $R/agent.toml
  # The image must list the new accelerator key, or the launcher will not offer it.
  sed -i 's/ai.backend.accelerators="\([^"]*\)"/ai.backend.accelerators="cuda,pro6000,pro6000-spot,pro5000l,a6000,rtx4050"/' $W/image/Dockerfile
  docker build -q -t stable/labgpu-test:1.0 -t local/stable/labgpu-test:1.0 $W/image | tail -1
  (cd $R && $BAI image rescan local 2>&1 | tail -1)
  nohup setsid $SPOT daemon > $SP/daemon.log 2>&1 < /dev/null &
  sleep 15
  bash $D/08_agent.sh 40 | sed 's/\x1b\[[0-9;]*m//g' | grep -E "spot|Resource slots" | cut -c1-200
  ;;
status)
  $SPOT status
  $PY $D/api.py slots 2>&1 | grep -A2 "^agent" | cut -c1-400
  tail -5 $SP/daemon.log
  ;;
stop)
  stop_monitor
  ;;
esac
