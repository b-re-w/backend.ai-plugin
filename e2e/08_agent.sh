D=$(dirname "$0"); . $D/env.sh; R=$W/run; cd $R
export LABGPU_FAKE_NVML=$R/labgpu/gpus.json LABGPU_CLAIM_DIR=$R/labgpu/claims
bash $D/stop_agent.sh
nohup setsid $PY -m ai.backend.cli ag start-server -f agent.toml > agent.log 2>&1 < /dev/null &
sleep ${1:-45}
sed 's/\x1b\[[0-9;]*m//g' agent.log | grep -E "labgpu|gpu_slot|cuda_frac|krunner|Resource slots|Slot types|initialization of plugin|Traceback|Error" | cut -c1-300 | head -30
