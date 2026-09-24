D=$(dirname "$0"); . $D/env.sh; R=$W/run; cd $R
export LABGPU_FAKE_NVML=$R/labgpu/gpus.json LABGPU_CLAIM_DIR=$R/labgpu/claims
pkill -f "ai.backend.cli mgr start-server" 2>/dev/null; pkill -f "ai.backend.cli ag start-server" 2>/dev/null; sleep 1
nohup setsid $PY -m ai.backend.cli mgr start-server -f manager.toml > manager.log 2>&1 < /dev/null &
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:8081/ && break; sleep 2; done
echo "manager: $(curl -s http://127.0.0.1:8081/ | head -c 200)"
nohup setsid $PY -m ai.backend.cli ag start-server -f agent.toml > agent.log 2>&1 < /dev/null &
sleep 40
echo "--- agent log (labgpu / plugin / errors)"
grep -E "labgpu|gpu_slot|cuda_frac|FAKE|Resource slots|Slot types|error during init|Error|Traceback" agent.log | cut -c1-260 | head -30
