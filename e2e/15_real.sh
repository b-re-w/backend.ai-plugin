D=$(dirname "$0"); . $D/env.sh; R=$W/run
bash $D/08_agent_real.sh 40 | grep -E "gpu_slot_4|Resource slots" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-200
echo "=== baseline: plain container, whole GPU, no HAMi-core"
docker run --rm --gpus all -v $D/cuda_probe.py:/p.py:ro stable/labgpu-test:1.0 python3 /p.py 2>&1 | tail -4
for n in g1 g2; do $PY $D/api.py start real-$n rtx4050.shares 0.5 2>&1 | grep status; done
$PY $D/api.py slots 2>&1 | grep -A2 "^agent"
for c in $(docker ps -q --filter label=ai.backend.kernel-id); do
  echo "=== inside session container ${c}"
  docker cp $D/cuda_probe.py $c:/tmp/p.py
  docker exec -u work $c sh -c 'env | grep -E "^(CUDA_DEVICE|LABGPU)" ; python3 /tmp/p.py' 2>&1 | tail -8 || \
  docker exec $c sh -c 'python3 /tmp/p.py' 2>&1 | tail -8
done
