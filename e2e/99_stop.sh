# Stop everything the E2E started; data under ~/labgpu-e2e is kept.
D=$(dirname "$0"); . $D/env.sh
for s in real-g1 real-g2 real-g3; do $PY $D/api.py destroy $s 2>&1 | grep -E "^destroyed"; done
pkill -TERM -f "labgpu-spot -c" 2>/dev/null
bash $D/stop_agent.sh
ps -eo pid,args | grep -E "backend.ai: (manager|storage|web)" | grep -v grep | awk '{print $1}' | xargs -r kill 2>/dev/null; sleep 3
docker ps -q --filter label=ai.backend.kernel-id | xargs -r docker rm -f >/dev/null
(cd $W/bai && docker compose -p $CP -f docker-compose.halfstack-main.yml stop 2>&1 | tail -3)
ps -eo args | grep -E "backend.ai|labgpu-spot" | grep -v grep || echo "no Backend.AI processes left"
docker ps --format '{{.Names}}' | grep -E "labgpu|kernel\." || echo "no E2E containers left"
