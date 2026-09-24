ps -eo pid,args | grep -E "backend.ai: agent|ag start-server|backend.ai: kernel-runner|agent.watcher" | grep -v grep | awk '{print $1}' | xargs -r kill 2>/dev/null
for i in $(seq 1 15); do ps -eo args | grep -qE "backend.ai: agent|ag start-server" | grep -v grep || break; sleep 1; done
ps -eo pid,args | grep -E "backend.ai: agent|ag start-server" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null; true
