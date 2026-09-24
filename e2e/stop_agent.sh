# Stop the agent gracefully. It saves its kernel registry on shutdown; killing it early leaves
# an empty registry file and the next start fails with KernelRegistryLoadError.
agent_pids() { ps -eo pid,args | grep -E "backend.ai: agent|ag start-server" | grep -v grep | awk '{print $1}'; }
agent_pids | xargs -r kill 2>/dev/null
for i in $(seq 1 60); do [ -z "$(agent_pids)" ] && break; sleep 1; done
if [ -n "$(agent_pids)" ]; then
  echo "agent did not stop within 60s; killing it" >&2
  agent_pids | xargs -r kill -9 2>/dev/null
fi
true
