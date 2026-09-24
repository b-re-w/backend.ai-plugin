D=$(dirname "$0"); . $D/env.sh; cd $W/run
bash $D/stop_agent.sh; ps -eo pid,args | grep -E "backend.ai" | grep -v grep | cut -c1-100
sed -i 's/\r$//' agent.toml manager.toml alembic.ini
sed -i '/^allocation-order/d' agent.toml
sed -i 's/^\[resource\]$/[resource]\nallocation-order = ["pro6000", "pro5000l", "a6000", "rtx4050", "cuda", "cpu", "mem"]/' agent.toml
grep -n -A2 "^\[resource\]" agent.toml
