# Restart manager + agent + gateway + webserver (e.g. after re-staging with 01_stage.sh).
D=$(dirname "$0"); . $D/env.sh; R=$W/run
ps -eo pid,args | grep -E "backend.ai: (manager|web|storage)" | grep -v grep | awk '{print $1}' | xargs -r kill 2>/dev/null
sleep 4
bash $D/20_webui.sh
echo "manager version now: $(curl -s http://127.0.0.1:8081/)"
