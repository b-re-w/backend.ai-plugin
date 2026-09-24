# Bring the whole E2E stack back up and serve the WebUI at http://localhost:8090 (fake Primary GPUs).
D=$(dirname "$0"); . $D/env.sh; R=$W/run
(cd $W/bai && docker compose -p $CP -f docker-compose.halfstack-main.yml up -d --wait \
  backendai-half-db backendai-half-redis backendai-half-etcd 2>&1 | tail -1)
findmnt -no PROPAGATION / | grep -q shared || echo "WARNING: run 'wsl -d Ubuntu-24.04 -u root mount --make-rshared /' first"
cd $R
if ! curl -s -o /dev/null http://127.0.0.1:8081/; then
  nohup setsid $PY -m ai.backend.cli mgr start-server -f manager.toml > manager.log 2>&1 < /dev/null &
  for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:8081/ && break; sleep 2; done
fi
echo "manager: $(curl -s http://127.0.0.1:8081/ | head -c 80)"
bash $D/prune_slots.sh
bash $D/08_agent.sh 35 | sed 's/\x1b\[[0-9;]*m//g' | grep "Resource slots" | cut -c1-200

cp $W/bai/configs/webserver/halfstack.conf webserver.conf
sed -i 's/\r$//' webserver.conf
sed -i 's@https://api.backend.ai@http://127.0.0.1:8081@' webserver.conf
sed -i 's@addr = "localhost:8111"@addr = "localhost:8110"@' webserver.conf
sed -i 's@^#\{0,1\}static_path = .*@static_path = "'$W'/bai/src/ai/backend/web/static"@' webserver.conf
grep -nE "^endpoint|^addr|static_path|^port" webserver.conf | head -8
ps -eo pid,args | grep -E "backend.ai: web|web start-server" | grep -v grep | awk '{print $1}' | xargs -r kill 2>/dev/null
nohup setsid $PY -m ai.backend.cli web start-server -f webserver.conf > webserver.log 2>&1 < /dev/null &
for i in $(seq 1 30); do curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/ | grep -q 200 && break; sleep 2; done
echo "webserver: HTTP $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/)"
# The new WebUI needs the GraphQL federation gateway for its v2 queries.
bash $D/21_gateway.sh
# Storage proxy for the folder (vfolder) pages.
bash $D/23_storage.sh
