# GraphQL federation gateway (Hive Gateway on :4000) the new WebUI needs for its v2 queries.
D=$(dirname "$0"); . $D/env.sh; R=$W/run
cd $W/bai

cp $W/bai/configs/graphql/gateway.config.ts gateway.config.ts
# Upstream's dev manager listens on 8091; ours on 8081 (reached via host.docker.internal on Docker Desktop).
sed "s@http://host.docker.internal:8091@http://host.docker.internal:8081@g" \
  $W/bai/docs/manager/graphql-reference/supergraph.graphql > supergraph.graphql
sed -i 's@host.docker.internal:8091@host.docker.internal:8081@g' gateway.config.ts   # transportEntries
sed -i 's/\r$//' gateway.config.ts supergraph.graphql
grep -n "join__graph(name" supergraph.graphql | cut -c1-140
docker compose -p $CP -f docker-compose.halfstack-main.yml up -d --force-recreate backendai-half-apollo-router 2>&1 | tail -2
for i in $(seq 1 30); do curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:4000/graphql?query=%7B__typename%7D | grep -q 200 && break; sleep 2; done
echo "gateway: HTTP $(curl -s -o /dev/null -w '%{http_code}' 'http://127.0.0.1:4000/graphql?query=%7B__typename%7D')"
cd $R
sed -i '/^\[apollo-router\]/,/^\[/ s/^enabled = false/enabled = true/' webserver.conf
grep -A2 "^\[apollo-router\]" webserver.conf
ps -eo pid,args | grep -E "backend.ai: web" | grep -v grep | awk '{print $1}' | xargs -r kill 2>/dev/null; sleep 2
nohup setsid $PY -m ai.backend.cli web start-server -f webserver.conf > webserver.log 2>&1 < /dev/null &
for i in $(seq 1 30); do curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/ | grep -q 200 && break; sleep 2; done
echo "webserver: HTTP $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/)"
