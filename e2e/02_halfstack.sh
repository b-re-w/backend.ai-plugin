set -eu
. "$(dirname "$0")/env.sh"
cd $W/bai
docker compose -p $CP -f docker-compose.halfstack-main.yml up -d --wait backendai-half-db backendai-half-redis backendai-half-etcd 2>&1 | tail -4
docker ps --filter label=com.docker.compose.project=$CP --format '{{.Names}} {{.Status}} {{.Ports}}'
