set -eu
. "$(dirname "$0")/env.sh"
cd $W/bai
# Since 25.8 the manager reads session live stats (the WebUI's per-session usage) from Prometheus,
# which scrapes the agents it discovers through the manager's internal port 18080.
cp configs/prometheus/prometheus.yaml prometheus.yaml
docker compose -p $CP -f docker-compose.halfstack-main.yml up -d --wait \
  backendai-half-db backendai-half-redis backendai-half-etcd backendai-half-prometheus 2>&1 | tail -5
docker ps --filter label=com.docker.compose.project=$CP --format '{{.Names}} {{.Status}} {{.Ports}}'
