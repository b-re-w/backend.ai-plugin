# Shared settings for the E2E scripts.
#
# BAI_REF picks the Backend.AI release (git tag) under test; its sources are exported from
# backend.ai/.git with `git archive`, so the checkout in backend.ai/ is never touched.
# Each version gets its own work dir, Python env and docker compose project so they never mix.
BAI_REF=${BAI_REF:-26.8.3}
# This script's folder (absolute, so scripts may cd freely) and the workspace root holding
# backend.ai/ and backend.ai-plugin/.
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$(cd "$D/../.." && pwd)
W=$HOME/labgpu-e2e-$BAI_REF
CP=labgpu-e2e-$(echo "$BAI_REF" | tr -c 'a-z0-9\n' '-')
PY=$HOME/.cache/labgpu-py313-$BAI_REF/bin/python
DB=$CP-backendai-half-db-1
export DOCKER_CONFIG=$W/.docker
mkdir -p $DOCKER_CONFIG; [ -f $DOCKER_CONFIG/config.json ] || echo '{}' > $DOCKER_CONFIG/config.json
UV=$HOME/.cache/labgpu-boot/bin/uv
export PYTHONPATH=$W/bai/src
