set -eu
D=$(dirname "$0"); . $D/env.sh
R=$W/run; mkdir -p $R/ipc $R/var $R/scratches $R/vfroot/local $R/labgpu
cd $R
ln -sfn $W/bai/fixtures fixtures   # manager reads fixtures/manager/manager.key_secret relative to cwd
BAI="$PY -m ai.backend.cli"
# --- manager ---
cp $W/bai/configs/manager/halfstack.toml manager.toml
sed -i "s/num-proc = .*/num-proc = 1/" manager.toml
sed -i "s@\(# \)\{0,1\}ipc-base-path = .*@ipc-base-path = \"$R/ipc\"@" manager.toml
cp $W/bai/configs/manager/halfstack.alembic.ini alembic.ini
sed -i "s@script_location = .*@script_location = $W/bai/src/ai/backend/manager/models/alembic@" alembic.ini
# --- agent ---
cp $W/bai/configs/agent/halfstack.toml agent.toml
sed -i "s@\(# \)\{0,1\}ipc-base-path = .*@ipc-base-path = \"$R/ipc\"@" agent.toml
sed -i "s@\(# \)\{0,1\}var-base-path = .*@var-base-path = \"$R/var\"@" agent.toml
sed -i "s@\(# \)\{0,1\}mount-path = .*@mount-path = \"$R/vfroot/local\"@" agent.toml
sed -i "s@scratch-root = .*@scratch-root = \"$R/scratches\"@" agent.toml
sed -i "s@^# block-compute-plugins =.*@block-compute-plugins = [\"labgpu.accelerator.cuda_frac\"]@" agent.toml
sed -i "s@^# allow-compute-plugins =.*@allow-compute-plugins = [\"labgpu.accelerator\"]@" agent.toml   # upstream accelerators are scanned from BUILD files otherwise
grep -n "block-compute\|ipc-base\|var-base\|mount-path\|scratch-root" agent.toml
# --- etcd ---
$BAI mgr -f manager.toml etcd put config/redis/addr "127.0.0.1:8110"
$BAI mgr -f manager.toml etcd put-json config/redis/redis_helper_config $W/bai/configs/manager/sample.etcd.redis-helper.json
cp $W/bai/configs/manager/sample.etcd.volumes.json volumes.json
$BAI mgr -f manager.toml etcd put-json volumes volumes.json
echo "--- etcd ok"
# --- DB ---
$BAI mgr -f manager.toml schema oneshot 2>&1 | tail -3
for fx in example-container-registries-local example-users example-keypairs example-resource-slot-types example-resource-presets; do
  $BAI mgr -f manager.toml fixture populate $W/bai/fixtures/manager/$fx.json 2>&1 | tail -1
done
