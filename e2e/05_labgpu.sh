set -eu
D=$(dirname "$0"); . $D/env.sh; R=$W/run; cd $R
BAI="$PY -m ai.backend.cli mgr -f manager.toml"
# Fake "Primary server" layout (SPEC 1.12). The file is re-read live.
cat > $R/labgpu/gpus.json <<'J'
{"driver": "fake-primary",
 "gpus": [
  {"uuid": "GPU-fake-p5000-72g", "name": "NVIDIA RTX PRO 5000 Blackwell", "memory": "72g"},
  {"uuid": "GPU-fake-p6000-1",   "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "memory": "96g"},
  {"uuid": "GPU-fake-p6000-2",   "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "memory": "96g"},
  {"uuid": "GPU-fake-a6000",     "name": "NVIDIA RTX A6000", "memory": "48g"}
 ]}
J
: > $R/labgpu/libvgpu.so   # placeholder hook: fake mode only checks presence
put() { $BAI etcd put "$1" "$2" 2>/dev/null; }
put config/plugins/accelerator/gpu_slot_1/key pro6000
put config/plugins/accelerator/gpu_slot_1/model_pattern "*PRO 6000*"
put config/plugins/accelerator/gpu_slot_1/hook_path $R/labgpu/libvgpu.so
put config/plugins/accelerator/gpu_slot_2/key pro5000l
put config/plugins/accelerator/gpu_slot_2/model_pattern "*PRO 5000*"
put config/plugins/accelerator/gpu_slot_2/min_memory 60g
put config/plugins/accelerator/gpu_slot_2/display_name "PRO 5000 72GB"
put config/plugins/accelerator/gpu_slot_2/hook_path $R/labgpu/libvgpu.so
put config/plugins/accelerator/gpu_slot_3/key a6000
put config/plugins/accelerator/gpu_slot_3/model_pattern "*A6000*"
put config/plugins/accelerator/gpu_slot_3/hook_path $R/labgpu/libvgpu.so
# gpu_slot_4 deliberately left unconfigured: it must be skipped.
for s in pro6000 pro5000l a6000; do put config/resource_slots/$s.shares count; done
$BAI etcd get --prefix config/plugins 2>/dev/null | head -20
# Slot types table (26.x keeps display metadata in the DB too).
$PY $D/slot_types.py $W/bai/fixtures/manager/example-resource-slot-types.json "pro6000:PRO 6000:PRO6000" "pro5000l:PRO 5000 72GB:PRO5000" "a6000:A6000:A6000" > $R/labgpu/slot-types.json
$BAI fixture populate $R/labgpu/slot-types.json 2>&1 | tail -1
# Minimal Backend.AI kernel image.
mkdir -p $W/image && cat > $W/image/Dockerfile <<'DF'
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
LABEL ai.backend.kernelspec="1" \
      ai.backend.features="batch query uid-match" \
      ai.backend.base-distro="ubuntu22.04" \
      ai.backend.runtime-type="python" \
      ai.backend.runtime-path="/usr/bin/python3" \
      ai.backend.resource.min.cpu="1" \
      ai.backend.resource.min.mem="256m" \
      ai.backend.accelerators="cuda,pro6000,pro5000l,a6000,rtx4050" \
      ai.backend.service-ports=""
DF
# The manager names local-registry images "local/<repo>", while the agent reports Docker tags as-is:
# without the second tag the image never counts as installed and the WebUI launcher hides it.
docker build -q -t stable/labgpu-test:1.0 -t local/stable/labgpu-test:1.0 $W/image | tail -1
# The launcher only offers images from registries the domain allows.
docker exec $DB psql -U postgres -d backend -qAtc "update domains set allowed_docker_registries = array_append(allowed_docker_registries, 'local') where name='default' and not ('local' = any(allowed_docker_registries));"
$BAI image rescan local 2>&1 | tail -3
docker exec $DB psql -U postgres -d backend -Atc "select name, image, tag, accelerators from images;"
