#!/bin/sh
# Per-model GPU slots for the lab cluster (SPEC 1.11). Run once on the manager host.
#
#   Primary:   PRO 5000 72GB (0), PRO 6000 x2 (1, 2), A6000 (3)
#   Secondary: PRO 5000 48GB x4 (0-3)
#
# The etcd config is shared by every agent; each plugin only takes GPUs matching its pattern,
# so the same settings work on both servers. Then, on every GPU node's agent.toml:
#
#   [agent]
#   allow-compute-plugins = ["labgpu.accelerator"]
#   block-compute-plugins = ["labgpu.accelerator.cuda_frac"]
#   [resource]
#   allocation-order = ["cuda-pro6000", "cuda-pro6000-spot", "cuda-pro5000-72", "cuda-pro5000-72-spot", "cuda-pro5000", "cuda-pro5000-spot",
#                       "cuda-a6000", "cuda-a6000-spot", "cpu", "mem"]
#
# Kernel images need no relabeling: the keys start with "cuda" (SPEC 1.11).
#
# Backend.AI 26.x also needs every slot in the manager DB table resource_slot_types, or agent
# heartbeats fail with a foreign-key error; the end of this script populates it (fixture).
set -eu
BAI=${BAI:-backend.ai}
HOOK=${HOOK:-}   # empty: the plugins use <plugin checkout>/.venv/lib/libvgpu.so
PY=${PY:-python3}
P=config/plugins/accelerator
HERE=$(cd "$(dirname "$0")" && pwd)
SLOT_TYPES=""   # SLOT:DISPLAY_NAME:DISPLAY_UNIT, one per line, for resource_slot_types

# display_unit is what the WebUI session launcher shows in its accelerator-type selector,
# so it must differ between slots (the two PRO 5000 variants share a model name).
slot() {  # slot <entry> <key> <pattern> <display name> <display unit> [min_memory] [max_memory]
  $BAI mgr etcd put "$P/$1/key" "$2"
  $BAI mgr etcd put "$P/$1/model_pattern" "$3"
  $BAI mgr etcd put "$P/$1/display_name" "$4"
  $BAI mgr etcd put "$P/$1/display_unit" "$5"
  if [ -n "$HOOK" ]; then $BAI mgr etcd put "$P/$1/hook_path" "$HOOK"; else $BAI mgr etcd delete "$P/$1/hook_path" || true; fi
  [ -n "${6:-}" ] && $BAI mgr etcd put "$P/$1/min_memory" "$6"
  [ -n "${7:-}" ] && $BAI mgr etcd put "$P/$1/max_memory" "$7"
  $BAI mgr etcd put "config/resource_slots/$2.shares" count
  SLOT_TYPES="$SLOT_TYPES
$2.shares:$4:$5"
}

slot gpu_slot_1 cuda-pro6000  "*PRO 6000*" "PRO 6000"      PRO6000
slot gpu_slot_2 cuda-pro5000-72 "*PRO 5000*" "PRO 5000 72GB" PRO5000-72 60g
slot gpu_slot_3 cuda-pro5000  "*PRO 5000*" "PRO 5000"      PRO5000    "" 60g
slot gpu_slot_4 cuda-a6000    "*A6000*"    "A6000"         A6000

# Spot launch mode (SPEC 2.12): one gpu_spot_N per model, same GPU selection as its owner slot.
spot() {  # spot <entry> <key> <pattern> <display unit> <display name> [min_memory] [max_memory]
  $BAI mgr etcd put "$P/$1/key" "$2"
  $BAI mgr etcd put "$P/$1/model_pattern" "$3"
  $BAI mgr etcd put "$P/$1/display_unit" "$4"
  $BAI mgr etcd put "$P/$1/display_name" "$5"
  [ -n "${6:-}" ] && $BAI mgr etcd put "$P/$1/min_memory" "$6"
  [ -n "${7:-}" ] && $BAI mgr etcd put "$P/$1/max_memory" "$7"
  $BAI mgr etcd put "config/resource_slots/$2.device" count
  SLOT_TYPES="$SLOT_TYPES
$2.device:$5:$4"
}

spot gpu_spot_1 cuda-pro6000-spot    "*PRO 6000*" PRO6000-SPOT    "PRO 6000 Spot"
spot gpu_spot_2 cuda-pro5000-72-spot "*PRO 5000*" PRO5000-72-SPOT "PRO 5000 72GB Spot" 60g
spot gpu_spot_3 cuda-pro5000-spot    "*PRO 5000*" PRO5000-SPOT    "PRO 5000 Spot" "" 60g
spot gpu_spot_4 cuda-a6000-spot      "*A6000*"    A6000-SPOT      "A6000 Spot"

$BAI mgr etcd get --prefix "$P"

# resource_slot_types rows (existing rows are skipped by fixture populate).
fixture=$(mktemp --suffix=.json)
old_ifs=$IFS; IFS="$(printf '
_')"; IFS=${IFS%_}; set -f
set -- $SLOT_TYPES
set +f; IFS=$old_ifs
"$PY" "$HERE/../scripts/slot_types_fixture.py" "$@" > "$fixture"
$BAI mgr fixture populate "$fixture"
rm -f "$fixture"
