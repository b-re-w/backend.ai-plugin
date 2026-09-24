#!/bin/sh
# Per-model GPU slots for the lab cluster (SPEC 1.11). Run once on the manager host.
#
#   Primary:   PRO 5000 72GB (0), PRO 6000 x2 (1, 2), A6000 (3)
#   Secondary: PRO 5000 48GB x5 (0-4)
#
# The etcd config is shared by every agent; each plugin only takes GPUs matching its pattern,
# so the same settings work on both servers. Then, on every GPU node's agent.toml:
#
#   [agent]
#   allow-compute-plugins = ["labgpu.accelerator"]
#   block-compute-plugins = ["labgpu.accelerator.cuda_frac"]
#   [resource]
#   allocation-order = ["pro6000", "pro5000l", "pro5000", "a6000", "cpu", "mem"]
set -eu
BAI=${BAI:-backend.ai}
HOOK=${HOOK:-/opt/labgpu/lib/libvgpu.so}
P=config/plugins/accelerator

# display_unit is what the WebUI session launcher shows in its accelerator-type selector,
# so it must differ between slots (the two PRO 5000 variants share a model name).
slot() {  # slot <entry> <key> <pattern> <display name> <display unit> [min_memory] [max_memory]
  $BAI mgr etcd put "$P/$1/key" "$2"
  $BAI mgr etcd put "$P/$1/model_pattern" "$3"
  $BAI mgr etcd put "$P/$1/display_name" "$4"
  $BAI mgr etcd put "$P/$1/display_unit" "$5"
  $BAI mgr etcd put "$P/$1/hook_path" "$HOOK"
  [ -n "${6:-}" ] && $BAI mgr etcd put "$P/$1/min_memory" "$6"
  [ -n "${7:-}" ] && $BAI mgr etcd put "$P/$1/max_memory" "$7"
  $BAI mgr etcd put "config/resource_slots/$2.shares" count
}

slot gpu_slot_1 pro6000  "*PRO 6000*" "PRO 6000"      PRO6000
slot gpu_slot_2 pro5000l "*PRO 5000*" "PRO 5000 72GB" PRO5000-72 60g
slot gpu_slot_3 pro5000  "*PRO 5000*" "PRO 5000 48GB" PRO5000-48 "" 60g
slot gpu_slot_4 a6000    "*A6000*"    "A6000"         A6000

$BAI mgr etcd get --prefix "$P"
