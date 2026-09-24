D=$(dirname "$0"); . $D/env.sh; R=$W/run; cd $R
BAI="$PY -m ai.backend.cli mgr -f manager.toml"
for s in s1-p6000 s2-p6000 s3-a6000 s4-p5000l s5-toobig; do $PY $D/api.py destroy $s 2>&1 | grep -E "^destroy"; done
put() { $BAI etcd put "$1" "$2" 2>/dev/null; }
put config/plugins/accelerator/gpu_slot_4/key rtx4050
put config/plugins/accelerator/gpu_slot_4/model_pattern "*RTX 4050*"
put config/plugins/accelerator/gpu_slot_4/hook_path $R/labgpu/hami/libvgpu.so
put config/resource_slots/rtx4050.shares count
$PY $D/slot_types.py $W/bai/fixtures/manager/example-resource-slot-types.json "rtx4050:RTX 4050:RTX4050" > $R/labgpu/slot-4050.json
$BAI fixture populate $R/labgpu/slot-4050.json 2>&1 | tail -1
$PY -c "import pynvml as n; n.nvmlInit(); h=n.nvmlDeviceGetHandleByIndex(0); print('NVML in WSL:', n.nvmlDeviceGetName(h), n.nvmlDeviceGetUUID(h), n.nvmlDeviceGetMemoryInfo(h).total>>20, 'MiB')"
