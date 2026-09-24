#!/usr/bin/env bash
# One-off experiment on a native GPU node (not part of Backend.AI): can cuda-checkpoint pause a
# running PyTorch process, free its GPU memory, and resume it on another GPU of the same model?
#
#   e2e/node/setup_venv.sh            # once: <repo>/.venv with torch and cuda-checkpoint
#   e2e/node/cc_migrate.sh --from 0 --to 1
#
# Needs driver >= 580 and two idle GPUs. Nothing is installed or changed; the training process is
# killed on exit. Results go to stdout and to $out (default <repo>/.venv/cc-result).
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
venv=$(cd "$here/../.." && pwd)/.venv
cc=$venv/bin/cuda-checkpoint
python=$venv/bin/python
from=0
to=1
hold_gb=8
out=$venv/cc-result
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cc) cc="$2"; shift 2 ;;
    --python) python="$2"; shift 2 ;;
    --from) from="$2"; shift 2 ;;
    --to) to="$2"; shift 2 ;;
    --hold-gb) hold_gb="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$out"
log="$out/train.log"
note() { echo "[$(date +%T)] $*" | tee -a "$out/summary.txt"; }
apps() { nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader; }
last_step() { grep -o '^step [0-9]*' "$log" | tail -1 | cut -d' ' -f2; }
timed() {  # timed <label> <cmd...>: run, log duration and output, fail loudly
  local label=$1 t0 t1 rc=0
  shift
  t0=$(date +%s.%N)
  "$@" >>"$out/cc.log" 2>&1 || rc=$?
  t1=$(date +%s.%N)
  note "$label: rc=$rc $(awk "BEGIN{printf \"%.2f\", $t1-$t0}")s"
  return $rc
}

mapfile -t uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
src=${uuids[$from]}
dst=${uuids[$to]}
note "driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
note "from GPU $from $src"
note "to   GPU $to $dst"
[[ "$(nvidia-smi -i "$from" --query-gpu=name --format=csv,noheader)" == \
   "$(nvidia-smi -i "$to" --query-gpu=name --format=csv,noheader)" ]] || note "WARNING: different GPU models"

busy=$(apps | grep -e "$src" -e "$dst" || true)
if [[ -n "$busy" ]]; then
  note "ABORT: the GPUs are in use:"; echo "$busy" | tee -a "$out/summary.txt"; exit 1
fi
"$cc" --help >/dev/null

pid=
cleanup() { [[ -n "$pid" ]] && kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null; true; }
trap cleanup EXIT

# Every GPU must stay visible to the process so it can be restored onto another one.
env -u CUDA_VISIBLE_DEVICES -u CUDA_DEVICE_ORDER \
  "$python" "$here/cc_train.py" --uuid "$src" --hold-gb "$hold_gb" >"$log" 2>&1 &
pid=$!
note "training pid $pid"
sleep 25
kill -0 "$pid" 2>/dev/null || { note "FAIL: training exited early"; tail "$log"; exit 1; }
note "before: $(apps | grep "^$pid," || echo 'not on any GPU')"

# 1) pause and resume on the same GPU
timed toggle "$cc" --toggle --pid "$pid"
note "suspended: $(apps | grep "^$pid," || echo 'no GPU memory held')"
s0=$(last_step); sleep 5; s1=$(last_step)
note "steps while suspended: $s0 -> $s1"
timed toggle "$cc" --toggle --pid "$pid"
sleep 10
note "resumed: $(apps | grep "^$pid," || echo 'not on any GPU'), step $(last_step)"

# 2) move to the other GPU
map=()
for u in "${uuids[@]}"; do
  case "$u" in
    "$src") map+=("$src=$dst") ;;
    "$dst") map+=("$dst=$src") ;;
    *) map+=("$u=$u") ;;
  esac
done
device_map=$(IFS=,; echo "${map[*]}")
timed lock "$cc" --action lock --pid "$pid"
timed checkpoint "$cc" --action checkpoint --pid "$pid"
note "checkpointed: $(apps | grep "^$pid," || echo 'no GPU memory held')"
s0=$(last_step); sleep 10; s1=$(last_step)
note "steps while checkpointed: $s0 -> $s1"
timed restore "$cc" --action restore --pid "$pid" --device-map "$device_map"
timed unlock "$cc" --action unlock --pid "$pid"
sleep 30
after=$(apps | grep "^$pid," || true)
note "after: ${after:-not on any GPU}"
kill -0 "$pid" 2>/dev/null || note "training process died"
note "last lines of the training log:"
tail -5 "$log" | tee -a "$out/summary.txt"
grep -iE "error|exception|traceback" "$log" | tee -a "$out/summary.txt" || true

if [[ "$after" == *"$dst"* ]] && kill -0 "$pid" 2>/dev/null && (( $(last_step) > s1 )); then
  note "RESULT: PASS (moved to GPU $to and kept training)"
else
  note "RESULT: FAIL (see $out)"
fi
