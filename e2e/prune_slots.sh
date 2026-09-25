# Keep only the slots this cluster really has in etcd config/resource_slots.
# The WebUI offers every registered accelerator slot and pre-selects the first one, so leftovers
# such as cuda.device show up as an unlimited "GPU" choice in the session launcher.
. "$(dirname "$0")/env.sh"; cd $W/run
KEEP=${KEEP:-"cpu mem pro6000.shares pro6000-spot.device pro5000l.shares a6000.shares rtx4050.shares"}
BAI="$PY -m ai.backend.cli mgr -f manager.toml"
for slot in $($BAI etcd get --prefix config/resource_slots 2>/dev/null | $PY -c 'import json,sys; print(" ".join(json.load(sys.stdin)))'); do
  case " $KEEP " in *" $slot "*) ;; *) $BAI etcd delete "config/resource_slots/$slot" 2>/dev/null && echo "pruned slot $slot" ;; esac
done
