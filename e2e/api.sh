D=$(dirname "$0"); . $D/env.sh
$PY $D/api.py "$@" 2>&1 | grep -v "RPC authentication\|DeprecationWarning\|warnings.warn"
