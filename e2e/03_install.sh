set -eu
D=$(dirname "$0"); . $D/env.sh
# Python env for this Backend.AI version (upstream pins CPython 3.13).
if [ ! -x $PY ]; then
  [ -x $UV ] || { python3 -m venv --without-pip $(dirname $(dirname $UV)) && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | $(dirname $UV)/python - -q && $(dirname $UV)/pip install -q uv; }
  $UV venv -q -p 3.13 $(dirname $(dirname $PY))
  $UV pip install -q -p $PY -r $W/bai/requirements.txt nvidia-ml-py pytest
fi
# Upstream registers its entry points in pants BUILD files; collect them into an installable shim.
$PY $D/make_shim.py $W/bai/src $W/shim
# Exclude upstream's stock/mock accelerators from the shim: we test our own plugin.
sed -i '/^\[project.entry-points."backendai_accelerator_v21"\]/,/^$/d' $W/shim/pyproject.toml
$UV pip install -q -p $PY $W/shim -e $W/plugin 2>&1 | tail -3
$PY - <<'P'
from importlib.metadata import entry_points
for g in ("backendai_accelerator_v21", "backendai_cli_v10", "backendai_hook_v20"):
    print(g, sorted(e.name for e in entry_points(group=g)))
P
