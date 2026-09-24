D=$(dirname "$0"); . $D/env.sh
rm -rf $W/plugin/src $W/plugin/tests && cp -r $SRC/backend.ai-plugin/src $SRC/backend.ai-plugin/tests $W/plugin/
find $W/plugin -name __pycache__ -prune -exec rm -rf {} +
cd $W/plugin && PYTHONPATH=src:$W/bai/src $PY -m pytest -q -p no:cacheprovider -W ignore 2>&1 | tail -2
$PY $D/api.py destroy owner-p6000-a owner-p6000-b owner-a6000 owner-p5000 too-big 2>&1 | grep -E "^destroy"
