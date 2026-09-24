D=$(dirname "$0"); . $D/env.sh; cd $W/run
sed -i 's/^port-range = .*/port-range = [41000, 41900]/' agent.toml; grep -n "port-range" agent.toml
for s in o1-p6000 o2-p6000 o3-a6000 o4-p5000l o5-toobig; do $PY $D/api.py destroy $s >/dev/null 2>&1; done
sed -i 's/o1-p6000/s1-p6000/; s/o2-p6000/s2-p6000/; s/o3-a6000/s3-a6000/; s/o4-p5000l/s4-p5000l/; s/o5-toobig/s5-toobig/' $D/09_sessions.sh
