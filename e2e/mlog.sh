# mlog.sh [pattern] [manager|agent|webserver|storage-proxy] [lines]: grep a server log without colour codes.
. "$(dirname "$0")/env.sh"
sed 's/\x1b\[[0-9;]*m//g' $W/run/${2:-manager}.log | grep -iE "${1:-sched|predicate|pending|error}" | cut -c1-280 | tail -${3:-25}
