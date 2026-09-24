. "$(dirname "$0")/env.sh"
docker exec $DB psql -U postgres -d backend -Atc "$1"
