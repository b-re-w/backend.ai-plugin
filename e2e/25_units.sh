D=$(dirname "$0"); . $D/env.sh
bash $D/08_agent.sh 35 | sed 's/\x1b\[[0-9;]*m//g' | grep "Resource slots" | cut -c1-120
docker exec $DB psql -U postgres -d backend -qAtc "
update resource_slot_types set display_unit='PRO6000' where slot_name='pro6000.shares';
update resource_slot_types set display_unit='PRO5000' where slot_name='pro5000l.shares';
update resource_slot_types set display_unit='A6000'   where slot_name='a6000.shares';
update resource_slot_types set display_unit='RTX4050' where slot_name='rtx4050.shares';
select slot_name, display_name, display_unit from resource_slot_types where slot_name like '%.shares' and slot_name <> 'cuda.shares';"
