"""Write a resource_slot_types fixture for our slots, matching the staged Backend.AI version's schema.

usage: slot_types.py <upstream example-resource-slot-types.json> key:display-name:unit ... > out.json
(26.8 has no uuid column; 26.9 does. Follow whatever the upstream example rows carry.)
"""

import json
import sys
import uuid

example = json.load(open(sys.argv[1]))["resource_slot_types"][0]
rows = []
for rank, spec in enumerate(sys.argv[2:]):
    key, name, unit = spec.split(":")
    row = {
        "slot_name": f"{key}.shares", "slot_type": "count", "required": False,
        "display_name": name, "description": f"Fractional {name}", "display_unit": unit,
        "display_icon": "gpu1", "number_format": {"binary": False, "round_length": 2}, "rank": 40 - rank,
    }
    if "uuid" in example:
        row["uuid"] = str(uuid.uuid5(uuid.NAMESPACE_URL, key))
    if "enabled" in example:
        row["enabled"] = True
    rows.append(row)
print(json.dumps({"resource_slot_types": rows}))
