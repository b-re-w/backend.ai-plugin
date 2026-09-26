"""
Write a `resource_slot_types` fixture for labgpu slots (SPEC 1.11).

Backend.AI 26.x keeps slot types in the manager DB, and `agent_resources.slot_name` references that
table: an agent reporting a slot missing there makes every heartbeat fail with a foreign-key error.
The manager has no API to add rows; `backend.ai mgr fixture populate <file>` is the supported way.

usage: slot_types_fixture.py SLOT:DISPLAY_NAME:DISPLAY_UNIT[:ROUND] ... > slot-types.json
  e.g. cuda-pro6000.shares:"PRO 6000":PRO6000:2 cuda-pro6000-spot.device:"PRO 6000 Spot":PRO6000-SPOT:0
"""

import json
import sys


def row(spec: str, rank: int) -> dict:
    slot, name, unit, *rest = spec.split(":")
    fractional = slot.endswith(".shares")
    return {
        "slot_name": slot,
        "slot_type": "count",
        "required": False,
        "display_name": name,
        "description": f"{'Fractional ' if fractional else ''}{name} (labgpu)",
        "display_unit": unit,
        "display_icon": "gpu1",
        "number_format": {"binary": False, "round_length": int(rest[0]) if rest else (2 if fractional else 0)},
        "rank": rank,
    }


if __name__ == "__main__":
    specs = sys.argv[1:]
    if not specs:
        sys.exit(__doc__)
    print(json.dumps({"resource_slot_types": [row(s, 40 - i) for i, s in enumerate(specs)]}, indent=1))
