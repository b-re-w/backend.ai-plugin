"""E2E driver: talks to the running manager through the Backend.AI client SDK."""

import asyncio
import json
import os
import sys

os.environ.setdefault("BACKEND_ENDPOINT", "http://127.0.0.1:8081")
os.environ.setdefault("BACKEND_ACCESS_KEY", "AKIAIOSFODNN7EXAMPLE")
os.environ.setdefault("BACKEND_SECRET_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
os.environ.setdefault("BACKEND_ENDPOINT_TYPE", "api")

from ai.backend.client.request import Request  # noqa: E402
from ai.backend.client.session import AsyncSession  # noqa: E402

IMAGE = "local/stable/labgpu-test:1.0"


async def slots(api):
    rqst = Request("GET", "/config/resource-slots/details")
    async with rqst.fetch() as resp:
        details = await resp.json()
    print("resource-slots/details:")
    for k, v in details.items():
        print(f"  {k:18} {v.get('human_readable_name')!r:28} unit={v.get('display_unit')}")
    rqst = Request("POST", "/admin/gql")
    rqst.set_json({"query": "{ agent_list(limit: 10, offset: 0, status: \"ALIVE\") { items { id available_slots occupied_slots compute_plugins } } }"})
    async with rqst.fetch() as resp:
        data = await resp.json()
    for a in data["data"]["agent_list"]["items"]:
        print("agent", a["id"])
        print("  available:", a["available_slots"])
        print("  occupied :", a["occupied_slots"])


async def rescan(api):
    rqst = Request("POST", "/admin/gql")
    rqst.set_json({"query": 'mutation { rescan_images(registry: "local") { ok msg task_id } }'})
    async with rqst.fetch() as resp:
        print(await resp.json())
    await asyncio.sleep(8)
    rqst = Request("POST", "/admin/gql")
    rqst.set_json({"query": "{ images { name tag registry installed supported_accelerators } }"})
    async with rqst.fetch() as resp:
        print(json.dumps((await resp.json()).get("data"), indent=1)[:800])


async def start(api, name, slot, amount, cmd="sleep infinity"):
    s = await api.ComputeSession.get_or_create(
        IMAGE,
        name=name,
        type_="batch",
        startup_command=cmd,
        resources={"cpu": 1, "mem": "512m", slot: amount},
        enqueue_only=False,
        max_wait=150,
    )
    print(f"{name}: status={s.status} id={s.id}")


async def info(api, name):
    s = api.ComputeSession(name)
    d = await s.detail()
    print(json.dumps(d, default=str, indent=1)[:1500])


async def destroy(api, name):
    try:
        await api.ComputeSession(name).destroy(forced=True)
        print("destroyed", name)
    except Exception as e:  # noqa: BLE001
        print("destroy", name, "->", e)


async def main():
    cmd, *args = sys.argv[1:]
    async with AsyncSession() as api:
        match cmd:
            case "slots":
                await slots(api)
            case "rescan":
                await rescan(api)
            case "start":
                await start(api, args[0], args[1], args[2], *args[3:])
            case "info":
                await info(api, args[0])
            case "destroy":
                for n in args:
                    await destroy(api, n)


asyncio.run(main())
