"""Resource presets per GPU model (SPEC 1.11), created through the manager GraphQL API."""

import asyncio
import json
import os

os.environ.setdefault("BACKEND_ENDPOINT", "http://127.0.0.1:8081")
os.environ.setdefault("BACKEND_ACCESS_KEY", "AKIAIOSFODNN7EXAMPLE")
os.environ.setdefault("BACKEND_SECRET_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
os.environ.setdefault("BACKEND_ENDPOINT_TYPE", "api")

from ai.backend.client.request import Request  # noqa: E402
from ai.backend.client.session import AsyncSession  # noqa: E402

GiB = 2**30
# name: (cpu, mem GiB, slot, amount). Sized for this 21-core / 15 GiB WSL test node.
PRESETS = {
    "pro6000-half": (4, 6, "pro6000.shares", "0.5"),
    "pro6000-full": (8, 12, "pro6000.shares", "1"),
    "pro5000l-quarter": (2, 3, "pro5000l.shares", "0.25"),
    "pro5000l-full": (4, 6, "pro5000l.shares", "1"),
    "a6000-half": (2, 3, "a6000.shares", "0.5"),
    "a6000-full": (4, 6, "a6000.shares", "1"),
}
# Upstream sample presets for slots this cluster does not have.
OBSOLETE = ["cuda01-small", "cuda02-medium", "cuda03-large"]


async def gql(query, variables=None):
    rqst = Request("POST", "/admin/gql")
    rqst.set_json({"query": query, "variables": variables or {}})
    async with rqst.fetch() as resp:
        data = await resp.json()
    if data.get("errors"):
        raise RuntimeError(data["errors"])
    return data["data"]


async def main():
    async with AsyncSession():
        existing = {p["name"]: p["id"] for p in (await gql("{ resource_presets { id name } }"))["resource_presets"]}
        for name in OBSOLETE:
            if name in existing:
                r = await gql("mutation($id: UUID) { delete_resource_preset(id: $id) { ok msg } }", {"id": existing[name]})
                print("deleted", name, r["delete_resource_preset"])
        for name, (cpu, mem, slot, amount) in PRESETS.items():
            slots = json.dumps({"cpu": str(cpu), "mem": str(mem * GiB), slot: amount})
            if name in existing:
                r = await gql(
                    "mutation($id: UUID, $p: ModifyResourcePresetInput!) { modify_resource_preset(id: $id, props: $p) { ok msg } }",
                    {"id": existing[name], "p": {"resource_slots": slots, "shared_memory": "1g"}},
                )
                print("updated", name, r["modify_resource_preset"])
            else:
                r = await gql(
                    "mutation($n: String!, $p: CreateResourcePresetInput!) { create_resource_preset(name: $n, props: $p) { ok msg } }",
                    {"n": name, "p": {"resource_slots": slots, "shared_memory": "1g"}},
                )
                print("created", name, r["create_resource_preset"])
        for p in (await gql("{ resource_presets { name resource_slots shared_memory } }"))["resource_presets"]:
            print(f"  {p['name']:18} {p['resource_slots']}")


asyncio.run(main())
