"""Send one GraphQL query straight to the manager with keypair (HMAC) auth: gql.py '<query>' [json-vars]"""
import asyncio, json, os, sys
os.environ.setdefault("BACKEND_ENDPOINT", "http://127.0.0.1:8081")
os.environ.setdefault("BACKEND_ACCESS_KEY", "AKIAIOSFODNN7EXAMPLE")
os.environ.setdefault("BACKEND_SECRET_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
os.environ.setdefault("BACKEND_ENDPOINT_TYPE", "api")
from ai.backend.client.request import Request
from ai.backend.client.session import AsyncSession

async def main():
    async with AsyncSession():
        rqst = Request("POST", "/admin/gql")
        rqst.set_json({"query": sys.argv[1], "variables": json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}})
        async with rqst.fetch() as resp:
            print(json.dumps(await resp.json())[:1500])
asyncio.run(main())
