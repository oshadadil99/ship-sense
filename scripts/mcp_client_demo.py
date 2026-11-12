"""A tiny MCP CLIENT: connects to our MCP server over HTTP, lists tools, calls a few.
Run (server must be running):  python -m src.mcp_server --http   then   python -m scripts.mcp_client_demo"""
import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = "http://127.0.0.1:8002/mcp"


async def main() -> None:
    async with streamable_http_client(URL) as (read, write, *_):
        async with ClientSession(read, write) as session:
            await session.initialize()  # handshake: client and server agree on capabilities

            listed = await session.list_tools()  # DISCOVERY: the client learns what we offer
            print("TOOLS:")
            for t in listed.tools:
                print(f"  - {t.name}: {t.description}")

            for name, args in [("get_shipment", {"identifier": "OCLU1023757"}),
                               ("list_delayed_shipments", {"port": "Singapore", "limit": 2}),
                               ("get_shipment", {"identifier": "hello"})]:
                result = await session.call_tool(name, args)
                print(f"\nCALL {name}({args}):\n{result.content[0].text[:400]}")

            res = await session.read_resource("shipsense://kpi/carriers")
            print("\nRESOURCE shipsense://kpi/carriers:\n", json.dumps(json.loads(res.contents[0].text), indent=1)[:300])


if __name__ == "__main__":
    asyncio.run(main())
