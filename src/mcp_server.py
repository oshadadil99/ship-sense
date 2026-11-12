"""MCP server: the same business tools, through a standard plug any AI client can use
(Copilot in VS Code, Claude Desktop, Foundry agents...). Uses MCP Python SDK 2.x (MCPServer).
Run:  python -m src.mcp_server          (stdio  - the client starts this process)
      python -m src.mcp_server --http   (streamable HTTP on http://127.0.0.1:8002/mcp)"""
import contextlib
import sys

from mcp.server.mcpserver import MCPServer

from src import lakehouse, tools
from src.scraper import track_containers

mcp = MCPServer(name="shipsense", instructions="Freight-forwarding tools: shipments, delays, carrier KPIs, policies.")


def _quiet(fn, *args, **kwargs):
    # stdio transport uses stdout for protocol messages -> any print() would corrupt it. Send prints to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        return fn(*args, **kwargs)


# The docstring + type hints become the tool description/schema the AI client sees.
@mcp.tool()
def get_shipment(identifier: str) -> dict:
    """Get one shipment by container number (e.g. OCLU1023757) or shipment id (e.g. SHP-2026003)."""
    return _quiet(tools.get_shipment, identifier)


@mcp.tool()
def list_delayed_shipments(port: str | None = None, carrier: str | None = None, limit: int = 10) -> dict:
    """List shipments delayed more than 48h, worst first. Optional filters: destination port, carrier."""
    return _quiet(tools.list_delayed_shipments, port, carrier, limit)


@mcp.tool()
def carrier_performance() -> dict:
    """On-time % and average delay hours per carrier, worst first."""
    return _quiet(tools.carrier_performance)


@mcp.tool()
def search_policies(question: str) -> dict:
    """Answer company policy questions (demurrage, customs, dangerous goods, claims, escalation) with citations."""
    return _quiet(tools.search_policies, question)


@mcp.tool()
async def live_track_container(container_no: str) -> dict:
    """Check the carrier portal NOW for a container's live ETA and events, then refresh the lakehouse."""
    c = (container_no or "").strip().upper()
    if not tools.CONTAINER_RE.match(c):
        return {"error": f"'{container_no}' is not a valid container number (4 letters + 7 digits)"}
    try:
        # async tool: we are already inside the server's event loop, so await directly (no asyncio.run)
        result = (await track_containers([c]))[0]
    except Exception as e:
        return {"error": f"carrier portal unavailable ({type(e).__name__})"}
    if "error" not in result:
        _quiet(lakehouse.ingest_tracking, [result])
        _quiet(lakehouse.build_silver)
        _quiet(lakehouse.build_gold)
    return result


# A resource = read-only data the client can load as context (no "action").
@mcp.resource("shipsense://kpi/carriers")
def carrier_kpis() -> str:
    """Carrier KPI table as JSON."""
    return lakehouse.read("gold", "kpi_by_carrier").to_json(orient="records")


if __name__ == "__main__":
    if "--http" in sys.argv:
        mcp.run(transport="streamable-http", port=8002)  # network service, many clients
    else:
        mcp.run()  # stdio: one client launches us as a subprocess
