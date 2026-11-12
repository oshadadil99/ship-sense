"""Business tools = the agent's hands. Each tool: clear docstring (the LLM reads it), validated input,
errors returned as data ({"error": ...}) so the agent can recover instead of crashing."""
import json
import re

import pandas as pd

from src import config, lakehouse

CONTAINER_RE = re.compile(r"^[A-Z]{4}\d{7}$")   # ISO 6346 shape, e.g. OCLU1023757
SHIPMENT_RE = re.compile(r"^SHP-\d{7}$")        # our ERP id, e.g. SHP-2026003
COLUMNS = ["shipment_id", "container_no", "customer", "carrier", "pol", "pod", "status", "service_level",
           "teu", "freight_usd", "original_eta", "current_eta", "live_eta", "live_status",
           "delay_hours", "is_delayed"]


def _shipments() -> pd.DataFrame:
    if not lakehouse.exists("silver", "shipments"):
        lakehouse.build_all()  # first run: build the lakehouse automatically
    return lakehouse.read("silver", "shipments")


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> plain JSON-safe dicts (dates as YYYY-MM-DD, NaN/NaT as null)."""
    df = df[COLUMNS].copy()
    for col in ("original_eta", "current_eta", "live_eta"):
        df[col] = df[col].dt.strftime("%Y-%m-%d")
    return json.loads(df.to_json(orient="records"))


# ------------------------------------------------------------------------------------------
def search_policies(question: str) -> dict:
    """Answer questions about company policies: demurrage/detention charges and free time, customs
    clearance in Sri Lanka, dangerous goods, cargo claims, status codes, delay and escalation rules.
    Returns an answer with citations like [demurrage_and_detention#2]."""
    if not question or len(question) > 500:
        return {"error": "question must be 1-500 characters"}
    from src.rag import answer  # lazy import: loads the index only when needed
    return answer(question)


def get_shipment(identifier: str) -> dict:
    """Get one shipment by container number (e.g. OCLU1023757) or shipment id (e.g. SHP-2026003).
    Returns customer, carrier, ports, ETAs, live portal status, delay_hours and is_delayed."""
    ident = (identifier or "").strip().upper()
    if CONTAINER_RE.match(ident):
        col = "container_no"
    elif SHIPMENT_RE.match(ident):
        col = "shipment_id"
    else:
        return {"error": f"'{identifier}' is not a valid container number (4 letters + 7 digits) "
                         f"or shipment id (SHP-1234567)"}
    rows = _records(_shipments().query(f"{col} == @ident"))  # @ident = safe parameter, no injection
    return rows[0] if rows else {"error": f"no shipment found for {ident}"}


def list_delayed_shipments(port: str | None = None, carrier: str | None = None, limit: int = 10) -> dict:
    """List delayed shipments (more than 48h late), worst first. Optional filters: destination port
    (e.g. 'Singapore') and carrier (e.g. 'Oceanic Lines'). Partial, case-insensitive match."""
    limit = max(1, min(int(limit), 50))
    df = _shipments()
    df = df[df["is_delayed"]]
    if port:
        df = df[df["pod"].str.contains(port, case=False, regex=False)]
    if carrier:
        df = df[df["carrier"].str.contains(carrier, case=False, regex=False)]
    df = df.sort_values("delay_hours", ascending=False)
    return {"count": len(df), "shipments": _records(df.head(limit))}


def carrier_performance() -> dict:
    """On-time performance per carrier: shipments, TEU, on_time_pct and avg_delay_hours (worst first)."""
    _shipments()  # makes sure the lakehouse exists
    return {"carriers": json.loads(lakehouse.read("gold", "kpi_by_carrier").to_json(orient="records"))}


def live_track_container(container_no: str) -> dict:
    """Check the carrier portal NOW for a container's live ETA and events, then refresh the lakehouse.
    Use when the user asks for the live / latest / current status of a container. Takes ~5 seconds."""
    c = (container_no or "").strip().upper()
    if not CONTAINER_RE.match(c):
        return {"error": f"'{container_no}' is not a valid container number (4 letters + 7 digits)"}
    try:
        from src.scraper import track_containers_sync
        result = track_containers_sync([c])[0]
    except Exception as e:  # portal down, browser missing...
        return {"error": f"carrier portal unavailable ({type(e).__name__})"}
    if "error" in result:
        return result
    lakehouse.ingest_tracking([result])   # bronze: keep history
    lakehouse.build_silver()              # recompute delays with the new live ETA
    lakehouse.build_gold()
    return {"live": result, "shipment": get_shipment(c)}


def extract_shipping_document(filename: str) -> dict:
    """Extract structured data from a shipping PDF (Bill of Lading or commercial invoice) in the
    documents folder, e.g. 'BL_OCL240917.pdf'. Returns fields, validation issues and needs_human_review."""
    docs_dir = config.DOCUMENTS_DIR.resolve()
    path = (docs_dir / (filename or "")).resolve()
    if path.parent != docs_dir:  # blocks '../../.env' style path traversal
        return {"error": "file must be inside the documents folder"}
    if path.suffix.lower() != ".pdf" or not path.exists():
        return {"error": f"'{filename}' not found",
                "available": sorted(p.name for p in docs_dir.glob("*.pdf"))}
    from src.extract import extract
    return extract(path)


ALL_TOOLS = [search_policies, get_shipment, list_delayed_shipments, carrier_performance,
             live_track_container, extract_shipping_document]
