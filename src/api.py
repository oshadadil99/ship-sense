"""ShipSense API: the agent as a secure web service + a signed webhook.
Run:  uvicorn src.api:app --port 8000      Docs: http://127.0.0.1:8000/docs"""
import hashlib
import hmac
import json
import logging
import time

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from src import config
from src.tools import extract_shipping_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("shipsense.api")

app = FastAPI(title="ShipSense API", version="1.0")


# ============================== Step 18: API key + chat ==============================
def require_api_key(x_api_key: str = Header(default="")) -> None:
    if not config.SHIPSENSE_API_KEY:  # fail CLOSED: no key configured = nobody gets in
        raise HTTPException(503, "API key not configured on server")
    if not hmac.compare_digest(x_api_key, config.SHIPSENSE_API_KEY):  # constant-time compare
        raise HTTPException(401, "invalid or missing X-API-Key")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)                    # limits = cost + abuse control
    thread_id: str = Field(default="default", pattern=r"^[A-Za-z0-9_-]{1,64}$")


class ChatResponse(BaseModel):
    answer: str
    tools_used: list[str]
    latency_ms: int


@app.get("/health")
def health():  # no auth: Azure Container Apps probes call this
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat_endpoint(req: ChatRequest):  # plain def -> runs in a threadpool (agent + Playwright are blocking)
    from src.agent import chat
    start = time.perf_counter()
    try:
        out = chat(req.message, req.thread_id)
    except Exception:
        log.exception("agent failed thread=%s", req.thread_id)
        raise HTTPException(502, "agent error, please retry")  # never leak internals to the caller
    ms = int((time.perf_counter() - start) * 1000)
    log.info("chat thread=%s chars=%d tools=%s latency_ms=%d", req.thread_id, len(req.message), out["tools_used"], ms)
    return ChatResponse(answer=out["answer"], tools_used=out["tools_used"], latency_ms=ms)


@app.post("/extract/{filename}", dependencies=[Depends(require_api_key)])
def extract_endpoint(filename: str):
    result = extract_shipping_document(filename)
    if "error" in result:
        raise HTTPException(404, result)
    return result


# ============================== Step 19: signed webhook ==============================
MAX_AGE_SECONDS = 300               # replay window: 5 minutes
WEBHOOK_RESULTS: dict[str, dict] = {}  # event_id -> status/result (production: Redis / Cosmos DB / Table Storage)


def sign(body: bytes, timestamp: str) -> str:
    """HMAC-SHA256 over 'timestamp.body' -> the timestamp is signed too, so it can't be changed."""
    return hmac.new(config.WEBHOOK_SECRET.encode(), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()


def _process_document(event_id: str, filename: str) -> None:
    """Runs AFTER the 202 response was sent (background task)."""
    try:
        result = extract_shipping_document(filename)
        WEBHOOK_RESULTS[event_id] = {"status": "failed" if "error" in result else "done", "result": result}
    except Exception as e:
        WEBHOOK_RESULTS[event_id] = {"status": "failed", "result": {"error": type(e).__name__}}
    log.info("webhook event=%s status=%s", event_id, WEBHOOK_RESULTS[event_id]["status"])


@app.post("/webhooks/document-uploaded", status_code=202)
async def document_uploaded(request: Request, background: BackgroundTasks,
                            x_signature: str = Header(default=""), x_timestamp: str = Header(default="")):
    if not config.WEBHOOK_SECRET:
        raise HTTPException(503, "webhook secret not configured")
    body = await request.body()  # RAW bytes: the signature is over the exact bytes sent

    # 1. Replay protection: timestamp must be recent
    try:
        age = abs(time.time() - int(x_timestamp))
    except ValueError:
        raise HTTPException(401, "missing or bad X-Timestamp")
    if age > MAX_AGE_SECONDS:
        raise HTTPException(401, "timestamp too old (possible replay)")

    # 2. Authenticity: signature must match
    if not hmac.compare_digest(x_signature, sign(body, x_timestamp)):
        raise HTTPException(401, "invalid signature")

    # 3. Only now trust and parse the body
    try:
        event = json.loads(body)
        event_id, filename = str(event["event_id"]), str(event["filename"])
    except (ValueError, KeyError):
        raise HTTPException(400, "body must be JSON with event_id and filename")

    # 4. Idempotency: same event_id processed once (senders retry on timeouts)
    if event_id in WEBHOOK_RESULTS:
        return {"status": "duplicate_ignored", "event_id": event_id}
    WEBHOOK_RESULTS[event_id] = {"status": "processing"}  # no await between check and set -> no race

    # 5. Reply fast, work later
    background.add_task(_process_document, event_id, filename)
    log.info("webhook event=%s accepted file=%s", event_id, filename)
    return {"status": "accepted", "event_id": event_id}


@app.get("/webhooks/results/{event_id}", dependencies=[Depends(require_api_key)])
def webhook_result(event_id: str):
    if event_id not in WEBHOOK_RESULTS:
        raise HTTPException(404, "unknown event_id")
    return WEBHOOK_RESULTS[event_id]
