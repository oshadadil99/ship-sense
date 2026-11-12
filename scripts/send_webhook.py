"""Pretend to be the customer portal: send a SIGNED 'document uploaded' event (twice), then poll the result.
Run (API must be running on :8000):
    python -m scripts.send_webhook BL_OCL240917.pdf
    python -m scripts.send_webhook --bad          (wrong signature -> 401)"""
import json
import sys
import time
import uuid

import httpx

from src import config
from src.api import sign

API = "http://127.0.0.1:8000"


def send(event: dict, bad: bool = False) -> httpx.Response:
    body = json.dumps(event).encode()           # sign the EXACT bytes we send
    ts = str(int(time.time()))
    signature = "0" * 64 if bad else sign(body, ts)
    return httpx.post(f"{API}/webhooks/document-uploaded", content=body,
                      headers={"Content-Type": "application/json", "X-Timestamp": ts, "X-Signature": signature})


def main() -> None:
    bad = "--bad" in sys.argv
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    event = {"event_id": str(uuid.uuid4()), "filename": files[0] if files else "BL_OCL240917.pdf"}

    r = send(event, bad=bad)
    print(f"1st send -> {r.status_code} {r.json()}")
    if bad:
        return
    r = send(event)  # same event_id again = a retry from the sender
    print(f"2nd send -> {r.status_code} {r.json()}")

    print("polling for result", end="", flush=True)
    for _ in range(60):
        res = httpx.get(f"{API}/webhooks/results/{event['event_id']}",
                        headers={"X-API-Key": config.SHIPSENSE_API_KEY}).json()
        if res.get("status") != "processing":
            print("\n" + json.dumps(res, indent=2))
            return
        print(".", end="", flush=True)
        time.sleep(1)
    print("\ntimed out waiting for result")


if __name__ == "__main__":
    main()
