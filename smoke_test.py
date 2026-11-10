"""
Azure OpenAI smoke test for ShipSense.

Checks 3 things:
  1. Your endpoint + key are correct (authentication works)
  2. Your CHAT model deployment answers
  3. Your EMBEDDING model deployment returns a vector

Run:  python smoke_test.py
"""
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, APIConnectionError

load_dotenv()

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "")

missing = [n for n, v in {
    "AZURE_OPENAI_ENDPOINT": ENDPOINT,
    "AZURE_OPENAI_API_KEY": API_KEY,
    "AZURE_OPENAI_CHAT_DEPLOYMENT": CHAT_DEPLOYMENT,
}.items() if not v]
if missing:
    sys.exit(f"[X] Missing in .env: {', '.join(missing)}")

# Azure OpenAI "v1" API: the normal OpenAI client, pointed at your Azure resource.
# No api_version needed. The "model" you pass is your DEPLOYMENT NAME.
client = OpenAI(api_key=API_KEY, base_url=f"{ENDPOINT}/openai/v1/")


def explain(err: Exception) -> str:
    if isinstance(err, APIConnectionError):
        return "Cannot reach the endpoint. Check AZURE_OPENAI_ENDPOINT (copy it from 'Keys and Endpoint')."
    if isinstance(err, APIStatusError):
        code = err.status_code
        if code == 401:
            return "401 Unauthorized: wrong API key, or key from a different resource."
        if code == 404:
            return "404 Not found: deployment name is wrong (use the DEPLOYMENT name, not the model name)."
        if code == 429:
            return "429 Rate limit / quota: your deployment has 0 or very low TPM quota."
        return f"HTTP {code}: {err.message}"
    return repr(err)


ok = True

# --- Test 1 + 2: chat ---
print(f"[..] Chat test using deployment '{CHAT_DEPLOYMENT}' ...")
try:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a logistics assistant. Answer in one short sentence."},
            {"role": "user", "content": "What is a Bill of Lading?"},
        ],
        max_completion_tokens=200,
    )
    ms = (time.time() - t0) * 1000
    print(f"[OK] Chat works ({ms:.0f} ms)")
    print(f"     Answer: {resp.choices[0].message.content.strip()}")
    if resp.usage:
        print(f"     Tokens: prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens}")
except Exception as e:  # noqa: BLE001
    ok = False
    print(f"[X] Chat failed -> {explain(e)}")

# --- Test 3: embeddings ---
if EMBED_DEPLOYMENT:
    print(f"\n[..] Embedding test using deployment '{EMBED_DEPLOYMENT}' ...")
    try:
        emb = client.embeddings.create(model=EMBED_DEPLOYMENT, input="container MSCU1234567 delayed at Colombo port")
        print(f"[OK] Embeddings work: vector length = {len(emb.data[0].embedding)}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[X] Embeddings failed -> {explain(e)}")
else:
    print("\n[--] Skipped embedding test (AZURE_OPENAI_EMBED_DEPLOYMENT not set)")

print("\nRESULT:", "ALL GOOD - Azure OpenAI is ready" if ok else "Something failed - send me the output above")
