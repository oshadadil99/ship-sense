# ShipSense — Step-by-Step Build Guide (for Claude Code)

> **How to use this file:** put it in the project root (`F:\My Projects\ship-sense\SHIPSENSE_BUILD_GUIDE.md`)
> and tell Claude Code: *"Read SHIPSENSE_BUILD_GUIDE.md and follow the Teaching Rules. Start with Step 0."*
> Then say **"next step"** each time you finish one.

---

## 0. Context for Claude Code (read first)

**Who I am:** Oshada, AI engineer (Python, FastAPI, LangGraph, RAG, MCP on GCP). I have a **physical interview in
2 days** for *Senior Executive – AI Integration Engineer* at a logistics company in Colombo. They will give me a
practical task. The job needs: agentic AI (LangGraph), advanced RAG (hybrid retrieval, reranking, evaluation),
Playwright automation, document automation, secure APIs / webhooks / MCP tools, **Microsoft Azure** and
**Microsoft Fabric**. Azure and Fabric and Playwright are new to me.

**Goal:** build a small project called **ShipSense** (an AI assistant for a freight forwarder) step by step, so I
**understand every part** and can **explain it and rebuild it under time pressure** in the interview.

**What is already done:**
- Azure subscription is active, budget alert set at $4.
- Resource group `rg-shipsense` (East US 2), Microsoft Foundry resource `shipsense-ai-oshada`, project `shipsense-project`.
- Deployments: `gpt-4.1-mini` (Global Standard, 20K TPM) and `text-embedding-3-small`.
- `.env` exists with `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_CHAT_DEPLOYMENT`,
  `AZURE_OPENAI_EMBED_DEPLOYMENT`. `smoke_test.py` passed (chat + embeddings work).
- The `data/` folder exists: `data/raw/erp_shipments.csv`, `data/knowledge/*.md` (5 policy docs),
  `data/documents/*.pdf` (a Bill of Lading and a commercial invoice).

**My machine:** Windows, PowerShell, Python venv in `venv\` (activate: `.\venv\Scripts\Activate.ps1`).

---

## 1. Teaching Rules (Claude Code MUST follow these)

1. **One step at a time.** Do only the current step. Then STOP and wait for me to say "next step".
2. **Before writing code**, explain in 3–5 simple sentences: *what* we are building, *why* a company needs it,
   and the *one key concept* behind it. Use a logistics example.
3. **Small code.** Keep each file short and readable. Add short comments on the important lines only.
   Prefer simple code over clever code. No extra features I didn't ask for.
4. **After writing code**, give me the exact PowerShell command to run it and tell me what output to expect.
5. **After it works**, give me:
   - a **2-line "interview sentence"** I can say about this step, and
   - **2 quick questions** to check I understood (give answers only after I reply).
6. If something fails, explain the error in simple words first, then fix it.
7. **Never** print or ask for my API keys. Secrets only live in `.env`. Never hard-code secrets.
8. Windows commands only (PowerShell). Use `python -m <module>` style so imports work from the project root.

---

## 2. Final project structure (what we are building towards)

```
ship-sense/
├── .env                      # secrets (already exists, never commit)
├── .env.example
├── .gitignore
├── requirements.txt
├── smoke_test.py             # already exists
├── data/                     # already exists
│   ├── raw/erp_shipments.csv
│   ├── knowledge/*.md
│   └── documents/*.pdf
├── mock_portal/app.py        # fake carrier website (Playwright target)
├── src/
│   ├── __init__.py
│   ├── config.py             # settings from .env
│   ├── llm.py                # Azure OpenAI chat + embeddings
│   ├── scraper.py            # Playwright
│   ├── lakehouse.py          # bronze/silver/gold Delta tables
│   ├── rag.py                # hybrid RAG
│   ├── eval_rag.py           # RAG evaluation
│   ├── extract.py            # PDF -> structured JSON
│   ├── tools.py              # business tools
│   ├── agent.py              # LangGraph agent
│   ├── api.py                # FastAPI + webhook
│   └── mcp_server.py         # MCP server
├── scripts/send_webhook.py
├── fabric/fabric_notebook.py # same lakehouse in PySpark for Microsoft Fabric
└── Dockerfile
```

---

## 3. Important technical notes (so Claude Code doesn't hit known problems)

- **Azure OpenAI v1 endpoint:** use the normal OpenAI-compatible clients with
  `base_url = f"{AZURE_OPENAI_ENDPOINT}/openai/v1/"` and pass the **deployment name** as `model`.
  No `api_version` needed. In LangChain: `ChatOpenAI(model=deployment, base_url=..., api_key=...)` and
  `OpenAIEmbeddings(model=deployment, base_url=..., api_key=..., check_embedding_ctx_length=False)`.
- **MCP Python SDK 2.x:** `FastMCP` was renamed. Use `from mcp.server.mcpserver import MCPServer`,
  `mcp = MCPServer(name="shipsense")`, `@mcp.tool()`, `mcp.run()` (stdio) or
  `mcp.run(transport="streamable-http", port=8002)`. (Or pin `mcp<2` and use FastMCP — pick one and say which.)
- **Playwright on Windows:** after `pip install playwright` run `playwright install chromium`.
- **Blocking tools inside FastAPI:** make `/chat` a normal `def` (not `async def`) so FastAPI runs it in a
  threadpool; then `asyncio.run(...)` for Playwright inside a tool is safe.
- **Model is "Legacy"** (gpt-4.1-mini) — fine for this project; mention version-upgrade planning in the interview.
- Keep costs tiny: always use `gpt-4.1-mini`, small prompts, `temperature=0`, `max_retries=3`, `timeout=60`.

---

## 4. requirements.txt (install once in Step 1)

```txt
python-dotenv>=1.0
openai>=1.60
langchain-core>=0.3
langchain-openai>=0.3
langgraph>=0.4
fastapi>=0.115
uvicorn[standard]>=0.30
httpx>=0.27
pydantic>=2.7
python-multipart>=0.0.9
mcp>=2.3
playwright>=1.48
rank-bm25>=0.2.2
numpy>=1.26
pypdf>=4.0
pandas>=2.2
pyarrow>=16.0
deltalake>=0.20
duckdb>=1.0
azure-identity>=1.17
azure-search-documents>=11.5
azure-ai-documentintelligence>=1.0
```

---

# THE STEPS

Each step has: **Goal · Concept · Build · Run · Done when**. Total ≈ 6–8 hours. Phases A–D are the core
(do these first). Phases E–F are bonus if time allows.

---

## PHASE A — Foundation (≈ 40 min)

### Step 0 — Understand the big picture (no code)
- **Goal:** I can explain what ShipSense does in 30 seconds.
- **Claude Code:** draw a simple ASCII diagram of the data flow:
  carrier portal → Playwright → lakehouse ← ERP CSV; policies → RAG; PDFs → extraction;
  all → tools → LangGraph agent → FastAPI / MCP → Azure OpenAI.
  Explain each box in one sentence. Map each box to a line in the job description.
- **Done when:** I can repeat the 30-second explanation back to you.

### Step 1 — Project setup
- **Goal:** clean project skeleton + all packages installed.
- **Build:** `requirements.txt` (section 4), `.gitignore` (`.env`, `venv/`, `__pycache__/`, `data/lakehouse/`,
  `data/index/`, `data/.portal_state.json`), empty `src/__init__.py`, `.env.example` (same keys as `.env`, fake values,
  plus `SHIPSENSE_API_KEY`, `WEBHOOK_SECRET`, `PORTAL_URL=http://127.0.0.1:8001`, `PORTAL_USER=demo`, `PORTAL_PASSWORD=demo123`).
- **Run:** `pip install -r requirements.txt` then `playwright install chromium`
- **Done when:** install finishes with no errors. Add the new keys to my real `.env` too.

### Step 2 — `src/config.py`: settings from environment
- **Concept:** 12-factor config — code reads settings from environment variables; secrets never in code.
  On Azure the same variables come from Container Apps secrets / Key Vault.
- **Build:** load `.env` with python-dotenv; define paths (`ROOT`, `DATA_DIR`, `KNOWLEDGE_DIR`, `DOCUMENTS_DIR`,
  `RAW_DIR`, `LAKEHOUSE_DIR`, `INDEX_DIR`) and all env values with safe defaults.
- **Run:** `python -c "from src import config; print(config.DATA_DIR, config.AZURE_OPENAI_CHAT_DEPLOYMENT)"`
- **Done when:** it prints the data path and `gpt-4.1-mini`.

### Step 3 — `src/llm.py`: one place to create the model
- **Concept:** a single factory for the LLM and embeddings, so switching Azure ↔ OpenAI is one setting.
  Explain *deployment name vs model name* in Azure, and what TPM means (rate limit, not price).
- **Build:** `get_chat_model()` (ChatOpenAI, temperature 0, timeout 60, max_retries 3) and `get_embeddings()`,
  both using the Azure v1 base URL (see section 3). Cache them with `functools.lru_cache`.
- **Run:** `python -c "from src.llm import get_chat_model; print(get_chat_model().invoke('Say hi in 3 words').content)"`
- **Done when:** the model replies.

---

## PHASE B — Data + Playwright + Lakehouse (≈ 1.5–2 h)

### Step 4 — Look at the data (no new code)
- **Goal:** know the data before using it.
- **Claude Code:** open `data/raw/erp_shipments.csv` with pandas in a one-liner and show me: columns, row count,
  the **dirty parts** (3 different date formats, duplicate rows, lowercase container numbers, empty `teu`).
  Briefly show what's in `data/knowledge/` and `data/documents/`.
- **Done when:** I can name 4 data-quality problems. (These are why we need a *silver* layer later.)

### Step 5 — `mock_portal/app.py`: a fake carrier website
- **Concept:** we need a website to automate. Real carrier sites have terms of use, so we practise on our own.
- **Build (FastAPI, plain HTML strings, no templates):**
  - `GET /login` form with inputs `#username`, `#password`, button `#login-btn`.
  - `POST /login`: if `demo/demo123` → set an httponly `session` cookie → redirect to `/track`; else back to
    `/login?error=...`.
  - `GET /track?container=XXX`: redirect to login if no session. Shows input `#container-input`, button
    `#search-btn`. If the container exists in the CSV (case-insensitive), show `#result` with `#eta` and a table
    `#events` with rows `tr.event` (columns: time, code, event, location). Else show `#not-found`.
  - Events + a "live ETA" must be **deterministic** (hash of the container number), live ETA = CSV `current_eta`
    + 0–3 days (so sometimes the portal knows about a delay the ERP doesn't).
- **Run:** `uvicorn mock_portal.app:app --port 8001` → open http://127.0.0.1:8001, login, track `OCLU1023757`.
- **Done when:** I can log in and see the events table in my browser.

### Step 6 — `src/scraper.py` part 1: login with Playwright
- **Concept:** Playwright drives a real browser. Explain: browser → context → page; **locators** and
  **auto-waiting** (why we never use `time.sleep`); headless vs headed; async API.
- **Build:** `async _login(browser)`: goto `/login`, fill username/password from config, click, `wait_for_url("**/track")`,
  then **save `storage_state`** to `data/.portal_state.json`.
- **Run:** a small `__main__` that launches Chromium **headed** and logs in.
- **Done when:** I *see* the browser open and log in, and the state file is created.

### Step 7 — `src/scraper.py` part 2: scrape many containers
- **Concept:** reuse the saved session (`storage_state`) so we log in once; run several pages **concurrently**
  with `asyncio.gather` + a **Semaphore** (be polite to the site).
- **Build:** `async _track_one(browser, container, sem)` → new context with storage_state, search, wait for
  `#result` **or** `#not-found` (`locator.or_()`), read rows → return
  `{container_no, live_eta, latest_status, events[], scraped_at}` or `{container_no, error: "not_found"|"timeout"|"session_expired"}`.
  `async track_containers(list, headless=True, concurrency=3)`: login if no state file; if any
  `session_expired` → login again once and retry. Plus `track_containers_sync()` wrapper using `asyncio.run`.
  CLI: `python -m src.scraper OCLU1023757 BWSU1007919 XXXX0000000 [--headed]`.
- **Run:** the CLI command above (portal must be running).
- **Done when:** JSON shows 2 results + 1 `not_found`.

### Step 8 — `src/lakehouse.py` part 1: BRONZE
- **Concept:** **Medallion architecture** (bronze = raw, silver = clean, gold = business KPIs) and **Delta Lake**
  (Parquet + transaction log = exactly what Microsoft Fabric OneLake stores). Bronze is never edited.
- **Build:** `ingest_erp_csv()` → read CSV as strings, add `_ingested_at`, `_source`, write Delta to
  `data/lakehouse/bronze/erp_shipments` (overwrite). `ingest_tracking(results)` → append scraper results
  (container_no + JSON payload) to `bronze/portal_tracking`. Helper `read(layer, name)`.
- **Done when:** the bronze folder contains Parquet files + a `_delta_log` folder. Show me the `_delta_log` and
  explain what it is.

### Step 9 — `src/lakehouse.py` part 2: SILVER
- **Concept:** cleaning rules = trust. Each rule fixes one problem found in Step 4.
- **Build:** `build_silver()`: dedupe by `shipment_id`; uppercase + trim `container_no`; parse 3 date formats;
  `teu` → int (default 1); `freight_usd` → number; merge **latest** portal `live_eta` / `live_status` per container
  (if the tracking table exists); `delay_hours = (live_eta or current_eta) - original_eta` in hours;
  `is_delayed = delay_hours > 48` (rule from `shipment_status_codes.md`).
- **Done when:** silver has 40 rows (duplicates gone) and correct date types.

### Step 10 — `src/lakehouse.py` part 3: GOLD + SQL
- **Concept:** gold tables answer business questions directly (Power BI, the agent). DuckDB = SQL over the
  tables, like Fabric's **Lakehouse SQL analytics endpoint**.
- **Build:** `build_gold()` with DuckDB SQL → `gold/kpi_by_carrier` (shipments, teu, on_time_pct,
  avg_delay_hours) and `gold/kpi_by_port` (shipments, delayed, freight_usd). `query(sql)` registers all
  silver/gold tables as `silver_<name>` / `gold_<name>`. `build_all()` + `__main__`.
- **Run:** `python -m src.lakehouse`, then scrape 10 containers and rebuild silver/gold — watch live ETAs appear.
- **Done when:** I see the carrier KPI table printed.

---

## PHASE C — RAG + Documents (≈ 1.5–2 h)

### Step 11 — `src/rag.py` part 1: chunking
- **Concept:** why chunk; **structure-aware chunking** (split Markdown on `## ` sections, keep the doc title
  inside every chunk so each chunk makes sense alone).
- **Build:** `Chunk` dataclass (id like `demurrage_and_detention#2`, source, title, section, text);
  `chunk_markdown()`.
- **Done when:** it prints 17 chunk ids with their section names.

### Step 12 — `src/rag.py` part 2: BM25 + vector index
- **Concept:** **BM25** = keyword scoring (great for exact terms: "Class 9", "CusDec", container numbers).
  **Embeddings** = meaning ("late" ≈ "delayed"). Cosine similarity; normalise vectors once so dot product = cosine.
- **Build:** `build_index()` → embed all chunks (Azure embeddings), save `data/index/chunks.json` + `vectors.npy`.
  `LocalIndex` class with `bm25_search(q,k)` (rank_bm25) and `vector_search(q,k)` (numpy).
  CLI: `python -m src.rag build`.
- **Done when:** index files exist; show me top-3 results from BM25 vs vector for
  "Do you ship lithium batteries?" and explain the difference.

### Step 13 — `src/rag.py` part 3: hybrid + RRF + rerank + answer
- **Concept:** **Reciprocal Rank Fusion**: `score = Σ 1/(60 + rank)` — merges two ranked lists without
  normalising scores (Azure AI Search uses RRF for hybrid). **Reranking**: a second, smarter pass on the top
  candidates (cross-encoder / LLM judge). **Grounding**: answer only from context, cite ids, say "not found".
- **Build:** `rrf(rankings)`, `LocalIndex.hybrid_search(q,k=8)`, `rerank(question, candidates, top_n=3)` using
  `with_structured_output` (list of 0–10 scores), `retrieve()`, `answer()` returning `{answer, sources}`.
  CLI: `python -m src.rag ask "..."`.
- **Run:** ask "What is the daily charge for a 40ft container on day 7 after free time?" and
  "What is the CEO's name?" (should say not found).
- **Done when:** correct answer with citation, and a proper "not found" for the second.

### Step 14 — `src/eval_rag.py`: prove it with numbers
- **Concept:** you can't improve what you don't measure. **Golden set**, **Hit@3**, **MRR**.
- **Build:** ~14 (question, expected chunk id) pairs covering all 5 docs (use different words than the docs!).
  Compare BM25 only / vector only / hybrid RRF / hybrid + rerank. Print a small table.
- **Run:** `python -m src.eval_rag`
- **Done when:** I have a results table and can explain which method won and why. **Write the numbers down —
  great for the interview.**

### Step 15 — `src/extract.py`: document automation
- **Concept:** 3 stages: **read** (OCR / text: Azure Document Intelligence `prebuilt-layout` in production,
  `pypdf` locally) → **extract** (LLM **structured output** into a Pydantic schema — no fragile JSON parsing) →
  **validate** (business rules → human review if anything fails = human-in-the-loop).
- **Build:** Pydantic `ShippingDocument` (doc_type, document_number, document_date, shipper_or_seller,
  consignee_or_buyer, ports, vessel_voyage, container_numbers, gross_weight_kg, incoterms, currency,
  total_amount, line_items[description, hs_code, quantity, unit_price, amount]). `read_text()` (Document Intelligence
  if configured else pypdf). `validate()`: container regex `^[A-Z]{4}\d{7}$` (ISO 6346 shape), line items sum = total,
  HS code present on invoice lines. `extract(path)` → `{file, reader, data, issues, needs_human_review}`.
- **Run:** `python -m src.extract`
- **Done when:** both PDFs return clean JSON; then break a value on purpose to see validation flag it.

---

## PHASE D — Agent + API + MCP (≈ 1.5–2 h)

### Step 16 — `src/tools.py`: business tools
- **Concept:** tools are the agent's hands. Good tools = clear docstring (the LLM reads it!), **validated input**,
  and **errors returned as data** (not exceptions) so the agent can recover.
- **Build:** `search_policies(question)`, `get_shipment(identifier)` (container or `SHP-xxxxxxx`, regex check),
  `list_delayed_shipments(port=None, carrier=None, limit=10)`, `carrier_performance()`,
  `live_track_container(container_no)` (scrape → ingest → rebuild silver/gold),
  `extract_shipping_document(filename)` (**block path traversal**: file must be inside `data/documents`).
- **Run:** call each tool in a `python -c` one-liner.
- **Done when:** all 6 return sensible data, and bad inputs return `{"error": ...}`.

### Step 17 — `src/agent.py`: LangGraph agent
- **Concept:** explicit graph: **State** (messages) → **agent node** (LLM with tools bound) → conditional edge
  (`tools_condition`) → **tools node** (`ToolNode`) → back to agent. **Checkpointer** (`InMemorySaver`) = memory
  per `thread_id`. **recursion_limit** = guardrail against tool loops. Explain why a graph is better than a plain
  loop (control, observability, human-in-the-loop later).
- **Build:** `StructuredTool.from_function` for the 6 tools; a clear system prompt (always cite policy ids, never
  invent numbers, show calculations); `build_agent()`; `chat(message, thread_id)` → `{answer, tools_used}`;
  a CLI chat loop.
- **Run:** `python -m src.agent` and try:
  `Which shipments to Singapore are delayed?` · `Get the live status of OCLU1023757` ·
  `What is the demurrage for a 40ft container 8 days after free time?` ·
  `Which carrier is worst on-time?` · `Extract BL_OCL240917.pdf` · then a follow-up "and who is the customer?"
- **Done when:** answers are correct and `tools_used` shows the right tools.

### Step 18 — `src/api.py` part 1: FastAPI + API key
- **Concept:** the agent becomes a service. API key auth with `hmac.compare_digest` (constant-time),
  Pydantic input limits, `/health` for Container Apps probes, logging tools + latency.
- **Build:** `GET /health`, `POST /chat` (header `X-API-Key`, body `{message, thread_id}`), `POST /extract/{filename}`.
- **Run:** `uvicorn src.api:app --port 8000` → http://127.0.0.1:8000/docs
- **Done when:** wrong key → 401; right key → agent answer.

### Step 19 — `src/api.py` part 2: secure webhook
- **Concept:** event-driven integration. **HMAC-SHA256 signature** over `timestamp + "." + raw body` (like
  Stripe/Meta), **replay protection** (reject timestamps older than 5 min), **idempotency** (same `event_id`
  processed once — senders retry), **return 202 fast + process in background**.
- **Build:** `sign(body, ts)`, `POST /webhooks/document-uploaded` (headers `X-Signature`, `X-Timestamp`),
  background task runs `extract_shipping_document`, `GET /webhooks/results/{event_id}`.
  Plus `scripts/send_webhook.py` that sends a signed event twice (2nd = duplicate) and polls for the result;
  `--bad` flag sends a wrong signature.
- **Run:** `python -m scripts.send_webhook BL_OCL240917.pdf` and `python -m scripts.send_webhook --bad`
- **Done when:** I see accepted → duplicate_ignored → extraction JSON, and the bad one gets 401.

### Step 20 — `src/mcp_server.py`: MCP server
- **Concept:** MCP = a standard plug so **any** AI client (Copilot in VS Code, Claude Desktop, Foundry agents)
  can use our tools without custom glue. Tools + resources + prompts; stdio vs streamable HTTP transport.
- **Build:** `MCPServer(name="shipsense")` exposing get_shipment, list_delayed_shipments, carrier_performance,
  search_policies, live_track_container (async). `--http` flag → port 8002.
- **Run:** `python -m src.mcp_server --http` and test with `npx @modelcontextprotocol/inspector` (needs Node.js),
  or a tiny Python script that calls `await mcp.list_tools()` / `await mcp.call_tool(...)`.
- **Done when:** I can list and call the tools.

---

## PHASE E — Azure + Fabric (bonus, ≈ 1–1.5 h)

### Step 21 — Azure AI Search (Free tier)
- **Concept:** managed hybrid search: BM25 + HNSW vectors + RRF built in. Free tier has **no semantic ranker** →
  we keep our own LLM reranker.
- **Portal (guide me click by click):** Create "AI Search" in `rg-shipsense`, **Pricing tier = Free** (double check!).
  Copy URL + Admin key into `.env` as `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_KEY`, `AZURE_SEARCH_INDEX=shipsense-knowledge`.
- **Build:** `src/azure_search.py`: `ensure_index()` (key, chunk_id, source, title, section, text, vector 1536 dims,
  HNSW profile), `upload_chunks()`, `hybrid_search(question, vector, k)` with `VectorizedQuery`.
  In `rag.py`: if Azure Search configured → upload in `build_index()` and use it in `retrieve()`.
- **Done when:** the same RAG questions work through Azure AI Search, and I can see the index in the portal.

### Step 22 — Azure AI Document Intelligence (F0 free)
- **Portal:** Create "Document intelligence", **Free F0**, copy endpoint/key → `AZURE_DOCINTEL_ENDPOINT/KEY`.
- **Build:** in `extract.py` `read_text()` use `DocumentIntelligenceClient.begin_analyze_document("prebuilt-layout", body=f)`.
- **Done when:** `reader` shows `azure-document-intelligence` in the output.

### Step 23 — Microsoft Fabric Lakehouse
- **Concept:** Fabric = SaaS analytics on **OneLake**; Lakehouse (files + Delta tables, Spark) vs Warehouse (SQL);
  Data Factory pipelines; Power BI on top; SQL analytics endpoint.
- **Account:** Fabric needs a work-style account: Azure portal → Microsoft Entra ID → Users → New user
  (`fabricadmin@<directory>.onmicrosoft.com`) → sign in at app.fabric.microsoft.com → start free trial.
- **Build:** `fabric/fabric_notebook.py` — the same bronze/silver/gold pipeline in **PySpark** (cells: bronze from
  `Files/raw/erp_shipments.csv` → `saveAsTable("bronze_erp_shipments")`; silver with `F.coalesce(F.to_date(...))`
  for 3 date formats, dedupe, types, delay rule; gold aggregates; a SQL cell).
  Optional cell: call Azure OpenAI from the notebook with the key from Key Vault (`notebookutils.credentials.getSecret`).
- **Done when:** tables appear under *Tables* in the Lakehouse; bonus: a 1-page Power BI report of on-time % by carrier.

---

## PHASE F — Production polish + interview prep (≈ 1 h)

### Step 24 — Dockerfile + deployment story
- **Build:** Dockerfile based on `mcr.microsoft.com/playwright/python` image, install requirements +
  `playwright install chromium`, non-root user, `CMD uvicorn src.api:app --host 0.0.0.0 --port 8000`. `.dockerignore`.
- **Explain (no need to deploy):** how I'd deploy to **Azure Container Apps**: Azure Container Registry → Container App →
  secrets / Key Vault refs → **Managed Identity** to call Foundry + AI Search (no keys) → App Insights → scale rules.
- **Done when:** I can explain the deployment in 60 seconds.

### Step 25 — README with architecture + talking points
- **Build:** README with: ASCII architecture diagram, how to run each step, and for every component a
  **"what it proves"** paragraph (agentic AI, RAG + eval numbers, document automation, Playwright, secure integration,
  MCP, lakehouse/Fabric, Azure RBAC control-plane vs data-plane lesson, production next steps).
- **Done when:** README is complete.

### Step 26 — Mock interview (Claude Code acts as the interviewer)
- Ask me 10 questions, one at a time, mixing: "walk me through your architecture", "why hybrid search",
  "how do you secure a webhook", "what if the carrier site changes its HTML", "how do you evaluate RAG",
  "how would you deploy this on Azure", "what is OneLake", "how do you stop the agent looping",
  "how do you handle a scanned PDF", "how do you control LLM cost".
- After each answer: score 1–5 and give a better 3-sentence version.

### Step 27 — Timed rebuild drill (most important for a practical task!)
- Give me a **45-minute mini task** similar to what the company might set, e.g.
  *"Build a FastAPI endpoint that takes a PDF invoice, extracts fields with Azure OpenAI into a Pydantic schema,
  validates them, and answers questions about it with a small LangGraph agent."*
- I code it myself; you only give hints when I ask. Review my code at the end.

---

## 5. Interview cheat-sheet (fill in as we go)

| Topic | My one-liner | My numbers / proof |
|---|---|---|
| Hybrid RAG | BM25 for exact codes + vectors for meaning, fused with RRF, then rerank | Hit@3 / MRR from Step 14: ___ |
| Agent | LangGraph state graph, ToolNode, checkpointer memory, recursion limit | tools used: ___ |
| Playwright | locators + auto-wait, storage_state login once, semaphore concurrency | ___ containers in ___ s |
| Webhook security | HMAC(timestamp.body), 5-min replay window, idempotent event_id, 202 + background | |
| Documents | Doc Intelligence → structured output → validation → human review | |
| Lakehouse | bronze/silver/gold Delta = Fabric OneLake format | silver fixed ___ issues |
| Azure | Foundry deployment, TPM cap, content filter, Managed Identity, RBAC data plane | cost so far: $___ |
