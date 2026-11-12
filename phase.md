Phase A: Foundation
0. Big picture (no code) ✅ (you still owe me your 30-second explanation)
1. Project setup: fix the .gitignore bug, install packages and Chromium
2. src/config.py: read settings from .env
3. src/llm.py: one function that creates the chat model and embeddings


Phase B: Data, Playwright and lakehouse
4. Look at the CSV and find the dirty data
5. mock_portal/app.py: a fake carrier website **
6. src/scraper.py part 1: Playwright logs in
7. src/scraper.py part 2: scrape many containers at once
8. src/lakehouse.py part 1: bronze
9. src/lakehouse.py part 2: silver (cleaning)
10. src/lakehouse.py part 3: gold KPIs and SQL

Phase C: RAG and documents
11. src/rag.py part 1: chunking
12. src/rag.py part 2: BM25 and vector index
13. src/rag.py part 3: hybrid search, RRF, rerank, answer
14. src/eval_rag.py: Hit@3 and MRR numbers
15. src/extract.py: PDF to validated JSON

Phase D: Agent, API and MCP
16. src/tools.py: the 6 business tools
17. src/agent.py: LangGraph agent
18. src/api.py part 1: FastAPI with an API key
19. src/api.py part 2: secure webhook
20. src/mcp_server.py: MCP server

Phase E: Azure and Fabric (bonus)
21. Azure AI Search (free tier)
22. Azure Document Intelligence (free tier)
23. Fabric Lakehouse (PySpark notebook)

Phase F: Polish and interview prep
24. Dockerfile and the Azure deployment story
25. README
26. Mock interview
27. Timed 45-minute rebuild drill