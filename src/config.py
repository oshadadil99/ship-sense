"""All settings in one place. Values come from environment variables (.env locally,
Container Apps secrets / Key Vault on Azure). Never hard-code secrets here."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ (does nothing if the file is missing)

# ---- Paths (all relative to the project root, so it works from any folder) ----
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
DOCUMENTS_DIR = DATA_DIR / "documents"
LAKEHOUSE_DIR = DATA_DIR / "lakehouse"
INDEX_DIR = DATA_DIR / "index"
PORTAL_STATE_FILE = DATA_DIR / ".portal_state.json"

# ---- Azure OpenAI ----
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1-mini")
AZURE_OPENAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")

# ---- Our API + webhook security ----
SHIPSENSE_API_KEY = os.getenv("SHIPSENSE_API_KEY", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

# ---- Mock carrier portal (Playwright target) ----
PORTAL_URL = os.getenv("PORTAL_URL", "http://127.0.0.1:8001").rstrip("/")
PORTAL_USER = os.getenv("PORTAL_USER", "demo")
PORTAL_PASSWORD = os.getenv("PORTAL_PASSWORD", "demo123")
