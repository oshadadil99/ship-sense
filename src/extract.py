"""Document automation: READ (PDF -> text) -> EXTRACT (LLM -> Pydantic schema) -> VALIDATE (business rules).
Run:  python -m src.extract"""
import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pypdf import PdfReader

from src import config
from src.llm import get_chat_model

CONTAINER_RE = re.compile(r"^[A-Z]{4}\d{7}$")  # ISO 6346 shape: 4 letters + 7 digits


# ---------- The schema = the contract. The LLM must fill exactly these fields. ----------
class LineItem(BaseModel):
    description: str
    hs_code: str | None = None
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = None


class ShippingDocument(BaseModel):
    doc_type: Literal["bill_of_lading", "commercial_invoice", "other"]
    document_number: str | None = None
    document_date: str | None = Field(None, description="ISO format YYYY-MM-DD")
    shipper_or_seller: str | None = None
    consignee_or_buyer: str | None = None
    port_of_loading: str | None = None
    port_of_discharge: str | None = None
    vessel_voyage: str | None = None
    container_numbers: list[str] = []
    gross_weight_kg: float | None = None
    incoterms: str | None = None
    currency: str | None = None
    total_amount: float | None = None
    line_items: list[LineItem] = []


# ---------- 1. READ ----------
def _read_with_docintel(path: Path) -> str:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    client = DocumentIntelligenceClient(config.AZURE_DOCINTEL_ENDPOINT, AzureKeyCredential(config.AZURE_DOCINTEL_KEY))
    with open(path, "rb") as f:
        poller = client.begin_analyze_document(   # long-running job: upload -> Azure analyses -> we poll
            "prebuilt-layout", body=f,            # ready-made model: OCR + reading order + tables
            output_content_format="markdown",     # tables come back as Markdown tables -> easy for the LLM
        )
    return poller.result().content


def read_text(path: Path) -> tuple[str, str]:
    """Return (text, reader_name). Azure Document Intelligence if configured (OCR -> works on scans),
    else pypdf (only works on PDFs that already contain text)."""
    if config.AZURE_DOCINTEL_ENDPOINT and config.AZURE_DOCINTEL_KEY:
        try:
            return _read_with_docintel(path), "azure-document-intelligence"
        except Exception as e:  # Azure down / quota hit -> keep working with the local reader
            print(f"Document Intelligence failed ({type(e).__name__}), falling back to pypdf", file=sys.stderr)
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return text, "pypdf"


# ---------- 2. EXTRACT ----------
def extract_fields(text: str) -> ShippingDocument:
    prompt = (
        "Extract the fields of this shipping document. Use null for anything not present - never guess.\n"
        "Dates as YYYY-MM-DD. Slash dates in these documents are DAY/MONTH/YEAR (12/10/2026 = 12 October).\n"
        "Numbers without commas or currency symbols. Weight in kg.\n\n"
        f"Document:\n{text}"
    )
    return get_chat_model().with_structured_output(ShippingDocument).invoke(prompt)  # typed object back


# ---------- 3. VALIDATE ----------
def validate(doc: ShippingDocument) -> list[str]:
    """Business rules. Any issue -> a human must review (human-in-the-loop)."""
    issues = []
    if not doc.document_number:
        issues.append("missing document number")
    if not doc.container_numbers:
        issues.append("no container number found")
    for c in doc.container_numbers:
        if not CONTAINER_RE.match(c):
            issues.append(f"invalid container number format: {c}")
    if doc.doc_type == "commercial_invoice":
        line_sum = sum(li.amount or 0 for li in doc.line_items)
        if doc.total_amount is None or abs(line_sum - doc.total_amount) > 0.01:
            issues.append(f"line items sum {line_sum:.2f} != total {doc.total_amount}")
        for i, li in enumerate(doc.line_items, start=1):
            if not li.hs_code:
                issues.append(f"line {i} has no HS code")
    return issues


def extract(path: str | Path) -> dict:
    path = Path(path)
    text, reader = read_text(path)
    doc = extract_fields(text)
    issues = validate(doc)
    return {"file": path.name, "reader": reader, "data": doc.model_dump(),
            "issues": issues, "needs_human_review": bool(issues)}


if __name__ == "__main__":
    for pdf in sorted(config.DOCUMENTS_DIR.glob("*.pdf")):
        print(json.dumps(extract(pdf), indent=2))

    # Demo: break values on purpose and watch validation catch them
    print("\n--- broken invoice demo ---")
    text, _ = read_text(config.DOCUMENTS_DIR / "INV_LAL-2026-0331.pdf")
    broken = extract_fields(text)
    broken.total_amount = 52_000.0               # total no longer matches the lines
    broken.container_numbers = ["BWSU100791"]    # one digit missing
    broken.line_items[1].hs_code = None          # HS code removed
    print(json.dumps({"issues": validate(broken), "needs_human_review": True}, indent=2))
