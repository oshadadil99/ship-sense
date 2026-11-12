"""Measure retrieval quality on a golden set. Run: python -m src.eval_rag
Hit@3 = expected chunk is in the top 3.  MRR = average of 1/rank of the expected chunk (0 if not found)."""
from src import azure_search
from src.rag import azure_hybrid_search, get_index, rerank

# (question in DIFFERENT words than the docs, expected chunk id) - covers all 5 docs
GOLDEN = [
    ("How many days can a refrigerated box stay at Colombo before we pay?", "demurrage_and_detention#1"),
    ("Daily cost for a forty-foot box left at the terminal two weeks past the allowance?", "demurrage_and_detention#2"),
    ("Can the port storage fees be cancelled if the shipping line caused the hold-up?", "demurrage_and_detention#3"),
    ("What's the difference between charges inside and outside the terminal?", "demurrage_and_detention#0"),
    ("Will you carry fireworks or nuclear material?", "dangerous_goods#1"),
    ("How early must a hazardous cargo booking be made?", "dangerous_goods#2"),
    ("What paperwork is needed before you accept hazardous cargo?", "dangerous_goods#0"),
    ("Which papers do I need to clear imported goods in Sri Lanka?", "customs_clearance_sri_lanka#0"),
    ("How long does a physical inspection by customs usually take?", "customs_clearance_sri_lanka#2"),
    ("How many digits must the tariff classification number have?", "customs_clearance_sri_lanka#1"),
    ("Customs stopped my box - how quickly will someone tell me?", "customs_clearance_sri_lanka#3"),
    ("When do we officially call a shipment late?", "shipment_status_codes#1"),
    ("When should a late shipment be raised to the account manager?", "shipment_status_codes#2"),
    ("What does DSC mean in tracking?", "shipment_status_codes#0"),
    ("How long after delivery can the customer report hidden damage?", "cargo_claims#0"),
    ("What is the most you pay per package if there is no insurance?", "cargo_claims#2"),
    # exact-term questions (codes / jargon) - where keyword search shines
    ("Is Class 9 accepted?", "dangerous_goods#1"),
    ("Where is the CusDec lodged in ASYCUDA?", "customs_clearance_sri_lanka#0"),
    ("HLD older than 24 hours?", "shipment_status_codes#2"),
    ("Hague-Visby limit USD 500", "cargo_claims#2"),
]


def _rank(ids: list[str], expected: str) -> int | None:
    return ids.index(expected) + 1 if expected in ids else None


def evaluate() -> dict[str, dict]:
    idx = get_index()
    methods = {
        "BM25 only": lambda q: [c.id for c, _ in idx.bm25_search(q, 8)],
        "Vector only": lambda q: [c.id for c, _ in idx.vector_search(q, 8)],
        "Hybrid RRF": lambda q: [c.id for c, _ in idx.hybrid_search(q, 8)],
        "Hybrid + rerank": lambda q: [c.id for c, _ in
                                      rerank(q, [c for c, _ in idx.hybrid_search(q, 8)], top_n=8)],
    }
    if azure_search.enabled():
        methods["Azure hybrid"] = lambda q: [c.id for c, _ in azure_hybrid_search(q, 8)]
        methods["Azure + rerank"] = lambda q: [c.id for c, _ in
                                               rerank(q, [c for c, _ in azure_hybrid_search(q, 8)], top_n=8)]
    results = {}
    for name, search in methods.items():
        ranks = [_rank(search(q), expected) for q, expected in GOLDEN]
        results[name] = {
            "hit@3": sum(1 for r in ranks if r and r <= 3) / len(GOLDEN),
            "mrr": sum(1 / r for r in ranks if r) / len(GOLDEN),
            "misses": [GOLDEN[i][0] for i, r in enumerate(ranks) if not r or r > 3],
        }
    return results


if __name__ == "__main__":
    results = evaluate()
    print(f"\n{'Method':18} {'Hit@3':>7} {'MRR':>7}   ({len(GOLDEN)} questions)")
    print("-" * 40)
    for name, r in results.items():
        print(f"{name:18} {r['hit@3']:7.2f} {r['mrr']:7.2f}")
    for name, r in results.items():
        for q in r["misses"]:
            print(f"  miss [{name}]: {q}")
