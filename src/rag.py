"""Hybrid RAG over the policy docs: chunk -> BM25 + vectors -> RRF fusion -> LLM rerank -> grounded answer.
CLI:  python -m src.rag build
      python -m src.rag search "Do you ship lithium batteries?"
      python -m src.rag ask "What is the demurrage for a 40ft container on day 7?" """
import json
import re
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from src import azure_search, config
from src.llm import get_chat_model, get_embeddings

CHUNKS_FILE = config.INDEX_DIR / "chunks.json"
VECTORS_FILE = config.INDEX_DIR / "vectors.npy"


# ============================== Step 11: CHUNKING ==============================
@dataclass
class Chunk:
    id: str       # e.g. "demurrage_and_detention#2"
    source: str   # file name
    title: str    # document title (# heading)
    section: str  # section name (## heading)
    text: str     # title + section + body -> makes sense on its own


def chunk_markdown() -> list[Chunk]:
    """Structure-aware chunking: one chunk per '## ' section, doc title kept in every chunk."""
    chunks = []
    for path in sorted(config.KNOWLEDGE_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        title = content.splitlines()[0].lstrip("# ").strip()
        for i, part in enumerate(content.split("\n## ")[1:]):  # [0] is the title block
            section, _, body = part.partition("\n")
            chunks.append(Chunk(
                id=f"{path.stem}#{i}", source=path.name, title=title, section=section.strip(),
                text=f"{title}\n## {section.strip()}\n{body.strip()}",
            ))
    return chunks


# ============================== Step 12: BM25 + VECTOR INDEX ==============================
def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)  # length 1 -> dot product == cosine


def build_index() -> int:
    chunks = chunk_markdown()
    vectors = np.array(get_embeddings().embed_documents([c.text for c in chunks]), dtype=np.float32)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_FILE.write_text(json.dumps([asdict(c) for c in chunks], indent=2), encoding="utf-8")
    np.save(VECTORS_FILE, _normalise(vectors))  # normalise ONCE at build time
    get_index.cache_clear()
    if azure_search.enabled():  # also push the SAME chunks + vectors to Azure AI Search
        azure_search.ensure_index()
        print("uploaded to Azure AI Search:", azure_search.upload_chunks(
            [asdict(c) for c in chunks], _normalise(vectors).tolist()))
    return len(chunks)


@lru_cache(maxsize=256)
def _embed_query(q: str) -> np.ndarray:  # cached: same question is embedded only once
    return _normalise(np.array(get_embeddings().embed_query(q), dtype=np.float32))


class LocalIndex:
    def __init__(self):
        self.chunks = [Chunk(**c) for c in json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))]
        self.by_id = {c.id: c for c in self.chunks}
        self.vectors = np.load(VECTORS_FILE)
        self.bm25 = BM25Okapi([_tokens(c.text) for c in self.chunks])

    def bm25_search(self, q: str, k: int = 8) -> list[tuple[Chunk, float]]:
        scores = self.bm25.get_scores(_tokens(q))  # keyword match score per chunk
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]  # 0 = no shared words

    def vector_search(self, q: str, k: int = 8) -> list[tuple[Chunk, float]]:
        scores = self.vectors @ _embed_query(q)  # cosine similarity with every chunk
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    # ---------- Step 13 ----------
    def hybrid_search(self, q: str, k: int = 8) -> list[tuple[Chunk, float]]:
        fused = rrf([[c.id for c, _ in self.bm25_search(q, k)], [c.id for c, _ in self.vector_search(q, k)]])
        return [(self.by_id[cid], score) for cid, score in fused[:k]]


@lru_cache
def get_index() -> LocalIndex:
    if not CHUNKS_FILE.exists():
        build_index()
    return LocalIndex()


# ============================== Step 13: RRF + RERANK + ANSWER ==============================
def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score = sum of 1/(k + rank). Uses ranks only, so no score scaling needed."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class _Score(BaseModel):
    chunk_id: str
    score: int = Field(ge=0, le=10, description="0 = irrelevant, 10 = directly answers the question")


class _Scores(BaseModel):
    scores: list[_Score]


def rerank(question: str, candidates: list[Chunk], top_n: int = 3) -> list[tuple[Chunk, int]]:
    """Second, smarter pass: the LLM reads each candidate and scores how well it answers the question."""
    listing = "\n\n".join(f"[{c.id}]\n{c.text}" for c in candidates)
    prompt = (f"Score how well each passage answers the question (0-10). Score every passage.\n\n"
              f"Question: {question}\n\nPassages:\n{listing}")
    result = get_chat_model().with_structured_output(_Scores).invoke(prompt)  # typed output, no JSON parsing
    score_of = {s.chunk_id: s.score for s in result.scores}
    ranked = sorted(candidates, key=lambda c: score_of.get(c.id, 0), reverse=True)  # stable: ties keep RRF order
    return [(c, score_of.get(c.id, 0)) for c in ranked[:top_n]]


def azure_hybrid_search(question: str, k: int = 8) -> list[tuple[Chunk, float]]:
    hits = azure_search.hybrid_search(question, _embed_query(question).tolist(), k)
    return [(Chunk(**{f: h[f] for f in ("id", "source", "title", "section", "text")}), h["score"]) for h in hits]


def retrieve(question: str, k: int = 8, top_n: int = 3) -> list[tuple[Chunk, int]]:
    # Same pipeline either way: hybrid candidates -> our LLM rerank. Only the search backend changes.
    hits = azure_hybrid_search(question, k) if azure_search.enabled() else get_index().hybrid_search(question, k)
    return rerank(question, [c for c, _ in hits], top_n)


def answer(question: str) -> dict:
    hits = retrieve(question)
    context = "\n\n".join(f"[{c.id}]\n{c.text}" for c, _ in hits)
    prompt = (
        "You are a freight-forwarding policy assistant. Answer ONLY from the context below.\n"
        "Cite the chunk id in square brackets after each fact, e.g. [demurrage_and_detention#2].\n"
        "Show calculations step by step. If the context does not contain the answer, reply exactly:\n"
        "\"I could not find this in the company policies.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    text = get_chat_model().invoke(prompt).content
    return {"answer": text, "sources": [{"id": c.id, "rerank_score": s} for c, s in hits]}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    q = " ".join(sys.argv[2:])
    if cmd == "chunks":
        for c in chunk_markdown():
            print(f"{c.id:35} {c.section}")
    elif cmd == "build":
        print("indexed chunks:", build_index())
    elif cmd == "search":
        idx = get_index()
        searches = [("BM25", idx.bm25_search(q, 3)), ("VECTOR", idx.vector_search(q, 3)),
                    ("HYBRID (RRF)", idx.hybrid_search(q, 3))]
        if azure_search.enabled():
            searches.append(("AZURE AI SEARCH HYBRID", azure_hybrid_search(q, 3)))
        for name, hits in searches:
            print(f"\n{name}:")
            for c, s in hits:
                print(f"  {s:8.4f}  {c.id:35} {c.section}")
    elif cmd == "ask":
        print(json.dumps(answer(q), indent=2))
    else:
        print(__doc__)
