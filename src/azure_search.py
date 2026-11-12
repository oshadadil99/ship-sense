"""Azure AI Search as the managed vector + keyword store. Hybrid search (BM25 + HNSW vectors + RRF)
runs inside Azure in ONE call. Free tier has no semantic ranker -> we keep our own LLM reranker."""
from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration, SearchableField, SearchField, SearchFieldDataType,
    SearchIndex, SimpleField, VectorSearch, VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from src import config

DIMS = 1536  # text-embedding-3-small vector size


def enabled() -> bool:
    return bool(config.AZURE_SEARCH_ENDPOINT and config.AZURE_SEARCH_KEY)


def _cred() -> AzureKeyCredential:
    return AzureKeyCredential(config.AZURE_SEARCH_KEY)  # production: Managed Identity instead of a key


@lru_cache
def _search_client() -> SearchClient:
    return SearchClient(config.AZURE_SEARCH_ENDPOINT, config.AZURE_SEARCH_INDEX, _cred())


def ensure_index() -> None:
    """Create or update the index schema (like CREATE TABLE)."""
    fields = [
        SimpleField(name="key", type=SearchFieldDataType.String, key=True),  # '#' not allowed in keys
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="section", type=SearchFieldDataType.String),
        SearchableField(name="text", type=SearchFieldDataType.String),        # BM25 runs on searchable fields
        SearchField(
            name="vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True, vector_search_dimensions=DIMS, vector_search_profile_name="hnsw-profile",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],  # HNSW = fast approximate nearest neighbours
        profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw")],
    )
    index = SearchIndex(name=config.AZURE_SEARCH_INDEX, fields=fields, vector_search=vector_search)
    SearchIndexClient(config.AZURE_SEARCH_ENDPOINT, _cred()).create_or_update_index(index)


def upload_chunks(chunks: list[dict], vectors: list[list[float]]) -> int:
    docs = [{"key": c["id"].replace("#", "-"), "chunk_id": c["id"], "source": c["source"], "title": c["title"],
             "section": c["section"], "text": c["text"], "vector": v} for c, v in zip(chunks, vectors)]
    results = _search_client().merge_or_upload_documents(docs)  # upsert: safe to run again
    return sum(r.succeeded for r in results)


def hybrid_search(question: str, vector: list[float], k: int = 8) -> list[dict]:
    """ONE request: BM25 on text + vector search, fused by Azure with RRF."""
    results = _search_client().search(
        search_text=question,                                                        # keyword part
        vector_queries=[VectorizedQuery(vector=vector, k_nearest_neighbors=k, fields="vector")],  # vector part
        select=["chunk_id", "source", "title", "section", "text"],
        top=k,
    )
    return [{"id": r["chunk_id"], "source": r["source"], "title": r["title"], "section": r["section"],
             "text": r["text"], "score": r["@search.score"]} for r in results]
