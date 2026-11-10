"""One factory for the LLM and embeddings. Every other module calls these functions,
so switching model / provider is a config change, not a code change."""
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src import config

# Azure OpenAI "v1" endpoint is OpenAI-compatible -> normal OpenAI clients, no api_version.
BASE_URL = f"{config.AZURE_OPENAI_ENDPOINT}/openai/v1/"


@lru_cache  # create the client once, reuse it everywhere
def get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.AZURE_OPENAI_CHAT_DEPLOYMENT,  # Azure: the DEPLOYMENT name, not the model name
        base_url=BASE_URL,
        api_key=config.AZURE_OPENAI_API_KEY,
        temperature=0,   # same question -> same answer (repeatable, testable)
        timeout=60,
        max_retries=3,   # retries 429 (TPM limit hit) and short network errors
    )


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.AZURE_OPENAI_EMBED_DEPLOYMENT,
        base_url=BASE_URL,
        api_key=config.AZURE_OPENAI_API_KEY,
        check_embedding_ctx_length=False,  # send plain text; Azure deployment names break tiktoken lookup
    )


if __name__ == "__main__":
    print("chat :", get_chat_model().invoke("Say hi in 3 words").content)
    print("embed dims:", len(get_embeddings().embed_query("container delayed at Colombo port")))
