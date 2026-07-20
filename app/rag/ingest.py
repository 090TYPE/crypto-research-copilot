"""Ingest pipeline: split -> embed -> store in Chroma with metadata.

Heavy deps (chromadb, sentence-transformers, langchain splitter) are imported
lazily so the app boots without the optional `ai` extra installed.
"""

from typing import Any

from app.config import get_settings

COLLECTION_NAME = "crypto_docs"


def chunk_text(text: str) -> list[str]:
    """Split raw text into overlapping chunks (RecursiveCharacterTextSplitter)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return [c for c in splitter.split_text(text) if c.strip()]


def get_collection() -> Any:
    """Return the persistent Chroma collection, embeddings wired to settings."""
    import chromadb
    from chromadb.utils import embedding_functions

    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_path)
    # chromadb's own stubs disagree on the EmbeddingFunction generic; treat as Any.
    ef: Any = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)


def ingest_text(
    text: str,
    *,
    source: str,
    url: str | None = None,
    date: str | None = None,
) -> int:
    """Chunk `text`, embed, and persist to Chroma with metadata. Returns chunk count."""
    import uuid

    chunks = chunk_text(text)
    if not chunks:
        return 0

    collection = get_collection()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas: list[dict[str, str]] = []
    for i in range(len(chunks)):
        meta: dict[str, str] = {"source": source, "chunk": str(i)}
        if url is not None:
            meta["url"] = url
        if date is not None:
            meta["date"] = date
        metadatas.append(meta)

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)
