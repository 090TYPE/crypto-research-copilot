"""Retrieval + grounded answer assembly with citations.

question -> embed -> top-k from Chroma -> context -> grounded prompt -> LLM.
The prompt instructs the model to answer only from context, curbing hallucination.
"""

from typing import Any

from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.llm.base import LLMProvider, get_provider
from app.rag.ingest import get_collection
from app.schemas import AskResponse, Source

_PROMPT_TEMPLATE = """You are a crypto research assistant. Answer the question using \
ONLY the context below. If the context does not contain the answer, say you don't know. \
Cite sources inline as [source].

Context:
{context}

Question: {question}

Answer:"""


def _retrieve(question: str) -> list[tuple[str, dict[str, Any]]]:
    """Query Chroma for the top-k chunks most similar to `question`."""
    settings = get_settings()
    collection = get_collection()
    res = collection.query(
        query_texts=[question],
        n_results=settings.retrieval_top_k,
        include=["documents", "metadatas"],
    )
    docs: list[str] = res["documents"][0] if res.get("documents") else []
    metas: list[dict[str, Any]] = res["metadatas"][0] if res.get("metadatas") else []
    return list(zip(docs, metas, strict=False))


def _build_context(hits: list[tuple[str, dict[str, Any]]]) -> str:
    lines = []
    for doc, meta in hits:
        label = meta.get("source", "unknown")
        lines.append(f"[{label}] {doc}")
    return "\n\n".join(lines)


async def answer_question(question: str, *, provider: LLMProvider | None = None) -> AskResponse:
    """Retrieve context and produce a cited answer grounded in that context."""
    provider = provider or get_provider()

    hits = await run_in_threadpool(_retrieve, question)
    if not hits:
        return AskResponse(answer="I don't know — no relevant context found.", sources=[])

    prompt = _PROMPT_TEMPLATE.format(context=_build_context(hits), question=question)
    answer = await provider.complete(prompt)

    sources = [
        Source(text=doc, source=meta.get("source"), url=meta.get("url"))
        for doc, meta in hits
    ]
    return AskResponse(answer=answer, sources=sources)
