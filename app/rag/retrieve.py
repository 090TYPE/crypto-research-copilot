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


Hit = tuple[str, dict[str, Any], float]  # (document, metadata, distance)


def _retrieve(question: str) -> list[Hit]:
    """Query Chroma for the top-k chunks most similar to `question`."""
    settings = get_settings()
    collection = get_collection()
    res = collection.query(
        query_texts=[question],
        n_results=settings.retrieval_top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs: list[str] = res["documents"][0] if res.get("documents") else []
    metas: list[dict[str, Any]] = res["metadatas"][0] if res.get("metadatas") else []
    dists: list[float] = res["distances"][0] if res.get("distances") else []
    return list(zip(docs, metas, dists, strict=False))


def _build_context(hits: list[Hit]) -> str:
    lines = []
    for doc, meta, _dist in hits:
        label = meta.get("source", "unknown")
        lines.append(f"[{label}] {doc}")
    return "\n\n".join(lines)


def _confidence(hits: list[Hit]) -> float:
    """Best-hit similarity in [0, 1], from Chroma L2 distance (1/(1+d))."""
    if not hits:
        return 0.0
    best_distance = min(dist for _doc, _meta, dist in hits)
    return round(1.0 / (1.0 + best_distance), 3)


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
        for doc, meta, _dist in hits
    ]
    return AskResponse(answer=answer, sources=sources, confidence=_confidence(hits))
