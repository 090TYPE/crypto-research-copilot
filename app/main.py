"""FastAPI entrypoint. Endpoints grow per stage (0: health, 1: /ask-raw)."""

from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.concurrency import run_in_threadpool

from app.llm.base import LLMProvider, get_provider
from app.schemas import (
    AskRawRequest,
    AskRawResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)

app = FastAPI(
    title="Crypto Research Copilot",
    description="RAG + agent copilot for crypto research.",
    version="0.1.0",
)

# DI seam: routes depend on the LLMProvider interface, not a concrete backend.
ProviderDep = Annotated[LLMProvider, Depends(get_provider)]


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/ask-raw", response_model=AskRawResponse)
async def ask_raw(req: AskRawRequest, provider: ProviderDep) -> AskRawResponse:
    """Ask the LLM directly, no RAG — smoke-test of the provider wiring."""
    answer = await provider.complete(req.prompt)
    return AskRawResponse(answer=answer, provider=provider.name)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    """Chunk + embed + store a document in Chroma. Returns chunk count."""
    from app.rag.ingest import ingest_text

    # Chroma/sentence-transformers are sync + CPU-bound: keep off the event loop.
    count = await run_in_threadpool(
        ingest_text, req.text, source=req.source, url=req.url, date=req.date
    )
    return IngestResponse(chunks_ingested=count)


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, provider: ProviderDep) -> AskResponse:
    """RAG answer: retrieve top-k context, then answer grounded with citations."""
    from app.rag.retrieve import answer_question

    return await answer_question(req.question, provider=provider)
