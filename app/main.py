"""FastAPI entrypoint. Endpoints grow per stage (0: health, 1: /ask-raw)."""

from typing import Annotated

from fastapi import Depends, FastAPI

from app.llm.base import LLMProvider, get_provider
from app.schemas import AskRawRequest, AskRawResponse, HealthResponse

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
