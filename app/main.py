"""FastAPI entrypoint. Stage 0: health check. Endpoints grow per stage."""

from fastapi import FastAPI

from app.schemas import HealthResponse

app = FastAPI(
    title="Crypto Research Copilot",
    description="RAG + agent copilot for crypto research.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")
