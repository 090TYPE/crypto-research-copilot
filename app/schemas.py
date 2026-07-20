"""Pydantic request/response schemas — structured I/O for every endpoint."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class AskRawRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Raw prompt sent straight to the LLM")


class AskRawResponse(BaseModel):
    answer: str
    provider: str


# --- Filled in later stages (ingest / RAG / agent) ---


class Source(BaseModel):
    text: str
    source: str | None = None
    url: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    confidence: float | None = None
