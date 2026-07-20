"""Pydantic request/response schemas — structured I/O for every endpoint."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class AskRawRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Raw prompt sent straight to the LLM")


class AskRawResponse(BaseModel):
    answer: str
    provider: str


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Document/news body to ingest")
    source: str = Field(..., min_length=1, description="Human-readable source name")
    url: str | None = None
    date: str | None = None


class IngestResponse(BaseModel):
    chunks_ingested: int


# --- Filled in later stages (RAG / agent) ---


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer from the RAG store")


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question for the tool-using agent")


class Source(BaseModel):
    text: str
    source: str | None = None
    url: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    confidence: float | None = None
