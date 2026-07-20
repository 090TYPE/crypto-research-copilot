"""Application settings — read from environment / .env via pydantic-settings.

Provider abstraction lives here: switch LLM backend with a single env var.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider selection
    llm_provider: Literal["ollama", "openai"] = "ollama"

    # Ollama (local, default — no API key needed)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # OpenAI (optional — only used when llm_provider="openai")
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # RAG / vector store
    chroma_path: str = "./data/chroma"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_top_k: int = 4


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — usable as a FastAPI dependency."""
    return Settings()
