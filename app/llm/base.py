"""LLM provider interface — the abstraction that lets us swap backends via .env.

Analogous to an `IExchangeGateway` seam: callers depend on `LLMProvider`,
not on Ollama or OpenAI concretely.
"""

from typing import Protocol


class LLMProvider(Protocol):
    name: str

    async def complete(self, prompt: str) -> str:
        """Return a completion for a single prompt."""
        ...


def get_provider() -> LLMProvider:
    """Factory: build the provider selected in settings.

    Imports are local to keep optional/cloud deps off the hot import path.
    """
    from app.config import get_settings

    settings = get_settings()
    if settings.llm_provider == "ollama":
        from app.llm.ollama import OllamaProvider

        return OllamaProvider()
    if settings.llm_provider == "openai":
        from app.llm.openai import OpenAIProvider

        return OpenAIProvider()
    raise ValueError(f"Unknown llm_provider: {settings.llm_provider}")
