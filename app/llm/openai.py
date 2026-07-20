"""OpenAI provider — optional cloud backend. Stub until Stage 1+ wires the SDK."""

from app.config import get_settings


class OpenAIProvider:
    name = "openai"

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self._api_key = settings.openai_api_key
        self._model = settings.openai_model

    async def complete(self, prompt: str) -> str:
        # Implemented in a later stage using the OpenAI SDK / chat completions.
        raise NotImplementedError("OpenAI provider not wired yet — use LLM_PROVIDER=ollama")
