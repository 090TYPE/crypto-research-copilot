"""Ollama provider — local model over HTTP. Default backend (no API key)."""

import httpx

from app.config import get_settings


class OllamaProvider:
    name = "ollama"

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model

    async def complete(self, prompt: str) -> str:
        payload = {"model": self._model, "prompt": prompt, "stream": False}
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self._base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return str(data.get("response", "")).strip()
