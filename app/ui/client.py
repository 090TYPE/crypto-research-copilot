"""Thin HTTP client the Streamlit UI uses to call the FastAPI backend."""

import os
from typing import Any

import httpx

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
_TIMEOUT = 180.0  # agent + local LLM can be slow


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = httpx.post(f"{API_BASE_URL}{path}", json=payload, timeout=_TIMEOUT)
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return data


def ingest(text: str, source: str, url: str | None = None) -> dict[str, Any]:
    return _post("/ingest", {"text": text, "source": source, "url": url})


def ask(question: str) -> dict[str, Any]:
    return _post("/ask", {"question": question})


def agent(question: str) -> dict[str, Any]:
    return _post("/agent", {"question": question})
