"""Global exception handlers — upstream failure -> 502, bad value -> 400."""

import httpx
from fastapi.testclient import TestClient

from app.llm.base import get_provider
from app.main import app


class _BoomProvider:
    name = "boom"

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def complete(self, prompt: str) -> str:
        raise self._exc


def _post_with_provider(exc: Exception) -> httpx.Response:
    app.dependency_overrides[get_provider] = lambda: _BoomProvider(exc)
    try:
        client = TestClient(app)
        return client.post("/ask-raw", json={"prompt": "hi"})
    finally:
        app.dependency_overrides.clear()


def test_upstream_httpx_error_maps_to_502() -> None:
    resp = _post_with_provider(httpx.ConnectError("connection refused"))
    assert resp.status_code == 502
    assert resp.json()["detail"] == "Upstream data source unavailable"


def test_value_error_maps_to_400() -> None:
    resp = _post_with_provider(ValueError("bad symbol"))
    assert resp.status_code == 400
    assert resp.json()["detail"] == "bad symbol"
