"""/ask-raw route — provider mocked via FastAPI dependency override (no network)."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.llm.base import get_provider
from app.main import app


class FakeProvider:
    name = "fake"

    async def complete(self, prompt: str) -> str:
        return f"echo: {prompt}"


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_provider] = lambda: FakeProvider()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_ask_raw_returns_answer_and_provider(client: TestClient) -> None:
    resp = client.post("/ask-raw", json={"prompt": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"answer": "echo: hello", "provider": "fake"}


def test_ask_raw_rejects_empty_prompt(client: TestClient) -> None:
    resp = client.post("/ask-raw", json={"prompt": ""})
    assert resp.status_code == 422
