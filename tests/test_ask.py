"""Stage 3 RAG tests. Retrieval hits real Chroma (temp dir); the LLM is faked,
so answers are deterministic and we can assert the prompt was grounded."""

import pytest
from fastapi.testclient import TestClient

from app.llm.base import get_provider
from app.main import app
from app.rag.ingest import ingest_text
from app.rag.retrieve import answer_question
from tests.conftest import FakeProvider

pytest.importorskip("chromadb")


@pytest.mark.usefixtures("temp_chroma")
async def test_answer_grounds_prompt_and_returns_sources() -> None:
    ingest_text(
        "The Zephyr protocol launched staking with a 5% yield.",
        source="zephyr-news",
        url="http://example.com/zephyr",
    )
    fake = FakeProvider()

    resp = await answer_question("What yield does Zephyr staking offer?", provider=fake)

    assert resp.answer == "FAKE ANSWER"
    assert resp.sources
    assert resp.sources[0].source == "zephyr-news"
    # Grounding: retrieved context must be injected into the LLM prompt.
    assert fake.last_prompt is not None
    assert "5% yield" in fake.last_prompt


@pytest.mark.usefixtures("temp_chroma")
async def test_empty_store_returns_dont_know_without_calling_llm() -> None:
    fake = FakeProvider()
    resp = await answer_question("anything?", provider=fake)

    assert "don't know" in resp.answer.lower()
    assert resp.sources == []
    assert fake.last_prompt is None  # LLM not called when no context


@pytest.mark.usefixtures("temp_chroma")
def test_ask_route_returns_answer_and_sources() -> None:
    ingest_text("Bitcoin ETF inflows hit a record.", source="btc-news")
    app.dependency_overrides[get_provider] = lambda: FakeProvider()
    try:
        client = TestClient(app)
        resp = client.post("/ask", json={"question": "What about Bitcoin ETF?"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "FAKE ANSWER"
    assert body["sources"][0]["source"] == "btc-news"
