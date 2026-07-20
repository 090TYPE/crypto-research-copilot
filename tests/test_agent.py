"""/agent route — the LangChain agent is stubbed (no LLM/network).

The live agent is exercised manually; here we verify the route wiring and
response shape deterministically.
"""

import pytest
from fastapi.testclient import TestClient

import app.agent.runner as runner
from app.main import app
from app.schemas import AskResponse, Source


def test_agent_route_returns_answer_tools_and_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_agent(question: str) -> AskResponse:
        return AskResponse(
            answer="ETH is $3,000; news says the Pectra upgrade shipped.",
            sources=[Source(text="Pectra shipped.", source="eth-daily")],
            tools_used=["get_price", "search_news"],
        )

    monkeypatch.setattr(runner, "run_agent", fake_run_agent)

    client = TestClient(app)
    resp = client.post("/agent", json={"question": "price of ETH and any news?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["tools_used"] == ["get_price", "search_news"]
    assert body["sources"][0]["source"] == "eth-daily"


def test_agent_route_rejects_empty_question() -> None:
    client = TestClient(app)
    resp = client.post("/agent", json={"question": ""})
    assert resp.status_code == 422
