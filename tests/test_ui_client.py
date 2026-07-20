"""UI HTTP client — httpx is faked, no backend needed."""

from typing import Any

import pytest

from app.ui import client


class _Resp:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_ask_posts_question(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> _Resp:
        captured["url"] = url
        captured["json"] = json
        return _Resp({"answer": "hi", "sources": []})

    monkeypatch.setattr(client.httpx, "post", fake_post)

    out = client.ask("what's new?")
    assert out["answer"] == "hi"
    assert captured["url"].endswith("/ask")
    assert captured["json"] == {"question": "what's new?"}


def test_ingest_posts_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> _Resp:
        captured["json"] = json
        return _Resp({"chunks_ingested": 2})

    monkeypatch.setattr(client.httpx, "post", fake_post)

    out = client.ingest("some text", "market-wire", "http://x.com")
    assert out["chunks_ingested"] == 2
    assert captured["json"] == {
        "text": "some text",
        "source": "market-wire",
        "url": "http://x.com",
    }
