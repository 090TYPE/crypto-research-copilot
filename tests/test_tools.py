"""Agent tools — CoinGecko fetch is faked; no real network."""

from typing import Any

import pytest

from app.agent import tools


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get(self, url: str, params: dict[str, Any] | None = None) -> _FakeResponse:
        return _FakeResponse(self._payload)


def _patch_httpx(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    monkeypatch.setattr(tools.httpx, "AsyncClient", lambda **kw: _FakeClient(payload))


async def test_get_price_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_httpx(monkeypatch, {"bitcoin": {"usd": 95000.5}})
    assert await tools.get_price("bitcoin") == 95000.5


async def test_get_price_unknown_coin(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_httpx(monkeypatch, {})
    with pytest.raises(ValueError):
        await tools.get_price("not-a-coin")


async def test_calc_indicator_sma(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_history(symbol: str, *, days: int = 30, vs: str = "usd") -> list[float]:
        return [float(i) for i in range(1, 31)]

    monkeypatch.setattr(tools, "get_price_history", fake_history)
    # SMA(3) over [..., 28, 29, 30] = 29
    assert await tools.calc_indicator("bitcoin", kind="sma", period=3) == 29.0


async def test_calc_indicator_bad_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_history(symbol: str, *, days: int = 30, vs: str = "usd") -> list[float]:
        return [float(i) for i in range(1, 31)]

    monkeypatch.setattr(tools, "get_price_history", fake_history)
    with pytest.raises(ValueError):
        await tools.calc_indicator("bitcoin", kind="macd")
