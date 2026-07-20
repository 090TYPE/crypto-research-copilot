"""Agent tools: live price + technical indicators via CoinGecko (httpx, no key).

`symbol` is a CoinGecko coin id, e.g. "bitcoin", "ethereum".
Pure math lives in `indicators.py`; here we fetch data and compute.
"""

import httpx

from app.agent.indicators import rsi, sma

_BASE = "https://api.coingecko.com/api/v3"
_TIMEOUT = 15.0


async def get_price(symbol: str, *, vs: str = "usd") -> float:
    """Spot price for a CoinGecko coin id."""
    params = {"ids": symbol, "vs_currencies": vs}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_BASE}/simple/price", params=params)
        resp.raise_for_status()
        data = resp.json()
    if symbol not in data:
        raise ValueError(f"unknown coin id: {symbol!r}")
    return float(data[symbol][vs])


async def get_price_history(symbol: str, *, days: int = 30, vs: str = "usd") -> list[float]:
    """Daily closing prices for the last `days` days (oldest -> newest)."""
    params = {"vs_currency": vs, "days": str(days), "interval": "daily"}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_BASE}/coins/{symbol}/market_chart", params=params)
        resp.raise_for_status()
        data = resp.json()
    # market_chart returns prices as [[timestamp_ms, price], ...]
    return [float(point[1]) for point in data.get("prices", [])]


async def calc_indicator(symbol: str, *, kind: str = "rsi", period: int = 14) -> float:
    """Fetch history and compute an indicator. kind: 'rsi' or 'sma'."""
    kind = kind.lower()
    days = max(period * 2, 30)
    prices = await get_price_history(symbol, days=days)
    if kind == "sma":
        return sma(prices, period)
    if kind == "rsi":
        return rsi(prices, period)
    raise ValueError(f"unknown indicator: {kind!r} (use 'rsi' or 'sma')")
