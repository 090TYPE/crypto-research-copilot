"""Agent tools: live price + technical indicators.

`get_price` hits CoinGecko via httpx. `calc_indicator` reuses indicator logic
(RSI/SMA) ported from StockAnalyzer. Wired into the agent in Stage 4.
"""

import httpx

COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"


async def get_price(symbol: str, *, vs: str = "usd") -> float:
    """Return spot price for a CoinGecko coin id (e.g. 'bitcoin', 'ethereum')."""
    params = {"ids": symbol, "vs_currencies": vs}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(COINGECKO_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    return float(data[symbol][vs])


def calc_indicator(prices: list[float], *, kind: str = "sma", period: int = 14) -> float:
    """Compute a simple indicator over a price series. Expanded in Stage 4."""
    raise NotImplementedError("Stage 4: RSI/SMA indicators")
