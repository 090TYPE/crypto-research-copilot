"""Agent orchestration — the LLM decides between RAG and live tools.

Uses a LangChain tool-calling agent over ChatOllama. Tools:
  - get_price / calc_indicator  (live CoinGecko data)
  - search_news                 (RAG over the Chroma store)

Note: the agent path binds to Ollama directly (tool-calling needs a chat model
that supports it), so it does not go through the generic LLMProvider seam.
"""

from typing import Any

from app.agent.tools import calc_indicator, get_price
from app.config import get_settings
from app.rag.retrieve import _build_context, _retrieve
from app.schemas import AskResponse, Source

_SYSTEM = (
    "You are a crypto research assistant. Use the tools to fetch live prices, "
    "compute indicators, or search ingested news when the question needs them. "
    "Prefer tools over guessing. Cite news sources in your answer."
)


async def run_agent(question: str) -> AskResponse:
    from langchain.agents import create_agent
    from langchain_core.tools import StructuredTool
    from langchain_ollama import ChatOllama

    settings = get_settings()
    collected: list[Source] = []

    async def price_tool(symbol: str) -> str:
        """Get the current USD price for a CoinGecko coin id (e.g. 'bitcoin')."""
        return f"{symbol} price: ${await get_price(symbol):,.2f} USD"

    async def indicator_tool(symbol: str, kind: str = "rsi", period: int = 14) -> str:
        """Compute a technical indicator ('rsi' or 'sma') for a CoinGecko coin id."""
        value = await calc_indicator(symbol, kind=kind, period=period)
        return f"{kind.upper()}({period}) for {symbol}: {value:.2f}"

    async def news_tool(query: str) -> str:
        """Search ingested crypto news/docs for context relevant to the query."""
        hits = _retrieve(query)
        for doc, meta in hits:
            collected.append(Source(text=doc, source=meta.get("source"), url=meta.get("url")))
        return _build_context(hits) or "No relevant documents found."

    tools = [
        StructuredTool.from_function(coroutine=price_tool, name="get_price"),
        StructuredTool.from_function(coroutine=indicator_tool, name="calc_indicator"),
        StructuredTool.from_function(coroutine=news_tool, name="search_news"),
    ]

    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )
    graph = create_agent(llm, tools, system_prompt=_SYSTEM)

    result: dict[str, Any] = await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    messages = result["messages"]

    tools_used: list[str] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            tools_used.append(call["name"])

    answer = messages[-1].content if messages else ""
    return AskResponse(answer=str(answer), sources=collected, tools_used=tools_used)
