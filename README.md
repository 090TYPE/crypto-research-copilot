# Crypto Research Copilot

[![CI](https://github.com/090TYPE/crypto-research-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/090TYPE/crypto-research-copilot/actions/workflows/ci.yml)

RAG + agent copilot for crypto research. Ingest news/docs, ask questions and get
answers **with source citations** (RAG), and let an **agent** call tools
(`get_price`, `calc_indicator`) when a question needs live market data.

Runs **fully local and free** on [Ollama](https://ollama.com) — no API key. The
LLM provider sits behind a single `LLMProvider` interface, so switching to
OpenAI/Claude is a one-line `.env` change.

## Stack

FastAPI · LangChain (agents) · Chroma · sentence-transformers · Ollama (llama3.1) ·
httpx · uv · ruff · mypy · pytest · Docker

## Architecture

```
            ┌──────────────── FastAPI ────────────────┐
 client ──▶ │ /ask-raw  /ingest  /ask  /agent  /health │
            └───┬─────────────┬──────────────┬─────────┘
                │             │              │
            LLMProvider   RAG pipeline    Agent (LangChain)
            (Ollama /     split→embed     tools: get_price,
             OpenAI)      →Chroma;        calc_indicator (httpx→CoinGecko),
                │         retrieve→cite   search_news (→RAG)
                └─────────────┴──────────────┘
                          Ollama (llama3.1)
```

## Quick start (local)

```bash
# 1. deps
uv sync --extra ai

# 2. local model (once)
ollama pull llama3.1

# 3. run
uv run uvicorn app.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

Config: copy `.env.example` → `.env` (defaults to local Ollama, no key needed).

## Run with Docker

```bash
docker compose up          # app + ollama; the model is auto-pulled on first boot
```

First boot downloads llama3.1 (~4.9 GB) into the ollama volume; the app waits
until the model is ready (compose healthcheck), then serves on `:8000`.

## Demo

**RAG with citations** — ingest, then ask:

```bash
curl -X POST localhost:8000/ingest -H "content-type: application/json" -d '{
  "text": "Bitcoin surged past $95,000 on strong ETF inflows this week.",
  "source": "market-wire", "url": "http://ex.com/btc"
}'
# -> {"chunks_ingested": 1}

curl -X POST localhost:8000/ask -H "content-type: application/json" -d '{
  "question": "What happened with Bitcoin'\''s price recently?"
}'
```
```json
{
  "answer": "Bitcoin's price surged past $95,000. [market-wire]",
  "sources": [{"text": "Bitcoin surged past $95,000 ...", "source": "market-wire", "url": "http://ex.com/btc"}],
  "tools_used": [],
  "confidence": 0.62
}
```

**Agent picks tools** — combines a live price with the news store:

```bash
curl -X POST localhost:8000/agent -H "content-type: application/json" -d '{
  "question": "What is the current price of bitcoin, and any recent news?"
}'
```
```json
{
  "answer": "The current price of bitcoin is $65,370.00 USD. Recent news: Bitcoin ETFs saw record inflows ... [market-wire]",
  "sources": [{"source": "market-wire", "...": "..."}],
  "tools_used": ["get_price", "search_news"],
  "confidence": null
}
```

## Endpoints

| Endpoint | What |
|---|---|
| `GET /health` | liveness |
| `POST /ask-raw` | raw LLM, no RAG |
| `POST /ingest` | chunk + embed + store |
| `POST /ask` | RAG answer with citations + confidence |
| `POST /agent` | agent chooses get_price / calc_indicator / search_news |

Errors are structured: bad input → 422/400, dead upstream (CoinGecko) → 502,
never a leaked traceback.

## Evaluation

A lightweight RAG eval harness (LLM-as-judge + embedding metrics, no external
eval framework) scores the pipeline on a curated set:

```bash
uv run python -m app.evaluation.evaluate
```
```
      faithfulness: 0.9     # answer grounded in retrieved context (LLM judge)
  answer_relevancy: 0.833   # answer addresses the question (LLM judge)
 answer_similarity: 0.719   # cosine(answer, ground truth) via embeddings
  context_hit_rate: 1.0     # retrieval surfaced the expected source
```

The judge is the configured `LLMProvider`, so the same harness works against
Ollama or a cloud model.

## Tests / lint

```bash
uv run pytest         # 25 tests (unit + Chroma integration; LLM/network mocked)
uv run ruff check .
uv run mypy app
```

CI runs all three on every push/PR.

## Layout

```
app/
  main.py        FastAPI routes
  config.py      pydantic-settings, provider selection
  schemas.py     request/response models
  errors.py      global exception handlers
  llm/           LLMProvider interface + ollama/openai
  rag/           ingest (split+embed+store) + retrieve (cited answers)
  agent/         tools (price/indicator) + indicators (RSI/SMA) + runner
  sources/       news loader
tests/
```

Portfolio project — a Python reimplementation of a C# AI-desk concept
(FastAPI + LangChain + Chroma), with the LLM provider abstracted so it runs
locally on Ollama or on a cloud API.
