# Crypto Research Copilot

RAG + agent copilot for crypto research. Ingest news/docs, ask questions and get
answers **with source citations** (RAG), and let an **agent** call tools
(`get_price`, `calc_indicator`) when the question needs live data.

Runs **fully local and free** on [Ollama](https://ollama.com); switch to
OpenAI/Claude with one line in `.env`. The LLM provider is abstracted behind a
single `LLMProvider` interface.

## Stack

FastAPI · LangChain · Chroma · sentence-transformers · Ollama · httpx · uv ·
ruff · mypy · pytest · Docker

## Quick start (local)

```bash
uv sync                       # core deps
uv run uvicorn app.main:app --reload
# open http://localhost:8000/docs  (Swagger UI)
# GET /health -> {"status":"ok"}
```

AI stack (from Stage 2 onward):

```bash
uv sync --extra ai
```

Config: copy `.env.example` to `.env`. Defaults to local Ollama, no key needed.
For the model: install Ollama, then `ollama pull llama3.1`.

## Run with Docker

```bash
docker compose up            # app + ollama
```

## Tests / lint

```bash
uv run pytest
uv run ruff check .
uv run mypy app
```

## Endpoints (built stage by stage)

| Endpoint | Status | What |
|---|---|---|
| `GET /health` | ✅ | liveness |
| `POST /ask-raw` | ✅ | raw LLM, no RAG |
| `POST /ingest` | ✅ | chunk + embed + store |
| `POST /ask` | ✅ | RAG answer with citations |
| `POST /agent` | ⏳ | agent + tools |

## Layout

```
app/
  main.py        FastAPI routes
  config.py      pydantic-settings, provider selection
  schemas.py     request/response models
  llm/           LLMProvider interface + ollama/openai
  rag/           ingest (split+embed+store) + retrieve (cited answers)
  agent/         tools (price/indicator) + runner (orchestration)
  sources/       news loader
tests/
```

Portfolio project — Python reimplementation of a C# AI-desk concept.
