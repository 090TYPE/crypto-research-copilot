# Slim image; uv handles deps. Heavy AI stack installed via the `ai` extra.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install deps first for layer caching.
COPY pyproject.toml uv.lock* ./
RUN uv sync --extra ai --no-dev --frozen || uv sync --extra ai --no-dev

COPY app ./app

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
