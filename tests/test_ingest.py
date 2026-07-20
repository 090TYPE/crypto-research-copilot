"""Stage 2 ingest tests. Integration test hits real Chroma in a temp dir
(downloads the embedding model on first run) — needs the `ai` extra installed."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from app.config import get_settings
from app.rag.ingest import chunk_text, get_collection, ingest_text

pytest.importorskip("chromadb")
pytest.importorskip("langchain_text_splitters")


def test_chunk_text_splits_and_drops_blanks() -> None:
    text = "Bitcoin news. " * 300  # long enough to force multiple chunks
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunk_text_empty() -> None:
    assert chunk_text("   ") == []


@pytest.fixture
def temp_chroma(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("temp_chroma")
def test_ingest_writes_vectors_with_metadata() -> None:
    text = "Ethereum upgrade shipped. " * 100
    count = ingest_text(text, source="test-news", url="http://example.com", date="2026-07-20")
    assert count > 0

    collection = get_collection()
    assert collection.count() == count

    got = collection.get(limit=1, include=["metadatas"])
    meta = got["metadatas"][0]
    assert meta["source"] == "test-news"
    assert meta["url"] == "http://example.com"
