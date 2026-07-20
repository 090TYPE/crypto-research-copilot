"""Ingest pipeline: split -> embed -> store in Chroma with metadata.

Implemented in Stage 2. Requires the optional `ai` dependency group.
"""


def ingest_text(text: str, *, source: str, url: str | None = None) -> int:
    """Chunk `text`, embed, and persist to Chroma. Returns chunk count."""
    raise NotImplementedError("Stage 2: chunk + embed + Chroma write")
