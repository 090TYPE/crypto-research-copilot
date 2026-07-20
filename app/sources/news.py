"""News loader — pull crypto headlines/articles (RSS/API) for ingestion.

Implemented in a later stage; feeds text into the RAG ingest pipeline.
"""


def fetch_news(limit: int = 10) -> list[dict[str, str]]:
    """Return recent news items as [{title, url, text, published}]."""
    raise NotImplementedError("Later stage: RSS/API news fetch")
