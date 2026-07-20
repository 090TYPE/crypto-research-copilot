"""Schema validation + defaults."""

import pytest
from pydantic import ValidationError

from app.schemas import AskResponse, IngestRequest


def test_ingest_request_requires_nonempty_text() -> None:
    with pytest.raises(ValidationError):
        IngestRequest(text="", source="x")


def test_ask_response_defaults() -> None:
    r = AskResponse(answer="hi")
    assert r.sources == []
    assert r.tools_used == []
    assert r.confidence is None
