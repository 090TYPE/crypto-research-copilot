"""Shared test fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from app.config import get_settings


@pytest.fixture
def temp_chroma(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Point Chroma at an isolated temp dir; reset the settings cache around it."""
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeProvider:
    """Records the last prompt; returns a fixed answer. No network."""

    name = "fake"

    def __init__(self) -> None:
        self.last_prompt: str | None = None

    async def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "FAKE ANSWER"
