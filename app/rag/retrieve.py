"""Retrieval + grounded answer assembly with citations.

Implemented in Stage 3. Question -> top-k from Chroma -> context -> grounded prompt.
"""

from app.schemas import AskResponse


def answer_question(question: str) -> AskResponse:
    """Retrieve context and produce a cited answer (context-only prompt)."""
    raise NotImplementedError("Stage 3: retrieve top-k + grounded prompt + citations")
