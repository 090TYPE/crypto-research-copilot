"""Agent orchestration — LLM decides between RAG and tool calls.

Implemented in Stage 4 with LangChain (tool-use / agent loop).
"""

from app.schemas import AskResponse


async def run_agent(question: str) -> AskResponse:
    """Route a question through the agent (RAG + tools) and merge results."""
    raise NotImplementedError("Stage 4: LangChain agent with tools")
