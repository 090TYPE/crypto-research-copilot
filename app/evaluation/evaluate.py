"""Lightweight RAG evaluation harness — LLM-as-judge + embedding metrics.

Deliberately dependency-free of RAGAS (which doesn't yet support our langchain
1.x stack). Metrics:

  - faithfulness      LLM judges whether the answer is grounded in the context
  - answer_relevancy  LLM judges whether the answer addresses the question
  - answer_similarity cosine(answer, ground_truth) via the embedding model
  - context_hit       did retrieval surface the expected source? (deterministic)

Provider-agnostic: the judge is the configured LLMProvider (Ollama by default).
Run:  uv run python -m app.evaluation.evaluate
"""

import asyncio
import re

from app.evaluation.dataset import DOCUMENTS, ITEMS
from app.llm.base import LLMProvider, get_provider
from app.rag.ingest import ingest_text
from app.rag.retrieve import _retrieve, answer_question

_FAITHFULNESS_PROMPT = """Rate how well the ANSWER is supported by the CONTEXT.
1.0 = every claim is backed by the context; 0.0 = unsupported or contradicted.
Reply with ONLY a number between 0.0 and 1.0.

CONTEXT:
{context}

ANSWER:
{answer}

Score:"""

_RELEVANCY_PROMPT = """Rate how well the ANSWER addresses the QUESTION.
1.0 = fully answers it; 0.0 = irrelevant.
Reply with ONLY a number between 0.0 and 1.0.

QUESTION: {question}
ANSWER: {answer}

Score:"""


def _parse_score(text: str) -> float:
    """Pull the first 0..1 float out of an LLM reply; clamp to [0, 1]."""
    match = re.search(r"\d*\.?\d+", text)
    if not match:
        return 0.0
    return max(0.0, min(1.0, float(match.group())))


async def _judge(provider: LLMProvider, prompt: str) -> float:
    return _parse_score(await provider.complete(prompt))


def _embed_similarity(a: str, b: str) -> float:
    """Cosine similarity of two strings under the RAG embedding model."""
    from sentence_transformers import SentenceTransformer, util

    from app.config import get_settings

    model = SentenceTransformer(get_settings().embedding_model)
    emb = model.encode([a, b], convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(emb[0], emb[1]).item())


async def run_eval() -> dict[str, float]:
    provider = get_provider()

    for doc in DOCUMENTS:
        ingest_text(doc.text, source=doc.source, url=doc.url)

    faithfulness: list[float] = []
    relevancy: list[float] = []
    similarity: list[float] = []
    context_hits: list[float] = []

    for item in ITEMS:
        hits = _retrieve(item.question)
        context = "\n".join(doc for doc, _meta, _dist in hits)
        sources = {meta.get("source") for _doc, meta, _dist in hits}

        resp = await answer_question(item.question, provider=provider)

        faithfulness.append(
            await _judge(provider, _FAITHFULNESS_PROMPT.format(context=context, answer=resp.answer))
        )
        relevancy.append(
            await _judge(
                provider, _RELEVANCY_PROMPT.format(question=item.question, answer=resp.answer)
            )
        )
        similarity.append(_embed_similarity(resp.answer, item.ground_truth))
        context_hits.append(1.0 if sources else 0.0)

    def mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    return {
        "faithfulness": mean(faithfulness),
        "answer_relevancy": mean(relevancy),
        "answer_similarity": mean(similarity),
        "context_hit_rate": mean(context_hits),
        "n": float(len(ITEMS)),
    }


def main() -> None:
    results = asyncio.run(run_eval())
    print("\nRAG evaluation")
    print("=" * 32)
    for name, value in results.items():
        print(f"{name:>18}: {value}")


if __name__ == "__main__":
    main()
