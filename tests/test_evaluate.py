"""Eval harness — pure score parsing + embedding similarity sanity."""

import pytest

from app.evaluation.evaluate import _parse_score


def test_parse_score_extracts_and_clamps() -> None:
    assert _parse_score("0.8") == 0.8
    assert _parse_score("Score: 0.42 out of 1") == 0.42
    assert _parse_score("1.5") == 1.0  # clamped
    assert _parse_score("no number here") == 0.0


def test_embed_similarity_orders_by_meaning() -> None:
    pytest.importorskip("sentence_transformers")
    from app.evaluation.evaluate import _embed_similarity

    same = _embed_similarity("Bitcoin rose on ETF inflows", "Bitcoin rose on ETF inflows")
    related = _embed_similarity("Bitcoin rose on ETF inflows", "BTC climbed due to ETF demand")
    unrelated = _embed_similarity("Bitcoin rose on ETF inflows", "The cat sat on the mat")

    assert same > related > unrelated
