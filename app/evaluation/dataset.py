"""Tiny curated eval set: documents to ingest + questions with ground truths.

Kept small so a local-model RAGAS run finishes in minutes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    text: str
    source: str
    url: str | None = None


@dataclass(frozen=True)
class EvalItem:
    question: str
    ground_truth: str


DOCUMENTS: list[Document] = [
    Document(
        text=(
            "Bitcoin surged past $95,000 this week, driven by record inflows into "
            "spot Bitcoin ETFs and renewed institutional demand."
        ),
        source="market-wire",
        url="http://example.com/btc",
    ),
    Document(
        text=(
            "Ethereum's Pectra upgrade went live, bundling several EIPs that cut "
            "layer-2 transaction fees and improved validator UX."
        ),
        source="eth-daily",
        url="http://example.com/eth",
    ),
    Document(
        text=(
            "Solana network activity hit a new high as a wave of memecoin trading "
            "pushed daily transactions above prior records."
        ),
        source="sol-report",
        url="http://example.com/sol",
    ),
]

ITEMS: list[EvalItem] = [
    EvalItem(
        question="Why did Bitcoin's price rise this week?",
        ground_truth="Record inflows into spot Bitcoin ETFs and renewed institutional demand.",
    ),
    EvalItem(
        question="What did Ethereum's Pectra upgrade change?",
        ground_truth="It cut layer-2 transaction fees and improved validator UX.",
    ),
    EvalItem(
        question="What drove Solana's activity to a new high?",
        ground_truth="A wave of memecoin trading pushed daily transactions above prior records.",
    ),
]
