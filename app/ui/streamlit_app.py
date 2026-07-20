"""Streamlit demo UI for the Crypto Research Copilot.

Talks to the running FastAPI backend over HTTP (set API_BASE_URL to override).
Run:  uv run streamlit run app/ui/streamlit_app.py
"""

import sys
from pathlib import Path

# Streamlit runs this file directly, so put the repo root on sys.path for `app.*`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import Any  # noqa: E402

import httpx  # noqa: E402
import streamlit as st  # noqa: E402

from app.ui.client import API_BASE_URL, agent, ask, ingest  # noqa: E402

st.set_page_config(page_title="Crypto Research Copilot", page_icon="🪙", layout="centered")
st.title("🪙 Crypto Research Copilot")
st.caption(f"RAG + agent over your ingested crypto news · backend: {API_BASE_URL}")

ask_tab, agent_tab, ingest_tab = st.tabs(["Ask (RAG)", "Agent", "Ingest"])


def _sources(result: dict[str, Any]) -> None:
    sources = result.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                label = s.get("source") or "unknown"
                url = s.get("url")
                head = f"**{label}**" + (f" — {url}" if url else "")
                st.markdown(f"{head}\n\n> {s.get('text', '')}")


with ask_tab:
    st.subheader("Ask the knowledge base")
    q = st.text_input("Question", key="ask_q", placeholder="What's new with Bitcoin?")
    if st.button("Ask", key="ask_btn") and q:
        with st.spinner("Retrieving + answering…"):
            try:
                res = ask(q)
                st.markdown(res["answer"])
                conf = res.get("confidence")
                if conf is not None:
                    st.progress(min(1.0, float(conf)), text=f"confidence: {conf}")
                _sources(res)
            except httpx.HTTPError as exc:
                st.error(f"Backend error: {exc}")

with agent_tab:
    st.subheader("Ask the tool-using agent")
    st.caption("The agent may fetch live prices, compute indicators, or search news.")
    aq = st.text_input("Question", key="agent_q", placeholder="Price of bitcoin and any news?")
    if st.button("Run agent", key="agent_btn") and aq:
        with st.spinner("Agent thinking (may call tools)…"):
            try:
                res = agent(aq)
                st.markdown(res["answer"])
                tools = res.get("tools_used") or []
                if tools:
                    st.info("Tools used: " + ", ".join(tools))
                _sources(res)
            except httpx.HTTPError as exc:
                st.error(f"Backend error: {exc}")

with ingest_tab:
    st.subheader("Ingest a document")
    text = st.text_area("Text", key="ing_text", height=160)
    col1, col2 = st.columns(2)
    source = col1.text_input("Source", key="ing_source", placeholder="market-wire")
    url = col2.text_input("URL (optional)", key="ing_url")
    if st.button("Ingest", key="ing_btn") and text and source:
        with st.spinner("Chunking + embedding…"):
            try:
                res = ingest(text, source, url or None)
                st.success(f"Ingested {res['chunks_ingested']} chunk(s).")
            except httpx.HTTPError as exc:
                st.error(f"Backend error: {exc}")
