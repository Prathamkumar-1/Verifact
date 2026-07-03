"""Tools the agents call: web search, Wikipedia, and a small RAG retriever.

Design goals:
  * Work with ONLY a Groq key — web search falls back to keyless DuckDuckGo
    when Tavily is unavailable, and Wikipedia needs no key at all.
  * Be defensive: every external call is wrapped so a flaky source degrades
    the run instead of crashing it.
  * Provide a tiny RAG layer over gathered evidence (HF embeddings + FAISS),
    which degrades gracefully to "use raw snippets" if those libraries are
    missing or the model can't be downloaded.
"""
from __future__ import annotations

import logging
from typing import Any

from . import config

log = logging.getLogger("verifact.tools")


# ─── Web search: Tavily if configured, else DuckDuckGo ──────────────────────
def build_web_search_tool() -> Any:
    """Return a search tool, preferring Tavily and falling back to DuckDuckGo.

    Both expose the same `.invoke(query) -> str` Runnable interface, so the
    researcher agent doesn't care which one it got.
    """
    if config.TAVILY_API_KEY:
        try:
            from langchain_tavily import TavilySearch

            log.info("Using Tavily for web search.")
            return TavilySearch(
                max_results=config.MAX_SEARCH_RESULTS,
                topic="general",
                # include_answer gives the agent a tidy summary to work with
                include_answer=True,
            )
        except Exception as exc:  # pragma: no cover - import/dep issues
            log.warning("Tavily unavailable (%s); falling back to DuckDuckGo.", exc)

    # Keyless fallback. DuckDuckGo throttles scrapers, so we keep results low.
    try:
        from langchain_community.tools import DuckDuckGoSearchResults

        log.info("Using DuckDuckGo (keyless) for web search.")
        return DuckDuckGoSearchResults(max_results=config.MAX_SEARCH_RESULTS)
    except Exception as exc:
        raise RuntimeError(
            "No web search backend available. Install tavily or ddgs: "
            "`pip install langchain-tavily ddgs`."
        ) from exc


# ─── Wikipedia (keyless) ────────────────────────────────────────────────────
def build_wikipedia_tool() -> Any:
    """Return a Wikipedia query tool. Needs no API key."""
    try:
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper

        return WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=4000)
        )
    except Exception as exc:  # pragma: no cover
        log.warning("Wikipedia tool unavailable (%s). Continuing without it.", exc)
        return None


# ─── RAG over gathered evidence ─────────────────────────────────────────────
class EvidenceRAG:
    """A tiny in-memory retriever over the evidence gathered during research.

    We embed every evidence snippet with a small local sentence-transformer
    model and store them in FAISS. The analyst can then `.retrieve(query, k)`
    to get only the most relevant chunks instead of the whole noisy pile.

    If the embedding stack can't be initialised (missing weights, no network,
    library not installed) we transparently degrade to returning all evidence.
    """

    def __init__(self) -> None:
        self._store = None
        self._texts: list[str] = []
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            self._FAISS = FAISS
            self._available = True
        except Exception as exc:  # pragma: no cover - env dependent
            log.warning("RAG unavailable (%s); will use raw evidence snippets.", exc)
            self._available = False

    def reset(self) -> None:
        """Clear the index so a fresh analysis round starts from scratch."""
        self._store = None
        self._texts = []

    def add(self, texts: list[str]) -> None:
        """Index a batch of evidence snippets."""
        if not texts:
            return
        self._texts.extend(texts)
        if not self._available:
            return
        try:
            chunk = self._FAISS.from_texts(texts, self._embeddings)
            if self._store is None:
                self._store = chunk
            else:
                self._store.merge_from(chunk)
        except Exception as exc:  # pragma: no cover
            log.warning("FAISS indexing failed (%s); keeping raw snippets.", exc)
            self._available = False

    def retrieve(self, query: str, k: int | None = None) -> list[str]:
        """Return up to k snippets most relevant to the query."""
        k = k or config.RAG_TOP_K
        if self._available and self._store is not None:
            try:
                docs = self._store.similarity_search(query, k=k)
                if docs:
                    return [d.page_content for d in docs]
            except Exception as exc:  # pragma: no cover
                log.warning("RAG retrieval failed (%s); returning raw snippets.", exc)
        # Graceful fallback: just hand back what we have, trimmed to k.
        return self._texts[:k]
