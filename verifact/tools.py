from __future__ import annotations

import logging

from . import config

log = logging.getLogger("verifact.tools")


def build_web_search_tool():
    if config.TAVILY_API_KEY:
        try:
            from langchain_tavily import TavilySearch
            return TavilySearch(
                max_results=config.MAX_SEARCH_RESULTS,
                topic="general",
                include_answer=True,
            )
        except Exception as exc:
            log.warning("Tavily unavailable (%s); using DuckDuckGo.", exc)

    try:
        from langchain_community.tools import DuckDuckGoSearchResults
        return DuckDuckGoSearchResults(max_results=config.MAX_SEARCH_RESULTS)
    except Exception as exc:
        raise RuntimeError("No web search backend available.") from exc


def build_wikipedia_tool():
    try:
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper
        return WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=4000)
        )
    except Exception as exc:
        log.warning("Wikipedia unavailable (%s).", exc)
        return None


class EvidenceRAG:
    def __init__(self):
        self._store = None
        self._texts = []
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
        except Exception as exc:
            log.warning("RAG unavailable (%s); using raw snippets.", exc)
            self._available = False

    def reset(self):
        self._store = None
        self._texts = []

    def add(self, texts):
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
        except Exception as exc:
            log.warning("FAISS indexing failed (%s).", exc)
            self._available = False

    def retrieve(self, query, k=None):
        k = k or config.RAG_TOP_K
        if self._available and self._store is not None:
            try:
                docs = self._store.similarity_search(query, k=k)
                if docs:
                    return [d.page_content for d in docs]
            except Exception as exc:
                log.warning("RAG retrieval failed (%s).", exc)
        return self._texts[:k]
