"""Central configuration: API keys, model names, and runtime knobs.

Everything that a user might want to tweak lives here so the rest of the
codebase stays simple. Keys are read from environment variables (load_dotenv
is called once on import).
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

# Load variables from a .env file in the project root if one is present.
# We do this at import time so any module that imports config picks it up.
load_dotenv()

# ─── API keys ────────────────────────────────────────────────────────────────
# Required.
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")

# Optional. Improves search reliability; if absent we fall back to DuckDuckGo.
TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")

# ─── Model selection ─────────────────────────────────────────────────────────
# A bigger, more capable model for the reasoning-heavy agents.
MODEL_STRONG = os.getenv("VERIFACT_MODEL_STRONG", "llama-3.3-70b-versatile")
# A small, fast, cheap model for the parallel research workers.
MODEL_FAST = os.getenv("VERIFACT_MODEL_FAST", "llama-3.1-8b-instant")

# ─── Search settings ─────────────────────────────────────────────────────────
MAX_SEARCH_RESULTS = int(os.getenv("VERIFACT_MAX_SEARCH_RESULTS", "4"))
# How many top evidence chunks the RAG layer hands to the analyst.
RAG_TOP_K = int(os.getenv("VERIFACT_RAG_TOP_K", "6"))

# ─── Flow control ────────────────────────────────────────────────────────────
# If the supervisor thinks the evidence is thin, it may request another research
# round. This caps how many times it can do so before being forced to judge.
MAX_RESEARCH_ROUNDS = int(os.getenv("VERIFACT_MAX_RESEARCH_ROUNDS", "2"))

# ─── Misc ────────────────────────────────────────────────────────────────────
VERBOSE = os.getenv("VERIFACT_VERBOSE", "0") == "1"


def require_groq_key() -> str:
    """Return the Groq API key or raise a friendly, actionable error.

    Every agent calls this at construction time so the user gets one clear
    message instead of a buried stack trace from the Groq client.
    """
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Grab a free key at "
            "https://console.groq.com/keys, then put it in a .env file "
            "(see .env.example)."
        )
    return GROQ_API_KEY
