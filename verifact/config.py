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

# ─── Failure handling (Week 4 requirement: build in ≥1 mechanism) ────────────
# (1) Retries: when the Judge returns a malformed/empty verdict, we retry with a
#     "fix your output" prompt before giving up.
JUDGE_MAX_RETRIES = int(os.getenv("VERIFACT_JUDGE_RETRIES", "2"))

# (2) Human-in-the-loop: before a verdict is finalized, pause the graph and let a
#     human approve it. Disable for fully-automated runs (e.g. batch tests).
HUMAN_IN_THE_LOOP = os.getenv("VERIFACT_HITL", "1") == "1"
# How many times a human can reject a verdict and ask the Judge to redo it.
HITL_MAX_REJECTIONS = int(os.getenv("VERIFACT_HITL_MAX_REJECTIONS", "1"))

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
