from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

MODEL_STRONG = os.getenv("VERIFACT_MODEL_STRONG", "llama-3.3-70b-versatile")
MODEL_FAST = os.getenv("VERIFACT_MODEL_FAST", "llama-3.1-8b-instant")

MAX_SEARCH_RESULTS = int(os.getenv("VERIFACT_MAX_SEARCH_RESULTS", "4"))
RAG_TOP_K = int(os.getenv("VERIFACT_RAG_TOP_K", "6"))

MAX_RESEARCH_ROUNDS = int(os.getenv("VERIFACT_MAX_RESEARCH_ROUNDS", "2"))
JUDGE_MAX_RETRIES = int(os.getenv("VERIFACT_JUDGE_RETRIES", "2"))

HUMAN_IN_THE_LOOP = os.getenv("VERIFACT_HITL", "1") == "1"
HITL_MAX_REJECTIONS = int(os.getenv("VERIFACT_HITL_MAX_REJECTIONS", "1"))

VERBOSE = os.getenv("VERIFACT_VERBOSE", "0") == "1"


def require_groq_key():
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and put it in a .env file."
        )
    return GROQ_API_KEY
