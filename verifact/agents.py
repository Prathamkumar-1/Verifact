"""The five Verifact agents.

Each agent is a thin, self-contained function that:
  1. builds a prompt describing its specific job,
  2. binds its tools / structured-output schema to a Groq model,
  3. returns a dict that updates the shared graph state.

Keeping each agent in its own function (instead of one big LLM-with-tools)
is what makes the division of responsibility explicit — the core requirement
of a multi-agent system.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.types import Command, interrupt

from . import config
from .schemas import (
    CredibilityReport,
    Evidence,
    EvidenceSummary,
    HumanReview,
    ResearchPlan,
    SubQuestion,
    Verdict,
)
from .tools import EvidenceRAG, build_wikipedia_tool, build_web_search_tool

log = logging.getLogger("verifact.agents")


# ============================================================================
# Shared helpers
# ============================================================================
def _llm(model: str, temperature: float = 0.2) -> ChatGroq:
    """Build a ChatGroq instance. Low temperature by default for reliability."""
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=config.require_groq_key(),
        max_retries=2,
    )


# Tools are built once and shared across researcher invocations. Building them
# at module import is cheap (they're lazy under the hood) and keeps agent
# functions free of setup boilerplate.
WEB_SEARCH = None
WIKIPEDIA = None
RAG = EvidenceRAG()


def _ensure_tools() -> tuple[Any, Any]:
    """Lazily initialise the web/Wiki tools on first use."""
    global WEB_SEARCH, WIKIPEDIA
    if WEB_SEARCH is None:
        WEB_SEARCH = build_web_search_tool()
    if WIKIPEDIA is None:
        WIKIPEDIA = build_wikipedia_tool()
    return WEB_SEARCH, WIKIPEDIA


def _summarize_search_text(raw: str, question: str, source_tag: str) -> list[Evidence]:
    """Ask the fast model to turn raw search text into tidy Evidence rows.

    Search tools return verbose, messy strings; we don't want those going
    straight into the verdict. A small extraction step keeps the evidence
    pool clean and consistently structured.
    """
    if not raw or not raw.strip():
        return []
    prompt = (
        f"You are an evidence extractor. Below is raw text from {source_tag} for "
        f"the question: \"{question}\".\n\n"
        f"Pull out up to 3 concise, factual snippets that bear on this question. "
        f"For each, set `stance` to supports/refutes/neutral relative to the "
        f"question being factually answered.\n\n"
        f"Raw text:\n{raw[:6000]}"
    )
    try:
        extractor = _llm(config.MODEL_FAST, temperature=0).with_structured_output(
            EvidenceList
        )
        items = extractor.invoke([HumanMessage(content=prompt)])
        return items.items[:3]
    except Exception as exc:  # pragma: no cover
        log.warning("Evidence extraction failed for %s: %s", source_tag, exc)
        return []


# A throwaway schema just for the extraction helper above.
from pydantic import BaseModel, Field  # noqa: E402  (local import kept tidy)

class EvidenceList(BaseModel):
    items: list[Evidence] = Field(default_factory=list)


# ============================================================================
# Agent 1 — Planner: break the claim into searchable sub-questions
# ============================================================================
def planner_agent(state: dict) -> dict:
    """Decompose the claim into 2–4 atomic, web-searchable sub-questions."""
    claim = state["claim"]
    log.info("[Planner] Decomposing claim: %s", claim)

    system = (
        "You are a research planner for a fact-checking system. Given a claim, "
        "break it into 2 to 4 focused sub-questions that, when answered, would "
        "let a judge decide if the claim is true, false, mixed, or unverified. "
        "Each sub-question must be independently web-searchable. Avoid yes/no "
        "questions; prefer 'what/when/who/does' factual queries."
    )
    plan = _llm(config.MODEL_STRONG).with_structured_output(ResearchPlan).invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=f"Claim: {claim}"),
        ]
    )

    questions = [sq.question for sq in plan.sub_questions][:4]
    log.info("[Planner] Produced %d sub-questions.", len(questions))
    return {"sub_questions": questions}


# ============================================================================
# Agent 2 — Researcher: gather evidence for ONE sub-question (runs ×N parallel)
# ============================================================================
def researcher_agent(state: dict) -> dict:
    """Gather evidence for a single sub-question using web search + Wikipedia.

    This node runs once per sub-question via parallel `Send(...)` fan-out, so
    it must be self-contained and side-effect-free except for returning new
    evidence. Results are aggregated by the `operator.add` reducer on the
    `evidence` field.
    """
    question: str = state["question"]
    log.info("[Researcher] Investigating: %s", question)
    web, wiki = _ensure_tools()

    gathered: list[Evidence] = []

    # --- Web search leg ---
    try:
        raw_web = web.invoke(question)
        gathered.extend(_summarize_search_text(str(raw_web), question, "web search"))
    except Exception as exc:  # pragma: no cover - network / rate-limit path
        log.warning("[Researcher] Web search failed for '%s': %s", question, exc)

    # --- Wikipedia leg (independent, keyless, adds a stable reference source) ---
    if wiki is not None:
        try:
            raw_wiki = wiki.invoke(question)
            gathered.extend(_summarize_search_text(str(raw_wiki), question, "wikipedia"))
        except Exception as exc:  # pragma: no cover
            log.warning("[Researcher] Wikipedia failed for '%s': %s", question, exc)

    # Stamp any source-less evidence with the wikipedia tag so citations are clean.
    for ev in gathered:
        if not ev.source_url:
            ev.source_url = "wikipedia"

    log.info("[Researcher] %d evidence items for '%s'.", len(gathered), question)
    return {"evidence": gathered}


# ============================================================================
# Agent 3 — Evidence Analyst: RAG-backed reading of the evidence
# ============================================================================
def evidence_analyst(state: dict) -> dict:
    """Retrieve the most relevant evidence via RAG and read it for/against."""
    claim = state["claim"]
    evidence: list[Evidence] = state.get("evidence", [])
    log.info("[Evidence Analyst] %d evidence items to review.", len(evidence))

    if not evidence:
        return {"evidence_summary": EvidenceSummary(
            supporting_points=[], refuting_points=[],
            open_questions=["No evidence was gathered for this claim."],
        )}

    # Build a clean corpus for the RAG layer and (re)index it.
    RAG.reset()
    corpus = [
        f"[{e.stance}] {e.snippet} (source: {e.source_title or e.source_url})"
        for e in evidence
    ]
    RAG.add(corpus)

    # Retrieve the chunks most relevant to the *original claim*.
    relevant = RAG.retrieve(claim)
    evidence_blob = "\n\n".join(f"- {r}" for r in relevant) or "(none retrieved)"

    system = (
        "You are an evidence analyst. You are given the most relevant retrieved "
        "snippets about a claim. Sort them into points that SUPPORT the claim, "
        "points that REFUTE it, and unresolved OPEN QUESTIONS. Each point must "
        "reference which source it came from. Be neutral and precise."
    )
    user = (
        f"CLAIM: {claim}\n\n"
        f"RELEVANT EVIDENCE:\n{evidence_blob}\n\n"
        f"Produce the structured summary."
    )
    summary = _llm(config.MODEL_STRONG).with_structured_output(EvidenceSummary).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return {"evidence_summary": summary}


# ============================================================================
# Agent 4 — Credibility Analyst: assess source quality/recency/agreement
# ============================================================================
def credibility_analyst(state: dict) -> dict:
    """Score the trustworthiness of the evidence base."""
    claim = state["claim"]
    evidence: list[Evidence] = state.get("evidence", [])
    log.info("[Credibility Analyst] Assessing %d sources.", len(evidence))

    # A compact description of the sources for the model to assess.
    sources = "\n".join(
        f"- [{e.stance}] {e.source_title or '(untitled)'} — {e.source_url}"
        for e in evidence
    ) or "(no sources gathered)"

    system = (
        "You are a credibility analyst for a fact-checking system. Given a claim "
        "and the list of sources gathered, estimate (0..1): source_quality "
        "(official/peer-reviewed = high, random blogs = low), recency (recent = "
        "high), and cross_source_agreement (independent sources agree = high). "
        "Also flag any obvious bias, conflict of interest, or contradiction."
    )
    user = f"CLAIM: {claim}\n\nSOURCES:\n{sources}"
    report = _llm(config.MODEL_STRONG).with_structured_output(CredibilityReport).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return {"credibility": report}


# ============================================================================
# Agent 5 — Judge: produce the final verdict (with retry failure handling)
# ============================================================================
def _verdict_is_valid(v: Verdict | None) -> bool:
    """Cheap validation pass on the Judge's output.

    This is the first failure-handling mechanism from the Week 4 notes
    (retries): if the model returns something empty or malformed, we detect it
    and retry with a corrective prompt before falling back.
    """
    if v is None:
        return False
    if not v.label or not v.one_line or not v.reasoning:
        return False
    if not (0.0 <= v.confidence <= 1.0):
        return False
    return True


def _judge_fallback(reasoning: str, claim: str) -> Verdict:
    """Fallback verdict used only if the Judge fails every retry.

    The second failure-handling mechanism: instead of crashing, we emit a
    conservative `unverified` verdict so the run degrades gracefully — exactly
    the "fallback agent" pattern from the Week 4 notes.
    """
    log.warning("[Judge] All retries exhausted — using fallback verdict.")
    return Verdict(
        label="unverified",
        confidence=0.0,
        one_line="The system could not produce a reliable verdict for this claim.",
        reasoning=reasoning,
        citations=[],
    )


def judge_agent(state: dict) -> dict:
    """Synthesise the analysis + credibility into a final verdict.

    Implements the **retry** failure-handling pattern: the model is called up
    to `JUDGE_MAX_RETRIES + 1` times. Each retry after the first adds a
    "your previous output was invalid, fix it" instruction so the model
    corrects itself rather than silently producing garbage.
    """
    claim = state["claim"]
    summary = state.get("evidence_summary")
    credibility = state.get("credibility")
    # Feedback from a human rejection (HITL) re-enters here.
    feedback = state.get("judge_feedback", "")
    log.info("[Judge] Issuing verdict for: %s", claim)

    system = (
        "You are the final adjudicator of a fact-check. Using the analyst's "
        "summary and the credibility report, issue a verdict:\n"
        "  - true: the claim is accurate\n"
        "  - false: the claim is inaccurate\n"
        "  - mixed: partly accurate, partly not\n"
        "  - unverified: not enough reliable evidence to decide\n"
        "Calibrate confidence to the credibility scores — if sources are weak or "
        "contradictory, confidence must be low. Always cite the strongest sources."
    )

    # Serialise the upstream outputs into a compact prompt.
    payload = {
        "supporting_points": summary.supporting_points if summary else [],
        "refuting_points": summary.refuting_points if summary else [],
        "open_questions": summary.open_questions if summary else [],
        "credibility": credibility.model_dump() if credibility else {},
    }
    user = (
        f"CLAIM: {claim}\n\n"
        f"ANALYSIS:\n{json.dumps(payload, indent=2)}\n\n"
        f"Issue the structured verdict."
    )
    if feedback:
        # A reviewer asked the judge to reconsider — fold their note in.
        user += f"\n\nREVIEWER FEEDBACK TO ADDRESS: {feedback}"

    structured = _llm(config.MODEL_STRONG).with_structured_output(Verdict)

    # Retry loop: try, validate, and re-ask with a corrective nudge if bad.
    attempts = config.JUDGE_MAX_RETRIES + 1
    verdict = None
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            messages = [SystemMessage(content=system), HumanMessage(content=user)]
            if last_error:
                messages.append(HumanMessage(
                    content=f"Your previous attempt failed validation: "
                    f"{last_error}. Please re-issue a complete, valid verdict."
                ))
            verdict = structured.invoke(messages)
            if _verdict_is_valid(verdict):
                log.info("[Judge] Valid verdict on attempt %d.", attempt)
                break
            last_error = "missing or empty fields, or confidence out of range"
            log.warning("[Judge] Attempt %d produced invalid verdict.", attempt)
        except Exception as exc:  # pragma: no cover - network / API path
            last_error = str(exc)
            log.warning("[Judge] Attempt %d raised: %s", attempt, exc)

    if not _verdict_is_valid(verdict):
        verdict = _judge_fallback(
            f"Judge failed after {attempts} attempts (last issue: {last_error}).",
            claim,
        )

    return {"verdict": verdict}


# ============================================================================
# Agent 6 — Approval Gate: human-in-the-loop checkpoint
# ============================================================================
def approval_gate(state: dict) -> Command:
    """Pause for human approval before the verdict is finalized.

    This is the **human-in-the-loop** failure-handling pattern from Week 4,
    implemented with LangGraph's `interrupt()`. The graph pauses here, surfaces
    the proposed verdict to the caller, and resumes only after the human
    decides. On rejection (within the rejection cap) we route back to the Judge
    with the reviewer's feedback; on approval (or cap reached) we finish.

    Returns a `Command` instead of a dict so we can route AND update state in
    one step — the `Command(goto=..., update=...)` pattern from Week 4.
    """
    verdict = state.get("verdict")
    rejections = state.get("hitl_rejections", 0)

    # Surface the proposed verdict to the caller. On resume, `review` becomes
    # the value passed to `Command(resume=HumanReview(...))`.
    review: HumanReview = interrupt({
        "kind": "verdict_review",
        "claim": state["claim"],
        "proposed_verdict": verdict.model_dump() if verdict else None,
        "rejections_so_far": rejections,
        "max_rejections": config.HITL_MAX_REJECTIONS,
    })

    if review.approved:
        log.info("[Gate] Verdict approved by reviewer.")
        return Command(goto="finalize", update={"approved": True})

    # Rejected. If we still have budget, send back to the judge with feedback.
    if rejections < config.HITL_MAX_REJECTIONS:
        log.info("[Gate] Verdict rejected — routing back to Judge (rejection %d).",
                 rejections + 1)
        return Command(
            goto="judge",
            update={
                "approved": False,
                "hitl_rejections": rejections + 1,
                "judge_feedback": review.feedback or "Reviewer asked for a redo.",
            },
        )

    # Out of retries: accept the current verdict to avoid an infinite loop.
    log.warning("[Gate] Rejection cap reached — accepting current verdict.")
    return Command(goto="finalize", update={"approved": True})


def finalize(state: dict) -> dict:
    """Terminal marker node: the verdict is locked in and approved."""
    return {"approved": True}


# ============================================================================
# Supervisor — routes the workflow
# ============================================================================
def supervisor_agent(state: dict) -> dict:
    """Decide which agent runs next based on the current state.

    This is the heart of the Supervisor orchestration pattern: instead of a
    fixed pipeline, a coordinator inspects progress and picks the next step —
    including sending the system back for another research round when the
    evidence looks thin.
    """
    claim = state["claim"]
    has_plan = bool(state.get("sub_questions"))
    evidence = state.get("evidence", [])
    research_rounds = state.get("research_rounds", 0)

    # Hard rule: never loop research forever.
    if not has_plan:
        next_step, reason = "plan", "no plan yet"
    elif len(evidence) < 2 and research_rounds < config.MAX_RESEARCH_ROUNDS:
        next_step, reason = "research", f"only {len(evidence)} evidence items"
    elif "evidence_summary" not in state or "credibility" not in state:
        next_step, reason = "analyze", "evidence ready, time to analyze"
    else:
        next_step, reason = "judge", "analysis complete"

    log.info("[Supervisor] → %s (%s)", next_step, reason)
    return {"next": next_step, "supervisor_note": reason}
