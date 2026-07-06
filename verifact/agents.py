from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from . import config
from .schemas import (
    CredibilityReport,
    Evidence,
    EvidenceSummary,
    HumanReview,
    ResearchPlan,
    Verdict,
)
from .tools import EvidenceRAG, build_web_search_tool, build_wikipedia_tool

log = logging.getLogger("verifact.agents")


def _llm(model, temperature=0.2):
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=config.require_groq_key(),
        max_retries=2,
    )


WEB_SEARCH = None
WIKIPEDIA = None
RAG = EvidenceRAG()


def _ensure_tools():
    global WEB_SEARCH, WIKIPEDIA
    if WEB_SEARCH is None:
        WEB_SEARCH = build_web_search_tool()
    if WIKIPEDIA is None:
        WIKIPEDIA = build_wikipedia_tool()
    return WEB_SEARCH, WIKIPEDIA


class EvidenceList(BaseModel):
    items: list[Evidence] = Field(default_factory=list)


def _summarize_search_text(raw, question, source_tag):
    if not raw or not raw.strip():
        return []
    prompt = (
        f"You are an evidence extractor. Below is raw text from {source_tag} for "
        f"the question: \"{question}\".\n\n"
        f"Pull out up to 3 concise, factual snippets. For each, set `stance` to "
        f"supports/refutes/neutral.\n\nRaw text:\n{raw[:6000]}"
    )
    try:
        extractor = _llm(config.MODEL_FAST, temperature=0).with_structured_output(EvidenceList)
        items = extractor.invoke([HumanMessage(content=prompt)])
        return items.items[:3]
    except Exception as exc:
        log.warning("Evidence extraction failed for %s: %s", source_tag, exc)
        return []


def planner_agent(state):
    claim = state["claim"]
    system = (
        "You are a research planner for a fact-checking system. Given a claim, "
        "break it into 2 to 4 focused sub-questions. Each must be independently "
        "web-searchable. Avoid yes/no questions."
    )
    plan = _llm(config.MODEL_STRONG).with_structured_output(ResearchPlan).invoke(
        [SystemMessage(content=system), HumanMessage(content=f"Claim: {claim}")]
    )
    questions = [sq.question for sq in plan.sub_questions][:4]
    return {"sub_questions": questions}


def researcher_agent(state):
    question = state["question"]
    web, wiki = _ensure_tools()
    gathered = []

    try:
        raw_web = web.invoke(question)
        gathered.extend(_summarize_search_text(str(raw_web), question, "web search"))
    except Exception as exc:
        log.warning("Web search failed for '%s': %s", question, exc)

    if wiki is not None:
        try:
            raw_wiki = wiki.invoke(question)
            gathered.extend(_summarize_search_text(str(raw_wiki), question, "wikipedia"))
        except Exception as exc:
            log.warning("Wikipedia failed for '%s': %s", question, exc)

    for ev in gathered:
        if not ev.source_url:
            ev.source_url = "wikipedia"

    return {"evidence": gathered}


def evidence_analyst(state):
    claim = state["claim"]
    evidence = state.get("evidence", [])

    if not evidence:
        return {"evidence_summary": EvidenceSummary(
            supporting_points=[], refuting_points=[],
            open_questions=["No evidence was gathered for this claim."],
        )}

    RAG.reset()
    corpus = [
        f"[{e.stance}] {e.snippet} (source: {e.source_title or e.source_url})"
        for e in evidence
    ]
    RAG.add(corpus)

    relevant = RAG.retrieve(claim)
    evidence_blob = "\n\n".join(f"- {r}" for r in relevant) or "(none retrieved)"

    system = (
        "You are an evidence analyst. Sort the snippets into points that SUPPORT "
        "the claim, points that REFUTE it, and unresolved OPEN QUESTIONS."
    )
    user = f"CLAIM: {claim}\n\nRELEVANT EVIDENCE:\n{evidence_blob}"
    summary = _llm(config.MODEL_STRONG).with_structured_output(EvidenceSummary).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return {"evidence_summary": summary}


def credibility_analyst(state):
    claim = state["claim"]
    evidence = state.get("evidence", [])

    sources = "\n".join(
        f"- [{e.stance}] {e.source_title or '(untitled)'} - {e.source_url}"
        for e in evidence
    ) or "(no sources gathered)"

    system = (
        "You are a credibility analyst. Estimate (0..1): source_quality, recency, "
        "and cross_source_agreement. Also flag any bias or contradiction."
    )
    user = f"CLAIM: {claim}\n\nSOURCES:\n{sources}"
    report = _llm(config.MODEL_STRONG).with_structured_output(CredibilityReport).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return {"credibility": report}


def _verdict_is_valid(v):
    if v is None:
        return False
    if not v.label or not v.one_line or not v.reasoning:
        return False
    if not (0.0 <= v.confidence <= 1.0):
        return False
    return True


def _judge_fallback(reasoning, claim):
    log.warning("[Judge] All retries exhausted - using fallback verdict.")
    return Verdict(
        label="unverified",
        confidence=0.0,
        one_line="Could not produce a reliable verdict for this claim.",
        reasoning=reasoning,
        citations=[],
    )


def judge_agent(state):
    claim = state["claim"]
    summary = state.get("evidence_summary")
    credibility = state.get("credibility")
    feedback = state.get("judge_feedback", "")

    system = (
        "You are the final adjudicator of a fact-check. Issue a verdict: true, "
        "false, mixed, or unverified. Calibrate confidence to the credibility "
        "scores. Always cite the strongest sources."
    )
    payload = {
        "supporting_points": summary.supporting_points if summary else [],
        "refuting_points": summary.refuting_points if summary else [],
        "open_questions": summary.open_questions if summary else [],
        "credibility": credibility.model_dump() if credibility else {},
    }
    user = f"CLAIM: {claim}\n\nANALYSIS:\n{json.dumps(payload, indent=2)}"
    if feedback:
        user += f"\n\nREVIEWER FEEDBACK TO ADDRESS: {feedback}"

    structured = _llm(config.MODEL_STRONG).with_structured_output(Verdict)
    attempts = config.JUDGE_MAX_RETRIES + 1
    verdict = None
    last_error = ""

    for attempt in range(1, attempts + 1):
        try:
            messages = [SystemMessage(content=system), HumanMessage(content=user)]
            if last_error:
                messages.append(HumanMessage(
                    content=f"Your previous attempt failed: {last_error}. Re-issue a valid verdict."
                ))
            verdict = structured.invoke(messages)
            if _verdict_is_valid(verdict):
                break
            last_error = "missing or empty fields"
        except Exception as exc:
            last_error = str(exc)

    if not _verdict_is_valid(verdict):
        verdict = _judge_fallback(f"Judge failed after {attempts} attempts: {last_error}", claim)

    return {"verdict": verdict}


def approval_gate(state):
    verdict = state.get("verdict")
    rejections = state.get("hitl_rejections", 0)

    review = interrupt({
        "kind": "verdict_review",
        "claim": state["claim"],
        "proposed_verdict": verdict.model_dump() if verdict else None,
        "rejections_so_far": rejections,
        "max_rejections": config.HITL_MAX_REJECTIONS,
    })

    if review.approved:
        return Command(goto="finalize", update={"approved": True})

    if rejections < config.HITL_MAX_REJECTIONS:
        return Command(
            goto="judge",
            update={
                "approved": False,
                "hitl_rejections": rejections + 1,
                "judge_feedback": review.feedback or "Reviewer asked for a redo.",
            },
        )

    return Command(goto="finalize", update={"approved": True})


def finalize(state):
    return {"approved": True}


def supervisor_agent(state):
    has_plan = bool(state.get("sub_questions"))
    evidence = state.get("evidence", [])
    research_rounds = state.get("research_rounds", 0)

    if not has_plan:
        next_step = "plan"
    elif len(evidence) < 2 and research_rounds < config.MAX_RESEARCH_ROUNDS:
        next_step = "research"
    elif "evidence_summary" not in state or "credibility" not in state:
        next_step = "analyze"
    else:
        next_step = "judge"

    return {"next": next_step}
