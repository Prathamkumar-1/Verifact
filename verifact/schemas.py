"""Pydantic models that define the data flowing between agents.

Using typed schemas (and `with_structured_output`) means each agent returns a
validated object instead of free text, which keeps the graph robust and makes
the whole pipeline inspectable.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ─── Pieces of evidence gathered during research ────────────────────────────
class Evidence(BaseModel):
    """A single factoid pulled from a source during research."""

    claim_aspect: str = Field(
        description="The sub-question this evidence speaks to."
    )
    snippet: str = Field(
        description="The relevant text found in the source, quoted or paraphrased."
    )
    source_url: str = Field(description="URL of the source, or 'wikipedia' if unknown.")
    source_title: str = Field(default="", description="Title of the source page/article.")
    stance: Literal["supports", "refutes", "neutral"] = Field(
        description="Whether this snippet supports, refutes, or is neutral toward the original claim."
    )


# ─── Planner output ──────────────────────────────────────────────────────────
class SubQuestion(BaseModel):
    """One atomic, searchable question derived from the claim."""

    question: str = Field(description="A focused web-searchable question.")
    rationale: str = Field(default="", description="Why this question matters for the claim.")


class ResearchPlan(BaseModel):
    """The Planner's decomposition of the claim."""

    sub_questions: list[SubQuestion] = Field(
        description="2 to 4 atomic questions that together cover the claim."
    )


# ─── Analyst outputs ─────────────────────────────────────────────────────────
class EvidenceSummary(BaseModel):
    """The Evidence Analyst's read of the gathered evidence."""

    supporting_points: list[str] = Field(
        default_factory=list, description="Points that support the claim, each with a citation hint."
    )
    refuting_points: list[str] = Field(
        default_factory=list, description="Points that refute the claim, each with a citation hint."
    )
    open_questions: list[str] = Field(
        default_factory=list, description="Things the evidence could not resolve."
    )


class CredibilityReport(BaseModel):
    """The Credibility Analyst's assessment of the evidence base."""

    source_quality: float = Field(
        ge=0, le=1, description="Average trustworthiness of the sources (1 = peer-reviewed/official)."
    )
    recency: float = Field(
        ge=0, le=1, description="How current the evidence is (1 = very recent)."
    )
    cross_source_agreement: float = Field(
        ge=0, le=1, description="How well independent sources agree (1 = strong agreement)."
    )
    bias_flags: list[str] = Field(
        default_factory=list, description="Noted biases, conflicts of interest, or contradictions."
    )


# ─── Supervisor routing ──────────────────────────────────────────────────────
class SupervisorDecision(BaseModel):
    """How the supervisor wants to route the workflow next."""

    next: Literal["plan", "research", "analyze", "judge"] = Field(
        description="The next node to run."
    )
    reasoning: str = Field(default="", description="One line on why this route was chosen.")


# ─── Final verdict ───────────────────────────────────────────────────────────
VerdictLabel = Literal["true", "false", "mixed", "unverified"]


class Verdict(BaseModel):
    """The Judge's final, structured ruling."""

    label: VerdictLabel = Field(
        description="true = the claim is accurate; false = it is inaccurate; "
        "mixed = partly accurate; unverified = insufficient reliable evidence."
    )
    confidence: float = Field(
        ge=0, le=1, description="Confidence in the verdict (0 to 1)."
    )
    one_line: str = Field(description="A single-sentence summary of the verdict.")
    reasoning: str = Field(description="2-4 sentences explaining the verdict.")
    citations: list[str] = Field(
        default_factory=list, description="URLs/sources that back the verdict."
    )
