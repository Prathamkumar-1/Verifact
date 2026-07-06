from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    claim_aspect: str = Field(description="The sub-question this evidence speaks to.")
    snippet: str = Field(description="The relevant text found in the source.")
    source_url: str = Field(description="URL of the source.")
    source_title: str = Field(default="", description="Title of the source.")
    stance: Literal["supports", "refutes", "neutral"] = Field(
        description="Whether this supports, refutes, or is neutral toward the claim."
    )


class SubQuestion(BaseModel):
    question: str = Field(description="A focused web-searchable question.")
    rationale: str = Field(default="", description="Why this question matters.")


class ResearchPlan(BaseModel):
    sub_questions: list[SubQuestion] = Field(
        description="2 to 4 atomic questions that cover the claim."
    )


class EvidenceSummary(BaseModel):
    supporting_points: list[str] = Field(default_factory=list)
    refuting_points: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class CredibilityReport(BaseModel):
    source_quality: float = Field(ge=0, le=1)
    recency: float = Field(ge=0, le=1)
    cross_source_agreement: float = Field(ge=0, le=1)
    bias_flags: list[str] = Field(default_factory=list)


class SupervisorDecision(BaseModel):
    next: Literal["plan", "research", "analyze", "judge"]
    reasoning: str = Field(default="")


class HumanReview(BaseModel):
    approved: bool = Field(description="True to accept, False to reject.")
    feedback: str = Field(default="", description="Feedback if rejected.")


VerdictLabel = Literal["true", "false", "mixed", "unverified"]


class Verdict(BaseModel):
    label: VerdictLabel = Field(
        description="true, false, mixed, or unverified."
    )
    confidence: float = Field(ge=0, le=1, description="Confidence from 0 to 1.")
    one_line: str = Field(description="One-sentence summary.")
    reasoning: str = Field(description="2-4 sentences explaining the verdict.")
    citations: list[str] = Field(default_factory=list)
