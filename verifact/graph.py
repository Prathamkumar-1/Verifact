from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import agents, config
from .schemas import CredibilityReport, Evidence, EvidenceSummary, Verdict


class VerifactState(TypedDict, total=False):
    claim: str
    sub_questions: list[str]
    evidence: Annotated[list[Evidence], add]
    evidence_summary: EvidenceSummary
    credibility: CredibilityReport
    verdict: Verdict
    approved: bool
    judge_feedback: str
    hitl_rejections: int
    next: str
    research_rounds: int
    supervisor_note: str


def _plan(state):
    return agents.planner_agent(state)


def _research(state):
    return {"research_rounds": state.get("research_rounds", 0) + 1}


def _research_one(state):
    return agents.researcher_agent(state)


def _analyze_evidence(state):
    return agents.evidence_analyst(state)


def _analyze_credibility(state):
    return agents.credibility_analyst(state)


def _judge(state):
    return agents.judge_agent(state)


def _approval_gate(state):
    return agents.approval_gate(state)


def _finalize(state):
    return agents.finalize(state)


def _supervise(state):
    return agents.supervisor_agent(state)


def _begin_analysis(state):
    return {}


def route_from_supervisor(state):
    return state.get("next", "plan")


def fan_out_research(state):
    questions = state.get("sub_questions") or [state["claim"]]
    return [Send("researcher", {"question": q}) for q in questions]


def build_graph(human_in_the_loop=None):
    from langgraph.checkpoint.memory import InMemorySaver

    if human_in_the_loop is None:
        human_in_the_loop = config.HUMAN_IN_THE_LOOP

    builder = StateGraph(VerifactState)

    builder.add_node("supervisor", _supervise)
    builder.add_node("planner", _plan)
    builder.add_node("start_research", _research)
    builder.add_node("researcher", _research_one)
    builder.add_node("analyze_step", _begin_analysis)
    builder.add_node("evidence_analyst", _analyze_evidence)
    builder.add_node("credibility_analyst", _analyze_credibility)
    builder.add_node("judge", _judge)
    builder.add_node("approval_gate", _approval_gate)
    builder.add_node("finalize", _finalize)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        ["planner", "start_research", "analyze_step", "judge"],
    )

    builder.add_edge("planner", "supervisor")

    builder.add_conditional_edges("start_research", fan_out_research, ["researcher"])
    builder.add_edge("researcher", "supervisor")

    builder.add_edge("analyze_step", "evidence_analyst")
    builder.add_edge("analyze_step", "credibility_analyst")
    builder.add_edge("evidence_analyst", "supervisor")
    builder.add_edge("credibility_analyst", "supervisor")

    if human_in_the_loop:
        builder.add_edge("judge", "approval_gate")
        builder.add_edge("finalize", END)
    else:
        builder.add_edge("judge", "finalize")
        builder.add_edge("finalize", END)

    return builder.compile(checkpointer=InMemorySaver())
