"""The Verifact LangGraph orchestration.

Orchestration patterns used:
  * **Supervisor** (primary): the `supervisor` node inspects state and routes to
    the next agent, including sending the system back for more research.
  * **Parallel + Aggregator** (map-reduce): the research step fans out one
    `Send("researcher", ...)` per sub-question, and the analysts run in parallel
    too. Outputs are aggregated via `operator.add` reducers on the state.
  * **Human-in-the-loop checkpoint** (failure handling): after the Judge rules,
    the `approval_gate` node calls `interrupt()` to pause for human approval. The
    gate returns a `Command(goto=..., update=...)` to either retry the Judge
    (on rejection) or finalize — combining routing + state update in one step.

The graph deliberately avoids a fixed `plan -> research -> analyze -> judge`
linear chain: the supervisor decides transitions dynamically. That dynamic
routing is what makes this a true supervisor rather than a pipeline.
"""
from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import agents, config
from .schemas import (
    CredibilityReport,
    Evidence,
    EvidenceSummary,
    SubQuestion,
    Verdict,
)


# ─── Shared state ────────────────────────────────────────────────────────────
# STATE DECISION (Week 4 guidance):
# We use a SHARED GLOBAL state. This is the recommended choice when the project
# is small enough to reason about key ownership directly. Each field below lists
# which agent(s) are allowed to WRITE it — read access is unrestricted.
class VerifactState(TypedDict, total=False):
    """The shared blackboard every agent reads from and writes to.

    The `Annotated[list, operator.add]` fields are the key to parallelism:
    when multiple researcher/analyst nodes write to them in the same
    super-step, LangGraph *concatenates* the lists instead of overwriting.
    """

    # Input
    claim: str                                   # write: caller (input only)

    # Planner output
    sub_questions: list[str]                     # write: planner

    # Researcher output — aggregated across the parallel fan-out.
    evidence: Annotated[list[Evidence], add]     # write: researcher (×N)

    # Analyst outputs (written once each)
    evidence_summary: EvidenceSummary            # write: evidence_analyst
    credibility: CredibilityReport               # write: credibility_analyst

    # Final verdict + human-in-the-loop bookkeeping
    verdict: Verdict                             # write: judge
    approved: bool                               # write: approval_gate / finalize
    judge_feedback: str                          # write: approval_gate (→ judge)
    hitl_rejections: int                         # write: approval_gate

    # Supervisor bookkeeping
    next: str                                    # write: supervisor
    research_rounds: int                         # write: start_research
    supervisor_note: str                         # write: supervisor


# ─── Node wrappers ───────────────────────────────────────────────────────────
def _plan(state: VerifactState) -> dict:
    return agents.planner_agent(state)


def _research(state: VerifactState) -> dict:
    """Fan out: one parallel researcher per sub-question.

    Implemented as a conditional edge returning `list[Send]]` (see
    `build_graph`). This wrapper only increments the research-round counter so
    the supervisor's loop guard stays accurate.
    """
    return {"research_rounds": state.get("research_rounds", 0) + 1}


def _research_one(state: dict) -> dict:
    """Per-sub-question researcher. `state` here is a Send-local dict."""
    return agents.researcher_agent(state)


def _analyze_evidence(state: VerifactState) -> dict:
    return agents.evidence_analyst(state)


def _analyze_credibility(state: VerifactState) -> dict:
    return agents.credibility_analyst(state)


def _judge(state: VerifactState) -> dict:
    return agents.judge_agent(state)


def _approval_gate(state: VerifactState):
    """Human-in-the-loop checkpoint. Returns a `Command` (routes + updates)."""
    return agents.approval_gate(state)


def _finalize(state: VerifactState) -> dict:
    return agents.finalize(state)


def _supervise(state: VerifactState) -> dict:
    return agents.supervisor_agent(state)


def _begin_analysis(state: VerifactState) -> dict:
    """No-op fan-out point: the two analysts are wired as parallel edges from
    here, so this node just exists to give the supervisor a single target."""
    return {}


# ─── Routing functions for conditional edges ─────────────────────────────────
def route_from_supervisor(state: VerifactState) -> str:
    """Map the supervisor's decision to a target node name."""
    return state.get("next", "plan")


def fan_out_research(state: VerifactState) -> list[Send]:
    """Parallel + Aggregator: spawn one researcher per sub-question.

    Each `Send("researcher", {...})` becomes an independent task that runs in
    the same super-step; their returned `evidence` lists are merged by the
    `operator.add` reducer. If planning produced nothing, send a single
    fallback researcher for the raw claim so the run never stalls.
    """
    questions = state.get("sub_questions") or [state["claim"]]
    return [Send("researcher", {"question": q}) for q in questions]


# ─── Graph builder ───────────────────────────────────────────────────────────
def build_graph(human_in_the_loop: bool | None = None):
    """Construct and compile the Verifact multi-agent graph.

    Args:
        human_in_the_loop: If True, the verdict is routed through an approval
            gate that pauses the graph for human review (Week 4 HITL pattern).
            If False, the verdict is finalized automatically. Defaults to the
            ``HUMAN_IN_THE_LOOP`` config flag.

    A checkpointer is ALWAYS attached, because `interrupt()` requires one and
    it also lets the CLI drive a pause/resume approval loop.
    """
    # Import here so the smoke test (which has no deps beyond langgraph) still
    # imports this module without requiring the checkpointer package.
    from langgraph.checkpoint.memory import InMemorySaver

    if human_in_the_loop is None:
        human_in_the_loop = config.HUMAN_IN_THE_LOOP

    builder = StateGraph(VerifactState)

    # Register every node.
    builder.add_node("supervisor", _supervise)
    builder.add_node("planner", _plan)
    builder.add_node("start_research", _research)      # bookkeeping + fan-out point
    builder.add_node("researcher", _research_one)      # runs N times in parallel
    builder.add_node("analyze_step", _begin_analysis)  # fan-out point for analysts
    builder.add_node("evidence_analyst", _analyze_evidence)
    builder.add_node("credibility_analyst", _analyze_credibility)
    builder.add_node("judge", _judge)
    builder.add_node("approval_gate", _approval_gate)  # HITL checkpoint
    builder.add_node("finalize", _finalize)            # terminal marker

    # Entry → supervisor.
    builder.add_edge(START, "supervisor")

    # Supervisor routes to one of: plan / research / analyze / judge.
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        ["planner", "start_research", "analyze_step", "judge"],
    )

    # After planning, hand control back to the supervisor.
    builder.add_edge("planner", "supervisor")

    # start_research fans out into N parallel researchers, then returns to
    # the supervisor (which will likely send us to analysis next).
    builder.add_conditional_edges("start_research", fan_out_research, ["researcher"])
    builder.add_edge("researcher", "supervisor")

    # Analysis is itself parallel: both analysts run, then both edges converge
    # back at the supervisor (fan-in is automatic in LangGraph).
    builder.add_edge("analyze_step", "evidence_analyst")
    builder.add_edge("analyze_step", "credibility_analyst")
    builder.add_edge("evidence_analyst", "supervisor")
    builder.add_edge("credibility_analyst", "supervisor")

    # The judge's output goes through the approval gate (or straight to finalize
    # when HITL is disabled). The gate routes via `Command(goto=...)`, so it has
    # no static outgoing edge here — LangGraph follows the Command instead.
    if human_in_the_loop:
        builder.add_edge("judge", "approval_gate")
        # approval_gate → (via Command) either "judge" (rejected, retry) or "finalize"
        builder.add_edge("finalize", END)
    else:
        builder.add_edge("judge", "finalize")
        builder.add_edge("finalize", END)

    return builder.compile(checkpointer=InMemorySaver())
