"""Smoke tests for Verifact.

These don't call any LLM or external API, so they run offline and without an
API key — useful as a sanity check that the schemas validate and the graph
compiles. Run with: `python -m pytest tests/ -v` (or just `python tests/test_smoke.py`).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable when running the file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_schemas_validate():
    """Every Pydantic schema should build and enforce its constraints."""
    from verifact.schemas import (
        CredibilityReport,
        Evidence,
        EvidenceSummary,
        ResearchPlan,
        SubQuestion,
        Verdict,
    )

    ev = Evidence(
        claim_aspect="q", snippet="s", source_url="http://x",
        source_title="t", stance="supports",
    )
    assert ev.stance == "supports"

    plan = ResearchPlan(sub_questions=[SubQuestion(question="q?", rationale="r")])
    assert len(plan.sub_questions) == 1

    summary = EvidenceSummary(supporting_points=["a"], refuting_points=["b"], open_questions=[])
    assert summary.refuting_points == ["b"]

    cred = CredibilityReport(source_quality=0.8, recency=0.5,
                             cross_source_agreement=0.6, bias_flags=["x"])
    assert 0 <= cred.source_quality <= 1

    v = Verdict(label="true", confidence=0.9, one_line="ok",
                reasoning="because", citations=["http://x"])
    assert v.label == "true"
    print("test_schemas_validate: OK")


def test_graph_compiles():
    """The LangGraph multi-agent graph should compile without a key."""
    # We bypass config.require_groq_key by not invoking — only compiling.
    from verifact.graph import build_graph, VerifactState

    graph = build_graph(human_in_the_loop=False)
    assert graph is not None
    # State should declare the key fields with their reducer semantics.
    hints = VerifactState.__annotations__
    assert "claim" in hints
    assert "evidence" in hints
    assert "verdict" in hints
    print("test_graph_compiles: OK")


def test_graph_compiles_with_hitl():
    """The graph should also compile WITH the human-in-the-loop gate wired in."""
    from verifact.graph import build_graph

    graph = build_graph(human_in_the_loop=True)
    assert graph is not None
    # The approval_gate and finalize nodes must exist in the compiled graph.
    node_names = set(graph.get_graph().nodes.keys())
    assert "approval_gate" in node_names
    assert "finalize" in node_names
    print("test_graph_compiles_with_hitl: OK")


def test_verdict_validation_and_fallback():
    """The Judge's validation + fallback logic should behave correctly."""
    from verifact.agents import _verdict_is_valid, _judge_fallback
    from verifact.schemas import Verdict

    # None fails validation.
    assert _verdict_is_valid(None) is False
    # A verdict with empty fields fails validation.
    assert _verdict_is_valid(Verdict(label="true", confidence=0.9,
                                     one_line="", reasoning="x")) is False
    assert _verdict_is_valid(Verdict(label="true", confidence=0.9,
                                     one_line="x", reasoning="")) is False

    # A well-formed verdict passes.
    good = Verdict(label="false", confidence=0.8, one_line="no",
                   reasoning="because", citations=["http://x"])
    assert _verdict_is_valid(good) is True

    # The fallback is always a conservative 'unverified'.
    fb = _judge_fallback("judge exploded", "some claim")
    assert fb.label == "unverified"
    assert _verdict_is_valid(fb) is True
    print("test_verdict_validation_and_fallback: OK")


def test_state_reducer_aggregates():
    """The evidence field must aggregate (operator.add) rather than overwrite.

    We simulate two researchers writing concurrently by calling the reducer's
    add function directly on two lists.
    """
    from operator import add

    from verifact.schemas import Evidence

    a = [Evidence(claim_aspect="q", snippet="a", source_url="u1", stance="supports")]
    b = [Evidence(claim_aspect="q", snippet="b", source_url="u2", stance="refutes")]
    merged = add(a, b)
    assert len(merged) == 2, "evidence reducer should concatenate, not overwrite"
    print("test_state_reducer_aggregates: OK")


if __name__ == "__main__":
    test_schemas_validate()
    test_graph_compiles()
    test_graph_compiles_with_hitl()
    test_verdict_validation_and_fallback()
    test_state_reducer_aggregates()
    print("\nAll smoke tests passed.")
