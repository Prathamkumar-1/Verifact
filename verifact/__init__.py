"""Verifact — a multi-agent claim verification system.

Five specialised agents (Planner, Researcher, Evidence Analyst,
Credibility Analyst, Judge) coordinate through a LangGraph supervisor to
turn a claim or news headline into a labelled verdict with citations.
"""

from .graph import build_graph, VerifactState

__all__ = ["build_graph", "VerifactState"]
__version__ = "1.0.0"
