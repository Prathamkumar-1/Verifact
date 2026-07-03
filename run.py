#!/usr/bin/env python
"""Verifact CLI — verify a claim with a multi-agent system.

Usage:
    python run.py "your claim here"
    python run.py --example 1        # run a built-in sample claim
    python run.py --list             # show the sample claims
    python run.py --verbose          # stream each agent step as it runs
"""
from __future__ import annotations

import argparse
import logging
import sys

# Importing config triggers load_dotenv(), so do it before anything that
# needs the API keys.
from verifact import config
from verifact.graph import build_graph
from examples import SAMPLE_CLAIMS, get_claim


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if (verbose or config.VERBOSE) else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_verdict(result: dict) -> None:
    """Pretty-print the final verdict. Uses rich if available, else plain text."""
    verdict = result.get("verdict")
    if verdict is None:
        print("\nNo verdict was produced. Run with --verbose to see what happened.")
        return

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        label_colors = {
            "true": "bold green",
            "false": "bold red",
            "mixed": "bold yellow",
            "unverified": "bold magenta",
        }
        color = label_colors.get(verdict.label, "bold white")

        console.print()
        console.print(Panel(
            f"[{color}]{verdict.label.upper()}[/{color}]   "
            f"confidence: {verdict.confidence:.0%}",
            title="VERDICT", expand=False,
        ))
        console.print(f"[bold]Claim:[/bold] {result['claim']}")
        console.print(f"[bold]Summary:[/bold] {verdict.one_line}\n")
        console.print(f"[bold]Reasoning:[/bold]\n{verdict.reasoning}\n")

        # Supporting tables from the analysts, if present.
        summary = result.get("evidence_summary")
        if summary:
            t = Table(title="Evidence at a glance", show_lines=False)
            t.add_column("Supports", style="green")
            t.add_column("Refutes", style="red")
            sup = "\n".join(f"• {p}" for p in summary.supporting_points) or "—"
            ref = "\n".join(f"• {p}" for p in summary.refuting_points) or "—"
            t.add_row(sup, ref)
            console.print(t)

        cred = result.get("credibility")
        if cred:
            console.print(
                f"\n[bold]Credibility:[/bold] quality {cred.source_quality:.2f} · "
                f"recency {cred.recency:.2f} · agreement {cred.cross_source_agreement:.2f}"
            )
            if cred.bias_flags:
                console.print("[bold]Bias flags:[/bold] " + "; ".join(cred.bias_flags))

        if verdict.citations:
            console.print("\n[bold]Sources:[/bold]")
            for i, c in enumerate(verdict.citations, 1):
                console.print(f"  {i}. {c}")
        console.print()
    except ImportError:
        # Plain-text fallback if rich isn't installed.
        print(f"\nVERDICT: {verdict.label.upper()}   (confidence {verdict.confidence:.0%})")
        print(f"Claim:   {result['claim']}")
        print(f"Summary: {verdict.one_line}")
        print(f"\nReasoning:\n{verdict.reasoning}")
        if verdict.citations:
            print("\nSources:")
            for i, c in enumerate(verdict.citations, 1):
                print(f"  {i}. {c}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verifact — multi-agent claim verification."
    )
    parser.add_argument("claim", nargs="?", help="The claim to verify.")
    parser.add_argument("--example", "-e", type=int, metavar="N",
                        help="Run the Nth sample claim (see --list).")
    parser.add_argument("--list", action="store_true", help="List sample claims and exit.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Stream each agent step as it runs.")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)

    if args.list:
        print("Sample claims:\n")
        for i, item in enumerate(SAMPLE_CLAIMS, 1):
            print(f"  {i}. [{item['expected'].upper():9}] {item['claim']}")
            print(f"       {item['note']}\n")
        return 0

    # Resolve the claim to verify.
    if args.example:
        try:
            claim = get_claim(args.example)
        except IndexError:
            print(f"Error: --example must be between 1 and {len(SAMPLE_CLAIMS)}.")
            return 2
    elif args.claim:
        claim = args.claim
    else:
        parser.error("Provide a claim, or use --example N / --list.")
        return 2  # pragma: no cover

    # Make sure we can actually call the model before building the graph.
    try:
        config.require_groq_key()
    except RuntimeError as exc:
        print(f"\n{exc}\n")
        return 1

    print(f"\nVerifying: \"{claim}\"\n" + "-" * 60)
    graph = build_graph()

    if args.verbose:
        # Stream per-node updates so you can watch the agents think.
        seen_steps = 0
        final_state = {"claim": claim}
        for chunk in graph.stream({"claim": claim}, stream_mode="updates"):
            for node, update in chunk.items():
                seen_steps += 1
                print(f"[{seen_steps:>2}] {node}: {list(update.keys())}")
        # The streamed chunks are diffs; get the full final state separately.
        final_state = graph.invoke({"claim": claim})
        result = final_state
    else:
        result = graph.invoke({"claim": claim})

    _print_verdict(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
