from __future__ import annotations

import argparse
import logging
import sys
import uuid

from verifact import config
from verifact.graph import build_graph
from verifact.schemas import HumanReview
from examples import SAMPLE_CLAIMS, get_claim


def _setup_logging(verbose):
    level = logging.DEBUG if (verbose or config.VERBOSE) else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _print_verdict(result):
    verdict = result.get("verdict")
    if verdict is None:
        print("\nNo verdict was produced. Try --verbose.")
        return

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        colors = {"true": "bold green", "false": "bold red",
                  "mixed": "bold yellow", "unverified": "bold magenta"}
        color = colors.get(verdict.label, "bold white")

        console.print()
        console.print(Panel(
            f"[{color}]{verdict.label.upper()}[/{color}]   confidence: {verdict.confidence:.0%}",
            title="VERDICT", expand=False,
        ))
        console.print(f"[bold]Claim:[/bold] {result['claim']}")
        console.print(f"[bold]Summary:[/bold] {verdict.one_line}\n")
        console.print(f"[bold]Reasoning:[/bold]\n{verdict.reasoning}\n")

        summary = result.get("evidence_summary")
        if summary:
            t = Table(title="Evidence at a glance")
            t.add_column("Supports", style="green")
            t.add_column("Refutes", style="red")
            sup = "\n".join(f"- {p}" for p in summary.supporting_points) or "-"
            ref = "\n".join(f"- {p}" for p in summary.refuting_points) or "-"
            t.add_row(sup, ref)
            console.print(t)

        cred = result.get("credibility")
        if cred:
            console.print(
                f"\n[bold]Credibility:[/bold] quality {cred.source_quality:.2f} "
                f"recency {cred.recency:.2f} agreement {cred.cross_source_agreement:.2f}"
            )
            if cred.bias_flags:
                console.print("[bold]Bias flags:[/bold] " + "; ".join(cred.bias_flags))

        if verdict.citations:
            console.print("\n[bold]Sources:[/bold]")
            for i, c in enumerate(verdict.citations, 1):
                console.print(f"  {i}. {c}")
        console.print()
    except ImportError:
        print(f"\nVERDICT: {verdict.label.upper()}   (confidence {verdict.confidence:.0%})")
        print(f"Claim:   {result['claim']}")
        print(f"Summary: {verdict.one_line}")
        print(f"\nReasoning:\n{verdict.reasoning}")
        if verdict.citations:
            print("\nSources:")
            for i, c in enumerate(verdict.citations, 1):
                print(f"  {i}. {c}")
        print()


def _prompt_human_approval(payload):
    verdict_data = payload.get("proposed_verdict") or {}
    label = verdict_data.get("label", "?").upper()
    conf = verdict_data.get("confidence", 0)
    one_line = verdict_data.get("one_line", "")
    reasoning = verdict_data.get("reasoning", "")

    print("\n" + "=" * 60)
    print("  HUMAN APPROVAL REQUIRED")
    print("=" * 60)
    print(f"  Proposed verdict: {label}   (confidence {conf:.0%})")
    print(f"  Summary: {one_line}")
    print(f"  Reasoning: {reasoning}")
    print("=" * 60)
    print("  [a]ccept   [r]eject")
    while True:
        choice = input("  Your choice [a/r]: ").strip().lower() or "a"
        if choice in ("a", "accept", "y", "yes"):
            return HumanReview(approved=True)
        if choice in ("r", "reject", "n", "no"):
            feedback = input("  Feedback for the Judge (optional): ").strip()
            return HumanReview(approved=False, feedback=feedback)
        print("  Type 'a' to accept or 'r' to reject.")


def _run_with_approval(graph, claim, verbose, auto_approve):
    from langgraph.types import Command

    thread_config = {"configurable": {"thread_id": f"verifact-{uuid.uuid4().hex[:8]}"}}

    if verbose:
        for chunk in graph.stream({"claim": claim}, config=thread_config, stream_mode="updates"):
            for node, update in chunk.items():
                print(f"  -> {node}: {list(update.keys())}")
    else:
        graph.invoke({"claim": claim}, config=thread_config)

    state = graph.get_state(thread_config)
    while state.next and state.tasks:
        interrupts = getattr(state, "interrupts", None) or ()
        if not interrupts:
            break
        payload = interrupts[0].value

        if auto_approve:
            print("\n  (auto-approving)")
            review = HumanReview(approved=True)
        else:
            review = _prompt_human_approval(payload)

        graph.invoke(Command(resume=review), config=thread_config)
        state = graph.get_state(thread_config)

    return state.values


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verifact - multi-agent claim verification.")
    parser.add_argument("claim", nargs="?", help="The claim to verify.")
    parser.add_argument("--example", "-e", type=int, metavar="N", help="Run the Nth sample claim.")
    parser.add_argument("--list", action="store_true", help="List sample claims and exit.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Stream agent steps.")
    parser.add_argument("--no-hitl", action="store_true", help="Disable approval gate.")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-approve the verdict.")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)

    if args.list:
        print("Sample claims:\n")
        for i, item in enumerate(SAMPLE_CLAIMS, 1):
            print(f"  {i}. [{item['expected'].upper():9}] {item['claim']}")
            print(f"     {item['note']}\n")
        return 0

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
        return 2

    try:
        config.require_groq_key()
    except RuntimeError as exc:
        print(f"\n{exc}\n")
        return 1

    use_hitl = config.HUMAN_IN_THE_LOOP and not args.no_hitl
    print(f"\nVerifying: \"{claim}\"  {'(HITL on)' if use_hitl else '(auto)'}")
    print("-" * 60)
    graph = build_graph(human_in_the_loop=use_hitl)

    result = _run_with_approval(graph, claim, verbose=args.verbose, auto_approve=args.yes)
    _print_verdict(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
