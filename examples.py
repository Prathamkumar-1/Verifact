"""A handful of sample claims you can feed to Verifact for demos.

Each entry is annotated with the *expected* verdict so you can see how well
the system does. Mix of true / false / mixed / unverified, and a couple that
are deliberately current/fresh.
"""
from __future__ import annotations

SAMPLE_CLAIMS = [
    {
        "claim": "The Great Wall of China is the only man-made object visible from space.",
        "expected": "false",
        "note": "Classic myth. Astronauts have repeatedly debunked it.",
    },
    {
        "claim": "An AI program defeated the human world champion at the game of Go in 2016.",
        "expected": "true",
        "note": "AlphaGo vs Lee Sedol, 4-1. Solidly verifiable.",
    },
    {
        "claim": "5G mobile networks spread the COVID-19 virus.",
        "expected": "false",
        "note": "Widely circulated conspiracy; thoroughly debunked.",
    },
    {
        "claim": "Drinking coffee stunts your growth.",
        "expected": "false",
        "note": "Long-standing belief with no scientific support.",
    },
    {
        "claim": "Eating eggs is bad for your health because of cholesterol.",
        "expected": "mixed",
        "note": "Dietary cholesterol guidance has shifted over the years.",
    },
    {
        "claim": "The next Summer Olympics after Paris 2024 will be held in Brisbane.",
        "expected": "true",
        "note": "Brisbane 2032 — agreed by the IOC in 2021.",
    },
]


def get_claim(index: int) -> str:
    """Return the claim text at the given index (1-based for CLI friendliness)."""
    return SAMPLE_CLAIMS[index - 1]["claim"]
