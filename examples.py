SAMPLE_CLAIMS = [
    {
        "claim": "The Great Wall of China is the only man-made object visible from space.",
        "expected": "false",
        "note": "Classic myth. Astronauts have repeatedly debunked it.",
    },
    {
        "claim": "An AI program defeated the human world champion at the game of Go in 2016.",
        "expected": "true",
        "note": "AlphaGo vs Lee Sedol, 4-1.",
    },
    {
        "claim": "5G mobile networks spread the COVID-19 virus.",
        "expected": "false",
        "note": "Conspiracy that was thoroughly debunked.",
    },
    {
        "claim": "Drinking coffee stunts your growth.",
        "expected": "false",
        "note": "Old belief with no scientific support.",
    },
    {
        "claim": "Eating eggs is bad for your health because of cholesterol.",
        "expected": "mixed",
        "note": "Dietary cholesterol guidance has shifted over the years.",
    },
    {
        "claim": "The next Summer Olympics after Paris 2024 will be held in Brisbane.",
        "expected": "true",
        "note": "Brisbane 2032.",
    },
]


def get_claim(index):
    return SAMPLE_CLAIMS[index - 1]["claim"]
