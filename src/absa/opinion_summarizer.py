"""
Opinion Summarizer for the ABSA pipeline.

Aggregates AspectSentimentPair objects from multiple reviews and produces
an OpinionSummary with product strengths and weaknesses.

Rules:
- Aspect terms are normalized (lowercase + strip) before aggregation
- Strength: Positive count > Negative count
- Weakness: Negative count > Positive count
- Tie (equal counts): excluded from both lists
- Each list sorted by dominant count descending
"""

from collections import defaultdict
from absa.models import AspectSentimentPair, AspectCount, OpinionSummary


def summarize(pairs: list[AspectSentimentPair]) -> OpinionSummary:
    """
    Aggregate aspect-sentiment pairs across multiple reviews into a summary.

    Args:
        pairs: List of AspectSentimentPair objects from one or more reviews.
               Can be empty — returns empty OpinionSummary.

    Returns:
        OpinionSummary with strengths and weaknesses sorted by count descending.
    """
    if not pairs:
        return OpinionSummary(strengths=[], weaknesses=[])

    # Count Positive and Negative occurrences per normalized aspect term
    # Structure: {normalized_term: {"Positive": int, "Negative": int, "Neutral": int}}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"Positive": 0, "Negative": 0, "Neutral": 0})

    for pair in pairs:
        normalized = pair.aspect.lower().strip()
        polarity = pair.polarity
        if polarity in counts[normalized]:
            counts[normalized][polarity] += 1

    strengths: list[AspectCount] = []
    weaknesses: list[AspectCount] = []

    for aspect, polarity_counts in counts.items():
        pos = polarity_counts["Positive"]
        neg = polarity_counts["Negative"]

        if pos > neg:
            strengths.append(AspectCount(aspect=aspect, count=pos))
        elif neg > pos:
            weaknesses.append(AspectCount(aspect=aspect, count=neg))
        # tie → excluded from both lists

    # Sort each list by count descending
    strengths.sort(key=lambda x: x.count, reverse=True)
    weaknesses.sort(key=lambda x: x.count, reverse=True)

    return OpinionSummary(strengths=strengths, weaknesses=weaknesses)
