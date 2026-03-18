"""
Aspect-Sentiment Pair Builder.

Converts the raw dicts from SentimentClassifier.classify() into
typed AspectSentimentPair dataclass objects.

This is the final per-review output — a list of (aspect, polarity) pairs
with confidence scores, ready for serialization or aggregation.
"""

from absa.models import AspectSentimentPair


def build_pairs(classifier_results: list[dict]) -> list[AspectSentimentPair]:
    """
    Convert sentiment classifier output dicts into AspectSentimentPair objects.

    Args:
        classifier_results: List of dicts from SentimentClassifier.classify().
                            Each dict has: term, span, polarity, confidence,
                            low_confidence, aspect_confidence.

    Returns:
        List of AspectSentimentPair objects.
        Returns empty list if classifier_results is empty.
    """
    if not classifier_results:
        return []

    return [
        AspectSentimentPair(
            aspect=r["term"],
            polarity=r["polarity"],
            confidence=r["confidence"],
            span=r["span"],
            low_confidence=r["low_confidence"],
        )
        for r in classifier_results
    ]
