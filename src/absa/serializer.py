"""
JSON Serializer and Pretty Printer for the ABSA pipeline.

Converts AspectSentimentPair lists and OpinionSummary objects to/from JSON.
Supports compact JSON and 2-space indented pretty-print.
Round-trip safe: serialize → deserialize → re-serialize produces identical output.
"""

import json
from absa.models import AspectSentimentPair, AspectCount, OpinionSummary


# ---------------------------------------------------------------------------
# AspectSentimentPair serialization
# ---------------------------------------------------------------------------

def pairs_to_json(pairs: list[AspectSentimentPair], pretty: bool = False) -> str:
    """
    Serialize a list of AspectSentimentPair objects to JSON string.

    Args:
        pairs:  List of AspectSentimentPair objects (can be empty).
        pretty: If True, output 2-space indented JSON. Default is compact.

    Returns:
        JSON string.
    """
    data = [_pair_to_dict(p) for p in pairs]
    indent = 2 if pretty else None
    return json.dumps(data, indent=indent, ensure_ascii=False)


def pairs_from_json(json_str: str) -> list[AspectSentimentPair]:
    """
    Deserialize a JSON string back into a list of AspectSentimentPair objects.

    Args:
        json_str: JSON string produced by pairs_to_json().

    Returns:
        List of AspectSentimentPair objects.
    """
    data = json.loads(json_str)
    return [_dict_to_pair(d) for d in data]


def _pair_to_dict(pair: AspectSentimentPair) -> dict:
    return {
        "aspect": pair.aspect,
        "polarity": pair.polarity,
        "confidence": pair.confidence,
        "span": pair.span,
        "low_confidence": pair.low_confidence,
    }


def _dict_to_pair(d: dict) -> AspectSentimentPair:
    return AspectSentimentPair(
        aspect=d["aspect"],
        polarity=d["polarity"],
        confidence=d["confidence"],
        span=d["span"],
        low_confidence=d["low_confidence"],
    )


# ---------------------------------------------------------------------------
# OpinionSummary serialization
# ---------------------------------------------------------------------------

def summary_to_json(summary: OpinionSummary, pretty: bool = False) -> str:
    """
    Serialize an OpinionSummary object to JSON string.

    Args:
        summary: OpinionSummary with strengths and weaknesses lists.
        pretty:  If True, output 2-space indented JSON.

    Returns:
        JSON string.
    """
    data = {
        "strengths":  [{"aspect": a.aspect, "count": a.count} for a in summary.strengths],
        "weaknesses": [{"aspect": a.aspect, "count": a.count} for a in summary.weaknesses],
    }
    indent = 2 if pretty else None
    return json.dumps(data, indent=indent, ensure_ascii=False)


def summary_from_json(json_str: str) -> OpinionSummary:
    """
    Deserialize a JSON string back into an OpinionSummary object.

    Args:
        json_str: JSON string produced by summary_to_json().

    Returns:
        OpinionSummary object.
    """
    data = json.loads(json_str)
    return OpinionSummary(
        strengths=[AspectCount(aspect=a["aspect"], count=a["count"]) for a in data["strengths"]],
        weaknesses=[AspectCount(aspect=a["aspect"], count=a["count"]) for a in data["weaknesses"]],
    )
