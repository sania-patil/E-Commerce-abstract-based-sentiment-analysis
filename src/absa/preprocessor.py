"""
Preprocessor for raw review text.

Cleans text before it enters the BERT tokenizer.
Operations (in order):
  1. Decode HTML entities  (e.g. &amp; -> &, &#39; -> ')
  2. Lowercase
  3. Strip leading/trailing whitespace
"""

import html


def normalize(text: str) -> str:
    """
    Normalize a raw review string for BERT tokenization.

    Args:
        text: Raw review string from the SemEval XML file.

    Returns:
        Cleaned, lowercased, whitespace-stripped string.

    Example:
        >>> normalize("  The BATTERY life is &amp; excellent!  ")
        'the battery life is & excellent!'
    """
    text = html.unescape(text)   # decode HTML entities
    text = text.lower()          # lowercase
    text = text.strip()          # strip leading/trailing whitespace
    return text
