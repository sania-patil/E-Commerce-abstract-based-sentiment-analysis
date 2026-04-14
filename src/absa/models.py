# xml_loader.py  →  creates a RawRecord  →  passes it to preprocessor.py
# preprocessor.py  →  returns cleaned text  →  passes it to tokenizer.py
# tokenizer.py  →  creates a TokenizerOutput  →  passes it to encoder
# encoder  →  creates embeddings  →  passes to aspect_extractor.py
# aspect_extractor.py + sentiment_classifier.py  →  creates AspectSentimentPair  →  passes to summarizer
# summarizer  →  creates OpinionSummary  →  final output shown to user



"""
Data models for the ABSA pipeline.
These are in-memory containers that hold data as it flows between pipeline stages.
No database storage — data lives only while the program is running.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Stage 1: Raw data from SemEval XML file
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedAspect:
    """One aspect term annotation from the SemEval XML dataset."""
    term: str           # e.g. "battery life"
    polarity: str       # "positive", "negative", "neutral", or "conflict"
    char_from: int      # start character offset in the sentence
    char_to: int        # end character offset in the sentence


@dataclass
class RawRecord:
    """One parsed sentence from the SemEval XML file."""
    sentence_id: str                            # unique ID from XML
    text: str                                   # raw review sentence
    aspect_terms: list[AnnotatedAspect] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 2: Output of the BERT tokenizer
# ---------------------------------------------------------------------------

@dataclass
class TokenizerOutput:
    """
    What the tokenizer produces for one sentence.
    Passed to the BERT encoder.
    """
    input_ids: list[int]                        # token IDs including [CLS] and [SEP]
    attention_mask: list[int]                   # 1 for real tokens, 0 for padding
    word_ids: list[Optional[int]]               # maps each token position to its original word index
    bio_tags: Optional[list[int]] = None        # encoded BIO tag per token (training only)


# ---------------------------------------------------------------------------
# Stage 3: Output of the aspect extractor + sentiment classifier
# ---------------------------------------------------------------------------

@dataclass
class AspectSentimentPair:
    """
    One extracted aspect term with its predicted sentiment polarity.
    This is the core output of the pipeline for a single review.
    """
    aspect: str         # extracted aspect term, e.g. "battery life"
    polarity: str       # predicted: "Positive", "Negative", or "Neutral"
    confidence: float   # model confidence in [0, 1]
    span: list[int]     # [start_token_index, end_token_index]
    low_confidence: bool = False  # True if confidence < configured threshold


# ---------------------------------------------------------------------------
# Stage 4: Aggregated opinion summary across multiple reviews
# ---------------------------------------------------------------------------

@dataclass
class AspectCount:
    """An aspect term with its aggregated occurrence count."""
    aspect: str     # normalized (lowercase, trimmed) aspect term
    count: int      # number of reviews where this aspect had dominant polarity


@dataclass
class OpinionSummary:
    """
    Final output: product strengths and weaknesses aggregated across reviews.
    strengths = aspects where Positive count > Negative count
    weaknesses = aspects where Negative count > Positive count
    Both lists are sorted by count descending.
    """
    strengths: list[AspectCount] = field(default_factory=list)
    weaknesses: list[AspectCount] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Training: saved checkpoint metadata
# ---------------------------------------------------------------------------

@dataclass
class ModelCheckpoint:
    """Metadata for a saved model checkpoint."""
    epoch: int
    val_f1_aspect: float        # validation F1 for aspect extraction
    val_f1_sentiment: float     # validation F1 for sentiment classification
    checkpoint_path: str        # path to the saved .pt file
