"""
Aspect Extractor — BIO Tagging Head for the ABSA pipeline.

A linear classification head on top of BERT contextual embeddings.
Predicts a BIO tag (O, B-ASP, I-ASP) for each token, then reconstructs
contiguous aspect term spans from the tag sequence.

BIO tag mapping:
    O     = 0  (not an aspect)
    B-ASP = 1  (beginning of an aspect term)
    I-ASP = 2  (inside/continuation of an aspect term)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# BIO label indices
O_IDX = 0
B_IDX = 1
I_IDX = 2
NUM_LABELS = 3


class AspectExtractor(nn.Module):
    """
    Linear BIO tagging head over BERT contextual embeddings.

    Usage:
        extractor = AspectExtractor(hidden_dim=768)
        spans = extractor.extract(embeddings, word_ids, original_words)
    """

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, NUM_LABELS)

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token BIO tag logits and softmax probabilities.

        Args:
            embeddings: Tensor of shape [batch, seq_len, hidden_dim]

        Returns:
            (tag_ids, probs):
                tag_ids: [batch, seq_len] — predicted tag index per token
                probs:   [batch, seq_len, 3] — softmax probabilities per tag
        """
        logits = self.classifier(embeddings)          # [batch, seq_len, 3]
        probs = F.softmax(logits, dim=-1)             # [batch, seq_len, 3]
        tag_ids = torch.argmax(probs, dim=-1)         # [batch, seq_len]
        return tag_ids, probs

    def extract(
        self,
        embeddings: torch.Tensor,
        word_ids: list[Optional[int]],
        original_words: list[str],
    ) -> list[dict]:
        """
        Full extraction pipeline: embeddings → BIO tags → aspect term spans.

        Args:
            embeddings:     Tensor [1, seq_len, hidden_dim] for a single sentence.
            word_ids:       Word-to-subword alignment from the tokenizer (length = seq_len).
            original_words: List of original words in the sentence (after split).

        Returns:
            List of dicts, each with:
                {
                    "term":       str,   # surface form of the aspect term
                    "span":       [int, int],  # [start_word_idx, end_word_idx] inclusive
                    "confidence": float  # mean softmax prob across span tokens
                }
            Returns empty list if no B-ASP tag is found.
        """
        with torch.no_grad():
            tag_ids, probs = self.forward(embeddings)

        # Work on the first (and only) item in the batch
        tag_ids_seq = tag_ids[0].tolist()       # [seq_len]
        probs_seq = probs[0].tolist()           # [seq_len][3]

        return _reconstruct_spans(tag_ids_seq, probs_seq, word_ids, original_words)


def _reconstruct_spans(
    tag_ids: list[int],
    probs: list[list[float]],
    word_ids: list[Optional[int]],
    original_words: list[str],
) -> list[dict]:
    """
    Reconstruct aspect term spans from a BIO tag sequence.

    Rules:
    - Only the first subword token of each word is considered (word_id changes)
    - B-ASP starts a new span
    - I-ASP continues the current span
    - O or None ends any open span
    - A lone B-ASP (not followed by I-ASP) is a valid single-word span
    - No B-ASP → returns empty list
    """
    spans = []
    current_span_words: list[int] = []       # word indices in current span
    current_span_probs: list[float] = []     # confidence probs for current span tokens
    prev_word_id = None

    for token_idx, (tag_id, word_id) in enumerate(zip(tag_ids, word_ids)):
        # Skip special tokens ([CLS], [SEP], padding)
        if word_id is None:
            if current_span_words:
                spans.append(_build_span(current_span_words, current_span_probs, original_words))
                current_span_words = []
                current_span_probs = []
            prev_word_id = None
            continue

        # Only process the first subword token of each word
        if word_id == prev_word_id:
            prev_word_id = word_id
            continue

        # New word encountered
        if tag_id == B_IDX:
            # Save any open span first
            if current_span_words:
                spans.append(_build_span(current_span_words, current_span_probs, original_words))
            # Start new span (guard against out-of-range word_id)
            if word_id < len(original_words):
                current_span_words = [word_id]
                current_span_probs = [probs[token_idx][tag_id]]

        elif tag_id == I_IDX and current_span_words:
            # Continue current span (guard against out-of-range word_id)
            if word_id < len(original_words):
                current_span_words.append(word_id)
                current_span_probs.append(probs[token_idx][tag_id])

        else:
            # O tag or I-ASP without preceding B-ASP — close any open span
            if current_span_words:
                spans.append(_build_span(current_span_words, current_span_probs, original_words))
                current_span_words = []
                current_span_probs = []

        prev_word_id = word_id

    # Close any remaining open span
    if current_span_words:
        spans.append(_build_span(current_span_words, current_span_probs, original_words))

    return spans


def _build_span(
    word_indices: list[int],
    span_probs: list[float],
    original_words: list[str],
) -> dict:
    """Build a span dict from collected word indices and probabilities."""
    start = word_indices[0]
    end = word_indices[-1]
    term = " ".join(original_words[i] for i in range(start, end + 1))
    confidence = sum(span_probs) / len(span_probs)
    return {"term": term, "span": [start, end], "confidence": round(confidence, 4)}
