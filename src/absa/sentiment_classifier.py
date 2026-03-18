"""
Sentiment Classifier — Polarity Head for the ABSA pipeline.

For each extracted aspect term span, classifies sentiment polarity as:
    Positive, Negative, or Neutral

Input representation per aspect:
    mean-pooled span token embeddings (768) + [CLS] token embedding (768)
    = concatenated vector of size 1536

Output:
    polarity label + confidence score + low_confidence flag
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Polarity label mapping
POLARITY_LABELS = ["Negative", "Neutral", "Positive"]
POLARITY2ID = {label: idx for idx, label in enumerate(POLARITY_LABELS)}
NUM_POLARITIES = len(POLARITY_LABELS)


class SentimentClassifier(nn.Module):
    """
    Linear polarity classification head for aspect-based sentiment analysis.

    Takes mean-pooled span embeddings concatenated with the [CLS] embedding
    and predicts one of: Positive, Negative, Neutral.

    Usage:
        classifier = SentimentClassifier(hidden_dim=768)
        results = classifier.classify(embeddings, spans, word_ids, threshold=0.5)
    """

    def __init__(self, hidden_dim: int = 768, confidence_threshold: float = 0.5):
        super().__init__()
        # Input: span mean-pool (hidden_dim) + CLS (hidden_dim) = 2 * hidden_dim
        self.classifier = nn.Linear(hidden_dim * 2, NUM_POLARITIES)
        self.confidence_threshold = confidence_threshold

    def forward(self, span_cls_vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute polarity logits and softmax probabilities.

        Args:
            span_cls_vector: Tensor [batch, hidden_dim * 2]
                             (mean-pooled span + CLS concatenated)

        Returns:
            (polarity_ids, probs):
                polarity_ids: [batch] — predicted polarity index
                probs:        [batch, 3] — softmax probabilities
        """
        logits = self.classifier(span_cls_vector)   # [batch, 3]
        probs = F.softmax(logits, dim=-1)            # [batch, 3]
        polarity_ids = torch.argmax(probs, dim=-1)   # [batch]
        return polarity_ids, probs

    def classify(
        self,
        embeddings: torch.Tensor,
        spans: list[dict],
        word_ids: list[Optional[int]],
        threshold: Optional[float] = None,
    ) -> list[dict]:
        """
        Classify sentiment polarity for each extracted aspect span.

        Args:
            embeddings: Tensor [1, seq_len, hidden_dim] — BERT output for one sentence.
            spans:      List of span dicts from AspectExtractor.extract()
                        Each has: {"term": str, "span": [start, end], "confidence": float}
            word_ids:   Word-to-subword alignment from tokenizer (length = seq_len).
            threshold:  Confidence threshold for low_confidence flag.
                        Defaults to self.confidence_threshold.

        Returns:
            List of dicts, each with:
            {
                "term":           str,
                "span":           [int, int],
                "aspect_confidence": float,
                "polarity":       str,   # "Positive", "Negative", or "Neutral"
                "confidence":     float,
                "low_confidence": bool
            }
        """
        if threshold is None:
            threshold = self.confidence_threshold

        if not spans:
            return []

        results = []
        # [CLS] token is always at position 0
        cls_embedding = embeddings[0, 0, :]  # [hidden_dim]

        # Build a map: word_index → list of token positions
        word_to_tokens: dict[int, list[int]] = {}
        for token_pos, word_id in enumerate(word_ids):
            if word_id is not None:
                word_to_tokens.setdefault(word_id, []).append(token_pos)

        with torch.no_grad():
            for span in spans:
                start_word, end_word = span["span"][0], span["span"][1]

                # Collect all token positions for words in this span
                token_positions = []
                for word_idx in range(start_word, end_word + 1):
                    token_positions.extend(word_to_tokens.get(word_idx, []))

                if not token_positions:
                    continue

                # Mean-pool span token embeddings
                span_tokens = embeddings[0, token_positions, :]  # [n_tokens, hidden_dim]
                span_mean = span_tokens.mean(dim=0)               # [hidden_dim]

                # Concatenate with [CLS]
                combined = torch.cat([span_mean, cls_embedding], dim=0).unsqueeze(0)  # [1, 1536]

                polarity_ids, probs = self.forward(combined)

                polarity_id = polarity_ids[0].item()
                confidence = probs[0, polarity_id].item()
                polarity_label = POLARITY_LABELS[polarity_id]

                results.append({
                    "term": span["term"],
                    "span": span["span"],
                    "aspect_confidence": span["confidence"],
                    "polarity": polarity_label,
                    "confidence": round(confidence, 4),
                    "low_confidence": confidence < threshold,
                })

        return results
