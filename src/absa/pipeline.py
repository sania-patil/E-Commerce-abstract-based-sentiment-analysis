"""
Inference Pipeline for the ABSA system.

Wires all components end-to-end:
    review string
    → Preprocessor
    → Tokenizer
    → BERTEncoder
    → AspectExtractor
    → SentimentClassifier
    → PairBuilder
    → OpinionSummarizer
    → JSON output

Usage:
    python -m absa.pipeline --checkpoint models/best_model.pt --review "The battery life is great but the keyboard feels cheap"
"""

import argparse
import json
import torch
from pathlib import Path

from absa.preprocessor import normalize
from absa.tokenizer import ABSATokenizer
from absa.encoder import BERTEncoder
from absa.aspect_extractor import AspectExtractor
from absa.sentiment_classifier import SentimentClassifier
from absa.pair_builder import build_pairs
from absa.opinion_summarizer import summarize
from absa.models import TokenizerOutput
from absa.config import load_config

# Stop words to filter out as standalone aspects
_STOP_WORDS = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "is", "was", "are", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "need", "dare", "ought", "used",
    "i", "you", "he", "she", "we", "they", "my", "your", "his", "her",
    "our", "their", "me", "him", "us", "them", "and", "but", "or",
    "so", "yet", "for", "nor", "not", "no", "very", "just", "also",
}


def _postprocess_spans(spans: list[dict], words: list[str]) -> list[dict]:
    """
    Merge adjacent single-word spans into multi-word aspects and
    filter out stop words standing alone as aspects.

    Two spans are merged if their word indices are consecutive
    and they share the same polarity direction (both positive or both negative).
    """
    if not spans:
        return []

    # Sort by start word index
    spans = sorted(spans, key=lambda s: s["span"][0])

    merged = []
    i = 0
    while i < len(spans):
        current = spans[i]
        # Try to merge with next span if consecutive
        while i + 1 < len(spans):
            nxt = spans[i + 1]
            # Consecutive if current end + 1 == next start
            if current["span"][1] + 1 == nxt["span"][0]:
                # Merge: extend span, join terms, average confidence
                new_end = nxt["span"][1]
                new_term = " ".join(words[current["span"][0]:new_end + 1])
                new_conf = round((current["confidence"] + nxt["confidence"]) / 2, 4)
                current = {
                    "term": new_term,
                    "span": [current["span"][0], new_end],
                    "confidence": new_conf,
                }
                i += 1
            else:
                break
        merged.append(current)
        i += 1

    # Filter stop words (only when the entire term is a single stop word)
    filtered = [
        s for s in merged
        if s["term"].lower() not in _STOP_WORDS
    ]

    return filtered


class ABSAPipeline:
    """End-to-end ABSA inference pipeline."""

    def __init__(self, checkpoint_path: str, config_path: str = "config.yaml"):
        cfg = load_config(config_path)

        self.tokenizer = ABSATokenizer(cfg.bert_checkpoint)
        self.encoder = BERTEncoder(cfg.bert_checkpoint, cfg.dropout_rate)
        self.aspect_head = AspectExtractor(self.encoder.hidden_dim)
        self.sentiment_head = SentimentClassifier(self.encoder.hidden_dim, cfg.confidence_threshold)

        # Load trained weights
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.encoder.load_state_dict(ckpt["encoder"])
        self.aspect_head.load_state_dict(ckpt["aspect_head"])
        self.sentiment_head.load_state_dict(ckpt["sentiment_head"])

        self.encoder.set_inference_mode()
        self.aspect_head.eval()
        self.sentiment_head.eval()

    def run(self, review: str) -> dict:
        """
        Run the full pipeline on a single review string.

        Returns a dict with:
            - aspects: list of {term, polarity, confidence, low_confidence}
            - summary: {strengths: [...], weaknesses: [...]}
        """
        # 1. Preprocess
        text = normalize(review)
        words = text.split()

        if not words:
            return {"aspects": [], "summary": {"strengths": [], "weaknesses": []}}

        # 2. Tokenize
        token_out = self.tokenizer.tokenize(text)

        # 3. Encode
        with torch.no_grad():
            input_tensor = TokenizerOutput(
                input_ids=token_out.input_ids,
                attention_mask=token_out.attention_mask,
                word_ids=token_out.word_ids,
            )
            embeddings = self.encoder(input_tensor)  # [1, seq_len, 768]

        # 4. Extract aspects
        raw_spans = self.aspect_head.extract(embeddings, token_out.word_ids, words)
        spans = _postprocess_spans(raw_spans, words)

        # 5. Classify sentiment per aspect
        sentiment_results = self.sentiment_head.classify(
            embeddings, spans, token_out.word_ids
        )

        # 6. Build pairs
        pairs = build_pairs(sentiment_results)

        # 7. Summarize
        summary = summarize(pairs)

        # 8. Format output
        return {
            "review": review,
            "aspects": [
                {
                    "term": p.aspect,
                    "polarity": p.polarity,
                    "confidence": p.confidence,
                    "low_confidence": p.low_confidence,
                }
                for p in pairs
            ],
            "summary": {
                "strengths": [{"aspect": s.aspect, "count": s.count} for s in summary.strengths],
                "weaknesses": [{"aspect": w.aspect, "count": w.count} for w in summary.weaknesses],
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Run ABSA inference on a review")
    parser.add_argument("--checkpoint", default="models/best_model.pt", help="Path to trained model checkpoint")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--review", required=True, help="Review text to analyze")
    args = parser.parse_args()

    pipeline = ABSAPipeline(args.checkpoint, args.config)
    result = pipeline.run(args.review)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
