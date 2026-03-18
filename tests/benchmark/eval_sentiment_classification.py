"""
Benchmark: Sentiment Classification F1

Runs the trained model on the SemEval-2014 test partition and reports
macro-averaged sentiment polarity F1.

Target: macro F1 >= 0.70

Usage:
    python -m tests.benchmark.eval_sentiment_classification
    python -m tests.benchmark.eval_sentiment_classification --checkpoint models/best_model.pt --data data/raw/Laptops_Train.xml
"""

import argparse
import torch
from absa.config import load_config
from absa.xml_loader import load as load_xml
from absa.preprocessor import normalize
from absa.dataset_splitter import split as split_dataset
from absa.tokenizer import ABSATokenizer
from absa.encoder import BERTEncoder
from absa.sentiment_classifier import SentimentClassifier, POLARITY2ID, POLARITY_LABELS
from absa.models import TokenizerOutput

POLARITY_MAP = {"positive": "Positive", "negative": "Negative",
                "neutral": "Neutral", "conflict": "Neutral"}


def compute_f1(preds, labels, num_classes):
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    for p, l in zip(preds, labels):
        if p == l:
            tp[p] += 1
        else:
            fp[p] += 1
            fn[l] += 1
    results = {}
    for c in range(num_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    macro_f1 = sum(v["f1"] for v in results.values()) / num_classes
    return results, round(macro_f1, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default="data/raw/Laptops_Train.xml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = ABSATokenizer(cfg.bert_checkpoint)
    encoder = BERTEncoder(cfg.bert_checkpoint, cfg.dropout_rate)
    sentiment_head = SentimentClassifier(encoder.hidden_dim, cfg.confidence_threshold)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    encoder.load_state_dict(ckpt["encoder"])
    sentiment_head.load_state_dict(ckpt["sentiment_head"])
    encoder.set_inference_mode()
    sentiment_head.eval()

    records = load_xml(args.data)
    splits = split_dataset(records, cfg.train_split, cfg.val_split, cfg.test_split, cfg.random_seed)
    test_records = splits.test

    print(f"Evaluating on {len(test_records)} test records...")

    all_preds, all_labels = [], []

    with torch.no_grad():
        for record in test_records:
            text = normalize(record.text)
            words = text.split()
            if not words or not record.aspect_terms:
                continue

            token_out = tokenizer.tokenize(text)
            input_tensor = TokenizerOutput(
                input_ids=token_out.input_ids,
                attention_mask=token_out.attention_mask,
                word_ids=token_out.word_ids,
            )
            embeddings = encoder(input_tensor)
            cls_emb = embeddings[0, 0, :]

            word_to_tokens = {}
            for t_pos, word_id in enumerate(token_out.word_ids):
                if word_id is not None:
                    word_to_tokens.setdefault(word_id, []).append(t_pos)

            for aspect in record.aspect_terms:
                polarity = POLARITY_MAP.get(aspect.polarity, "Neutral")
                true_label = POLARITY2ID[polarity]

                aspect_words = aspect.term.lower().split()
                span_start = -1
                for j in range(len(words) - len(aspect_words) + 1):
                    if [w.strip(".,!?;:\"'") for w in words[j:j+len(aspect_words)]] == aspect_words:
                        span_start = j
                        break
                if span_start < 0:
                    continue

                span_end = span_start + len(aspect_words) - 1
                token_positions = []
                for w in range(span_start, span_end + 1):
                    token_positions.extend(word_to_tokens.get(w, []))
                if not token_positions:
                    continue

                span_emb = embeddings[0, token_positions, :].mean(dim=0)
                combined = torch.cat([span_emb, cls_emb], dim=0).unsqueeze(0)
                p_ids, _ = sentiment_head(combined)

                all_preds.append(p_ids[0].item())
                all_labels.append(true_label)

    per_class, macro_f1 = compute_f1(all_preds, all_labels, num_classes=3)

    print("\n=== Sentiment Classification Results ===")
    for c, metrics in per_class.items():
        print(f"  {POLARITY_LABELS[c]:8s} | P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")
    print(f"\n  Macro F1: {macro_f1:.4f}  (target >= 0.70)")
    status = "PASS" if macro_f1 >= 0.70 else "BELOW TARGET"
    print(f"  Status:   {status}")


if __name__ == "__main__":
    main()
