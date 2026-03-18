"""
Benchmark: Aspect Extraction F1

Runs the trained model on the SemEval-2014 test partition and reports
token-level BIO tagging F1 for aspect extraction.

Target: macro F1 >= 0.75

Usage:
    python -m tests.benchmark.eval_aspect_extraction
    python -m tests.benchmark.eval_aspect_extraction --checkpoint models/best_model.pt --data data/raw/Laptops_Train.xml
"""

import argparse
import torch
from absa.config import load_config
from absa.xml_loader import load as load_xml
from absa.preprocessor import normalize
from absa.dataset_splitter import split as split_dataset
from absa.tokenizer import ABSATokenizer, BIO_LABEL2ID
from absa.encoder import BERTEncoder
from absa.aspect_extractor import AspectExtractor
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
    aspect_head = AspectExtractor(encoder.hidden_dim)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    encoder.load_state_dict(ckpt["encoder"])
    aspect_head.load_state_dict(ckpt["aspect_head"])
    encoder.set_inference_mode()
    aspect_head.eval()

    records = load_xml(args.data)
    splits = split_dataset(records, cfg.train_split, cfg.val_split, cfg.test_split, cfg.random_seed)
    test_records = splits.test

    print(f"Evaluating on {len(test_records)} test records...")

    all_preds, all_labels = [], []

    with torch.no_grad():
        for record in test_records:
            text = normalize(record.text)
            words = text.split()
            if not words:
                continue

            # Build ground-truth BIO labels
            word_labels = ["O"] * len(words)
            for aspect in record.aspect_terms:
                aspect_words = aspect.term.lower().split()
                for j in range(len(words) - len(aspect_words) + 1):
                    if [w.strip(".,!?;:\"'") for w in words[j:j+len(aspect_words)]] == aspect_words:
                        word_labels[j] = "B-ASP"
                        for k in range(1, len(aspect_words)):
                            word_labels[j+k] = "I-ASP"

            token_out = tokenizer.tokenize(text, word_labels=word_labels)
            input_tensor = TokenizerOutput(
                input_ids=token_out.input_ids,
                attention_mask=token_out.attention_mask,
                word_ids=token_out.word_ids,
            )
            embeddings = encoder(input_tensor)
            tag_ids, _ = aspect_head(embeddings)

            # Collect predictions and labels for non-padding tokens
            for t_pos, (tag_id, word_id) in enumerate(zip(tag_ids[0].tolist(), token_out.word_ids)):
                if word_id is not None:
                    all_preds.append(tag_id)
                    label_str = token_out.bio_tags[t_pos] if token_out.bio_tags else 0
                    if isinstance(label_str, str):
                        all_labels.append(BIO_LABEL2ID.get(label_str, 0))
                    else:
                        all_labels.append(label_str)

    per_class, macro_f1 = compute_f1(all_preds, all_labels, num_classes=3)
    label_names = {0: "O", 1: "B-ASP", 2: "I-ASP"}

    print("\n=== Aspect Extraction Results ===")
    for c, metrics in per_class.items():
        print(f"  {label_names[c]:6s} | P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")
    print(f"\n  Macro F1: {macro_f1:.4f}  (target >= 0.75)")
    status = "PASS" if macro_f1 >= 0.75 else "BELOW TARGET"
    print(f"  Status:   {status}")


if __name__ == "__main__":
    main()
