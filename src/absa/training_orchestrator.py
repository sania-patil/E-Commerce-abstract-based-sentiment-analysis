"""
Training Orchestrator for the ABSA pipeline.

Jointly trains the BERT encoder, Aspect Extractor (BIO head),
and Sentiment Classifier (polarity head) on SemEval-2014 data.

Usage:
    python -m absa.training_orchestrator --data data/raw/Laptops_Train.xml
    python -m absa.training_orchestrator --config config.yaml --data data/raw/Laptops_Train.xml
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from absa.config import load_config, TrainingConfig
from absa.xml_loader import load as load_xml
from absa.preprocessor import normalize
from absa.dataset_splitter import split as split_dataset
from absa.tokenizer import ABSATokenizer, BIO_LABEL2ID
from absa.encoder import BERTEncoder
from absa.aspect_extractor import AspectExtractor
from absa.sentiment_classifier import SentimentClassifier, POLARITY2ID
from absa.models import RawRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

POLARITY_MAP = {"positive": "Positive", "negative": "Negative",
                "neutral": "Neutral", "conflict": "Neutral"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ABSADataset(Dataset):
    """PyTorch Dataset for ABSA training examples."""

    def __init__(self, records: list[RawRecord], tokenizer: ABSATokenizer, max_len: int = 512):
        self.examples = []
        for record in records:
            text = normalize(record.text)
            words = text.split()

            # Build word-level BIO labels from ground-truth aspect terms
            word_labels = ["O"] * len(words)
            for aspect in record.aspect_terms:
                # Find which words overlap with the aspect character span
                char_pos = 0
                for i, word in enumerate(words):
                    word_start = text.find(word, char_pos)
                    word_end = word_start + len(word)
                    if word_start >= aspect.char_from and word_end <= aspect.char_to + 1:
                        word_labels[i] = "B-ASP" if word_labels[i] == "O" else "I-ASP"
                    char_pos = word_start + 1

                # Simpler fallback: mark by aspect term surface form
                aspect_words = aspect.term.lower().split()
                for j in range(len(words) - len(aspect_words) + 1):
                    if [w.strip(".,!?;:\"'") for w in words[j:j+len(aspect_words)]] == aspect_words:
                        word_labels[j] = "B-ASP"
                        for k in range(1, len(aspect_words)):
                            word_labels[j+k] = "I-ASP"

            token_out = tokenizer.tokenize(text, word_labels=word_labels)

            # Build sentiment labels for each aspect term
            sentiment_examples = []
            for aspect in record.aspect_terms:
                polarity = POLARITY_MAP.get(aspect.polarity, "Neutral")
                polarity_id = POLARITY2ID[polarity]
                aspect_words = aspect.term.lower().split()
                # Find span in words
                span_start, span_end = -1, -1
                for j in range(len(words) - len(aspect_words) + 1):
                    if [w.strip(".,!?;:\"'") for w in words[j:j+len(aspect_words)]] == aspect_words:
                        span_start, span_end = j, j + len(aspect_words) - 1
                        break
                if span_start >= 0:
                    sentiment_examples.append((span_start, span_end, polarity_id))

            self.examples.append({
                "input_ids": token_out.input_ids,
                "attention_mask": token_out.attention_mask,
                "word_ids": token_out.word_ids,
                "bio_tags": token_out.bio_tags,
                "sentiment_examples": sentiment_examples,
                "words": words,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, attention_masks, bio_tags_batch = [], [], []

    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [0] * pad_len)
        attention_masks.append(ex["attention_mask"] + [0] * pad_len)
        tags = ex["bio_tags"] if ex["bio_tags"] else [0] * len(ex["input_ids"])
        bio_tags_batch.append(tags + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "bio_tags": torch.tensor(bio_tags_batch, dtype=torch.long),
        "raw": batch,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_f1(preds: list[int], labels: list[int], num_classes: int) -> dict:
    """Compute per-class and macro-averaged F1."""
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for p, l in zip(preds, labels):
        if p == l:
            tp[p] += 1
        else:
            fp[p] += 1
            fn[l] += 1

    f1s = []
    for c in range(num_classes):
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {"per_class_f1": f1s, "macro_f1": macro_f1}


# ---------------------------------------------------------------------------
# Training Orchestrator
# ---------------------------------------------------------------------------

class TrainingOrchestrator:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self._set_seed(cfg.random_seed)

        self.tokenizer = ABSATokenizer(cfg.bert_checkpoint)
        self.encoder = BERTEncoder(cfg.bert_checkpoint, cfg.dropout_rate)
        self.aspect_head = AspectExtractor(self.encoder.hidden_dim)
        self.sentiment_head = SentimentClassifier(self.encoder.hidden_dim, cfg.confidence_threshold)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        self.encoder.to(self.device)
        self.aspect_head.to(self.device)
        self.sentiment_head.to(self.device)

        self.optimizer = AdamW(
            list(self.encoder.parameters()) +
            list(self.aspect_head.parameters()) +
            list(self.sentiment_head.parameters()),
            lr=cfg.learning_rate,
        )

        self.bio_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.sentiment_loss_fn = nn.CrossEntropyLoss()

        self.best_val_f1 = 0.0
        self.checkpoint_dir = Path("models")
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self, data_path: str):
        logger.info("Loading dataset from %s", data_path)
        records = load_xml(data_path)
        splits = split_dataset(records, self.cfg.train_split, self.cfg.val_split,
                               self.cfg.test_split, self.cfg.random_seed)

        logger.info("Split: train=%d, val=%d, test=%d",
                    len(splits.train), len(splits.val), len(splits.test))

        train_ds = ABSADataset(splits.train, self.tokenizer, self.cfg.max_seq_length)
        val_ds = ABSADataset(splits.val, self.tokenizer, self.cfg.max_seq_length)
        test_ds = ABSADataset(splits.test, self.tokenizer, self.cfg.max_seq_length)

        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size,
                                  shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size,
                                shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=self.cfg.batch_size,
                                 shuffle=False, collate_fn=collate_fn)

        for epoch in range(1, self.cfg.num_epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            val_metrics = self._evaluate(val_loader, split="val")

            val_f1 = (val_metrics["aspect"]["macro_f1"] + val_metrics["sentiment"]["macro_f1"]) / 2
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_aspect_f1=%.4f | val_sentiment_f1=%.4f",
                epoch, self.cfg.num_epochs, train_loss,
                val_metrics["aspect"]["macro_f1"], val_metrics["sentiment"]["macro_f1"],
            )

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self._save_checkpoint(epoch, val_metrics)
                logger.info("Checkpoint saved (val_f1=%.4f)", val_f1)

        logger.info("Training complete. Running final test evaluation...")
        test_metrics = self._evaluate(test_loader, split="test")
        self._save_report(test_metrics)
        logger.info("Test aspect F1: %.4f | Test sentiment F1: %.4f",
                    test_metrics["aspect"]["macro_f1"], test_metrics["sentiment"]["macro_f1"])

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.encoder.set_training_mode()
        self.aspect_head.train()
        self.sentiment_head.train()

        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            bio_tags = batch["bio_tags"].to(self.device)

            # Forward pass through encoder
            from absa.models import TokenizerOutput
            token_out = TokenizerOutput(
                input_ids=input_ids, attention_mask=attention_mask, word_ids=[]
            )
            embeddings = self.encoder(token_out)

            # BIO tagging loss
            tag_logits = self.aspect_head.classifier(embeddings)
            bio_loss = self.bio_loss_fn(
                tag_logits.view(-1, 3), bio_tags.view(-1)
            )

            # Sentiment loss (from raw examples in batch)
            sentiment_loss = torch.tensor(0.0, device=self.device)
            sentiment_count = 0
            for i, ex in enumerate(batch["raw"]):
                if not ex["sentiment_examples"]:
                    continue
                cls_emb = embeddings[i, 0, :]
                word_to_tokens = {}
                for t_pos, w_id in enumerate(ex["word_ids"]):
                    if w_id is not None:
                        word_to_tokens.setdefault(w_id, []).append(t_pos)

                for start, end, polarity_id in ex["sentiment_examples"]:
                    token_positions = []
                    for w in range(start, end + 1):
                        token_positions.extend(word_to_tokens.get(w, []))
                    if not token_positions:
                        continue
                    span_emb = embeddings[i, token_positions, :].mean(dim=0)
                    combined = torch.cat([span_emb, cls_emb], dim=0).unsqueeze(0)
                    _, probs = self.sentiment_head(combined)
                    target = torch.tensor([polarity_id], device=self.device)
                    sentiment_loss = sentiment_loss + self.sentiment_loss_fn(
                        self.sentiment_head.classifier(combined), target
                    )
                    sentiment_count += 1

            if sentiment_count > 0:
                sentiment_loss = sentiment_loss / sentiment_count

            loss = bio_loss + sentiment_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader) if loader else 0.0

    def _evaluate(self, loader: DataLoader, split: str) -> dict:
        self.encoder.set_inference_mode()
        self.aspect_head.eval()
        self.sentiment_head.eval()

        bio_preds, bio_labels = [], []
        sent_preds, sent_labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                bio_tags = batch["bio_tags"].to(self.device)

                from absa.models import TokenizerOutput
                token_out = TokenizerOutput(
                    input_ids=input_ids, attention_mask=attention_mask, word_ids=[]
                )
                embeddings = self.encoder(token_out)

                tag_ids, _ = self.aspect_head(embeddings)
                mask = attention_mask.bool()
                bio_preds.extend(tag_ids[mask].cpu().tolist())
                bio_labels.extend(bio_tags[mask].cpu().tolist())

                for i, ex in enumerate(batch["raw"]):
                    if not ex["sentiment_examples"]:
                        continue
                    cls_emb = embeddings[i, 0, :]
                    word_to_tokens = {}
                    for t_pos, w_id in enumerate(ex["word_ids"]):
                        if w_id is not None:
                            word_to_tokens.setdefault(w_id, []).append(t_pos)

                    for start, end, polarity_id in ex["sentiment_examples"]:
                        token_positions = []
                        for w in range(start, end + 1):
                            token_positions.extend(word_to_tokens.get(w, []))
                        if not token_positions:
                            continue
                        span_emb = embeddings[i, token_positions, :].mean(dim=0)
                        combined = torch.cat([span_emb, cls_emb], dim=0).unsqueeze(0)
                        p_ids, _ = self.sentiment_head(combined)
                        sent_preds.append(p_ids[0].item())
                        sent_labels.append(polarity_id)

        aspect_metrics = compute_f1(bio_preds, bio_labels, num_classes=3)
        sentiment_metrics = compute_f1(sent_preds, sent_labels, num_classes=3) if sent_labels else {"macro_f1": 0.0, "per_class_f1": []}

        return {"aspect": aspect_metrics, "sentiment": sentiment_metrics}

    def _save_checkpoint(self, epoch: int, val_metrics: dict):
        path = self.checkpoint_dir / "best_model.pt"
        torch.save({
            "epoch": epoch,
            "encoder": self.encoder.state_dict(),
            "aspect_head": self.aspect_head.state_dict(),
            "sentiment_head": self.sentiment_head.state_dict(),
            "val_f1_aspect": val_metrics["aspect"]["macro_f1"],
            "val_f1_sentiment": val_metrics["sentiment"]["macro_f1"],
        }, path)

    def _save_report(self, test_metrics: dict):
        report_path = Path("outputs") / "eval_report.json"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        logger.info("Evaluation report saved to %s", report_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ABSA model")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--data", required=True, help="Path to SemEval XML training file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    orchestrator = TrainingOrchestrator(cfg)
    orchestrator.train(args.data)


if __name__ == "__main__":
    main()
