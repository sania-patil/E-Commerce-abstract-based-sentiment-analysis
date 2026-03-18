"""
Configuration loader and validator for the ABSA pipeline.
Loads config.yaml and raises descriptive errors for invalid values.
"""

import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    bert_checkpoint: str
    learning_rate: float
    batch_size: int
    num_epochs: int
    dropout_rate: float
    max_seq_length: int
    train_split: float
    val_split: float
    test_split: float
    confidence_threshold: float
    random_seed: int


_REQUIRED_KEYS = [
    "bert_checkpoint", "learning_rate", "batch_size", "num_epochs",
    "dropout_rate", "max_seq_length", "train_split", "val_split",
    "test_split", "confidence_threshold", "random_seed",
]


def load_config(path: str | Path = "config.yaml") -> TrainingConfig:
    """Load and validate configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping, got: {type(raw)}")

    # Check for missing keys
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    cfg = TrainingConfig(**{k: raw[k] for k in _REQUIRED_KEYS})

    # Validate ranges
    if cfg.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got: {cfg.learning_rate}")
    if cfg.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got: {cfg.batch_size}")
    if cfg.num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got: {cfg.num_epochs}")
    if not (0.0 < cfg.dropout_rate < 1.0):
        raise ValueError(f"dropout_rate must be in (0, 1), got: {cfg.dropout_rate}")
    if cfg.max_seq_length < 1 or cfg.max_seq_length > 512:
        raise ValueError(f"max_seq_length must be in [1, 512], got: {cfg.max_seq_length}")
    if not (0.0 < cfg.confidence_threshold < 1.0):
        raise ValueError(f"confidence_threshold must be in (0, 1), got: {cfg.confidence_threshold}")

    split_sum = round(cfg.train_split + cfg.val_split + cfg.test_split, 10)
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train_split + val_split + test_split must equal 1.0, got: {split_sum}"
        )
    for name, val in [("train_split", cfg.train_split), ("val_split", cfg.val_split), ("test_split", cfg.test_split)]:
        if not (0.0 < val < 1.0):
            raise ValueError(f"{name} must be in (0, 1), got: {val}")

    return cfg
