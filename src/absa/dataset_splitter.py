"""
Dataset Splitter for the ABSA pipeline.

Splits a list of RawRecord objects into train, validation, and test partitions.
- Default split: 80% train / 10% validation / 10% test
- Shuffles records before splitting (using the configured random seed)
- Partitions are disjoint — no record appears in more than one partition
"""

import random
from dataclasses import dataclass

from absa.models import RawRecord


@dataclass
class DatasetSplit:
    """Holds the three partitions after splitting."""
    train: list[RawRecord]
    val: list[RawRecord]
    test: list[RawRecord]


def split(
    records: list[RawRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetSplit:
    """
    Split records into train, validation, and test partitions.

    Args:
        records:     List of RawRecord objects to split.
        train_ratio: Fraction for training (default 0.8).
        val_ratio:   Fraction for validation (default 0.1).
        test_ratio:  Fraction for test (default 0.1).
        seed:        Random seed for reproducible shuffling (default 42).

    Returns:
        DatasetSplit with .train, .val, and .test lists.

    Raises:
        ValueError: If ratios don't sum to 1.0 or any ratio is <= 0.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.6f}"
        )
    for name, val in [("train_ratio", train_ratio), ("val_ratio", val_ratio), ("test_ratio", test_ratio)]:
        if val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")

    # Shuffle a copy — never mutate the original list
    shuffled = records.copy()
    random.seed(seed)
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return DatasetSplit(
        train=shuffled[:train_end],
        val=shuffled[train_end:val_end],
        test=shuffled[val_end:],
    )
