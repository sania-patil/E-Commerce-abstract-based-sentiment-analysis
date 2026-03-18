"""
BERT Tokenizer wrapper for the ABSA pipeline.

Wraps HuggingFace BertTokenizerFast to produce TokenizerOutput objects.
Handles:
- [CLS] and [SEP] special tokens
- Attention masks
- Word-to-subword alignment mapping (needed for BIO tag alignment)
- BIO tag alignment: first subword token of each word gets the word's tag,
  all subsequent subword tokens get O (label index 0)
- Truncation at 512 tokens with a logged warning
- Batch padding to the longest sequence in the batch
"""

import logging
from typing import Optional

from transformers import BertTokenizerFast

from absa.models import TokenizerOutput

logger = logging.getLogger(__name__)

# BIO tag label mapping
BIO_LABEL2ID = {"O": 0, "B-ASP": 1, "I-ASP": 2}
BIO_ID2LABEL = {v: k for k, v in BIO_LABEL2ID.items()}

_MAX_LENGTH = 512


class ABSATokenizer:
    """
    Wraps BertTokenizerFast for the ABSA pipeline.

    Usage:
        tokenizer = ABSATokenizer()
        output = tokenizer.tokenize("the battery life is excellent")
        batch  = tokenizer.tokenize_batch(["review one", "review two"])
    """

    def __init__(self, checkpoint: str = "bert-base-uncased"):
        self._tokenizer = BertTokenizerFast.from_pretrained(checkpoint)

    def tokenize(
        self,
        text: str,
        word_labels: Optional[list[str]] = None,
    ) -> TokenizerOutput:
        """
        Tokenize a single review string.

        Args:
            text:        Preprocessed review string.
            word_labels: Optional list of BIO tag strings per word (training only).
                         e.g. ["O", "B-ASP", "I-ASP", "O"]
                         Must have the same length as the number of whitespace-split words.

        Returns:
            TokenizerOutput with input_ids, attention_mask, word_ids, and
            optionally bio_tags aligned to subword tokens.
        """
        encoding = self._tokenizer(
            text,
            is_split_into_words=False,
            truncation=True,
            max_length=_MAX_LENGTH,
            padding=False,
            return_offsets_mapping=False,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        word_ids = encoding.word_ids()  # None for [CLS]/[SEP], int for real tokens

        # Warn if truncation occurred
        words = text.split()
        if len(words) > _MAX_LENGTH - 2:  # -2 for [CLS] and [SEP]
            logger.warning(
                "Review truncated to %d tokens (original ~%d words): '%s...'",
                _MAX_LENGTH, len(words), text[:50],
            )

        # Align BIO tags to subword tokens if provided
        bio_tags = None
        if word_labels is not None:
            bio_tags = _align_bio_tags(word_ids, word_labels)

        return TokenizerOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_ids=word_ids,
            bio_tags=bio_tags,
        )

    def tokenize_batch(
        self,
        texts: list[str],
        word_labels_batch: Optional[list[list[str]]] = None,
    ) -> list[TokenizerOutput]:
        """
        Tokenize a batch of review strings, padding to the longest sequence.

        Args:
            texts:              List of preprocessed review strings.
            word_labels_batch:  Optional list of per-review word label lists (training only).

        Returns:
            List of TokenizerOutput objects, all padded to the same length.
        """
        encoding = self._tokenizer(
            texts,
            is_split_into_words=False,
            truncation=True,
            max_length=_MAX_LENGTH,
            padding=True,   # pad to longest in batch
            return_offsets_mapping=False,
        )

        results = []
        for i, text in enumerate(texts):
            word_ids = encoding.word_ids(batch_index=i)
            input_ids = encoding["input_ids"][i]
            attention_mask = encoding["attention_mask"][i]

            bio_tags = None
            if word_labels_batch is not None:
                bio_tags = _align_bio_tags(word_ids, word_labels_batch[i])

            results.append(TokenizerOutput(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_ids=word_ids,
                bio_tags=bio_tags,
            ))

        return results


def _align_bio_tags(
    word_ids: list[Optional[int]],
    word_labels: list[str],
) -> list[int]:
    """
    Align word-level BIO labels to subword token positions.

    Rule:
    - [CLS] and [SEP] tokens (word_id is None) → O (0)
    - First subword token of a word → that word's label
    - Subsequent subword tokens of the same word → O (0)

    Args:
        word_ids:    List mapping each token position to its word index (or None).
        word_labels: List of BIO label strings per original word.

    Returns:
        List of integer label IDs, one per token position.
    """
    aligned = []
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # Special token ([CLS] or [SEP])
            aligned.append(BIO_LABEL2ID["O"])
        elif word_id != prev_word_id:
            # First subword token of this word — use the word's label
            label = word_labels[word_id] if word_id < len(word_labels) else "O"
            aligned.append(BIO_LABEL2ID.get(label, BIO_LABEL2ID["O"]))
        else:
            # Subsequent subword token of the same word → O
            aligned.append(BIO_LABEL2ID["O"])

        prev_word_id = word_id

    return aligned
