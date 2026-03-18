"""
BERT Encoder for the ABSA pipeline.

Loads a pre-trained BertModel and produces contextual embeddings
(hidden states) for each token in the input sequence.

Output shape: [batch_size, seq_len, hidden_dim]
  - bert-base-uncased: hidden_dim = 768
  - bert-large-uncased: hidden_dim = 1024

Two modes:
  - Training mode  : dropout active (model.train())
  - Inference mode : dropout disabled, deterministic (model.eval())
"""

import torch
import torch.nn as nn
from transformers import BertModel

from absa.models import TokenizerOutput


class BERTEncoder(nn.Module):
    """
    Wraps a pre-trained BertModel to produce contextual embeddings.

    Usage:
        encoder = BERTEncoder()                    # loads bert-base-uncased
        encoder.set_inference_mode()               # disable dropout
        embeddings = encoder(tokenizer_output)     # shape: [batch, seq_len, 768]
    """

    def __init__(self, checkpoint: str = "bert-base-uncased", dropout_rate: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.hidden_dim = self.bert.config.hidden_size  # 768 for base, 1024 for large

    def forward(self, tokenizer_output: TokenizerOutput) -> torch.Tensor:
        """
        Produce contextual embeddings for a batch of tokenized inputs.

        Args:
            tokenizer_output: TokenizerOutput with input_ids and attention_mask
                              as lists or tensors. Can be a single sequence or
                              a batch (2D lists/tensors).

        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim] with contextual
            embeddings for each token position.
        """
        input_ids = _to_tensor(tokenizer_output.input_ids)
        attention_mask = _to_tensor(tokenizer_output.attention_mask)

        # Ensure batch dimension exists
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # last_hidden_state: [batch, seq_len, hidden_dim]
        hidden_states = outputs.last_hidden_state

        # Apply dropout (active in training mode, disabled in eval mode)
        return self.dropout(hidden_states)

    def set_training_mode(self):
        """Enable dropout — use during model training."""
        self.train()

    def set_inference_mode(self):
        """Disable dropout, deterministic output — use during inference."""
        self.eval()


def _to_tensor(data) -> torch.Tensor:
    """Convert list or tensor to a LongTensor."""
    if isinstance(data, torch.Tensor):
        return data.long()
    return torch.tensor(data, dtype=torch.long)
