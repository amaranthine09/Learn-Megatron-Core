"""Feed-Forward Network (MLP) operation for Transformer architectures."""

import torch
import torch.nn as nn

from config import GPTConfig


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (MLP).
    Expands dimension to 4 * n_embd and applies GELU non-linearity.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
