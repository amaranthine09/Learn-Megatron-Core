"""Transformer Block combining Attention and MLP with Pre-LayerNorm and residual connections."""

import torch
import torch.nn as nn

from config import GPTConfig
from attention import CausalSelfAttention
from mlp import FeedForward


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block containing Attention and MLP with residual connections."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm architecture (modern GPT standard)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
