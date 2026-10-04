"""Megatron Core Tensor Parallel Transformer Block."""

import torch
import torch.nn as nn

from config import GPTConfig
from megatron_core.attention import MegatronCausalSelfAttention
from megatron_core.mlp import MegatronMLP


class MegatronTransformerBlock(nn.Module):
    """
    Tensor-Parallel Transformer Block.

    Notice that:
    - LayerNorm layers operate on identical, full hidden representations on all ranks.
    - Each block performs exactly 2 All-Reduces in the forward pass:
      1 inside MegatronCausalSelfAttention (at RowParallel c_proj)
      1 inside MegatronMLP (at RowParallel c_proj)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.attn = MegatronCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.mlp = MegatronMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
