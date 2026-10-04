"""Megatron / GPT Architecture Package.

Provides modular components for autoregressive Transformer models:
- GPTConfig: Configuration dataclass and presets
- CausalSelfAttention: Scaled dot-product multi-head causal attention
- FeedForward: Position-wise MLP with GELU
- TransformerBlock: Pre-LN residual transformer block
- GPT: Full language model
- generate_tokens: Autoregressive sampling and text generation
"""

from config import GPTConfig
from attention import CausalSelfAttention
from mlp import FeedForward
from block import TransformerBlock
from model import GPT
from generate import generate_tokens

__all__ = [
    "GPTConfig",
    "CausalSelfAttention",
    "FeedForward",
    "TransformerBlock",
    "GPT",
    "generate_tokens",
]
