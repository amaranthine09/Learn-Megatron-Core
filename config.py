"""Configuration for the GPT model architecture."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    """Configuration for the GPT model."""
    vocab_size: int = 50257     # Vocabulary size (e.g. GPT-2 default)
    block_size: int = 1024      # Context length / max sequence length
    n_layer: int = 12           # Number of transformer blocks
    n_head: int = 12            # Number of attention heads
    n_embd: int = 768           # Embedding dimension
    dropout: float = 0.1        # Dropout rate
    bias: bool = True           # Use bias in Linear and LayerNorm layers

    # Architecture Presets
    @classmethod
    def gpt2_small(cls) -> "GPTConfig":
        """GPT-2 Small: 124M parameters."""
        return cls(n_layer=12, n_head=12, n_embd=768)

    @classmethod
    def gpt2_medium(cls) -> "GPTConfig":
        """GPT-2 Medium: 350M parameters."""
        return cls(n_layer=24, n_head=16, n_embd=1024)

    @classmethod
    def gpt2_large(cls) -> "GPTConfig":
        """GPT-2 Large: 774M parameters."""
        return cls(n_layer=36, n_head=20, n_embd=1280)

    @classmethod
    def gpt2_xl(cls) -> "GPTConfig":
        """GPT-2 XL: 1558M parameters."""
        return cls(n_layer=48, n_head=25, n_embd=1600)
