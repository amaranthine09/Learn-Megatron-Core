"""Full GPT Language Model Architecture.

Composes token & position embeddings, stacked TransformerBlocks, and language modeling head.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig
from attention import CausalSelfAttention
from mlp import FeedForward
from block import TransformerBlock
from generate import generate_tokens

# Re-export for convenience and backwards compatibility
__all__ = [
    "GPT",
    "GPTConfig",
    "TransformerBlock",
    "CausalSelfAttention",
    "FeedForward",
    "generate_tokens",
]


class GPT(nn.Module):
    """Full GPT Language Model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, elementwise_affine=config.bias),
        ))

        # Language modeling head projects back to vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between input token embedding and output lm_head
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Standard GPT weight initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.elementwise_affine:
                torch.nn.init.ones_(module.weight)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            idx: (B, T) tensor of token indices.
            targets: Optional (B, T) tensor of ground-truth token indices for loss computation.

        Returns:
            logits: (B, T, vocab_size) if targets is provided, or (B, 1, vocab_size) at inference
            loss: Optional scalar cross-entropy loss
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        # Positions: (T,) -> (1, T)
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        # Forward through embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (1, T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final LayerNorm
        x = self.transformer.ln_f(x)

        # Language modeling head
        if targets is not None:
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference optimization: compute logits only for the last position
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Autoregressive token generation delegating to generate_tokens."""
        return generate_tokens(
            model=self,
            idx=idx,
            max_new_tokens=max_new_tokens,
            block_size=self.config.block_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )


if __name__ == "__main__":
    # Example usage / Sanity check with a small configuration
    small_config = GPTConfig(
        vocab_size=1000,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1
    )

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Instantiating model on device: {device}")

    model = GPT(small_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params / 1e6:.2f}M")

    # 1. Test forward pass with loss computation
    batch_size, seq_len = 2, 16
    dummy_input = torch.randint(0, small_config.vocab_size, (batch_size, seq_len), device=device)
    dummy_targets = torch.randint(0, small_config.vocab_size, (batch_size, seq_len), device=device)

    logits, loss = model(dummy_input, targets=dummy_targets)
    print(f"Logits shape: {logits.shape}, Loss: {loss.item():.4f}")

    # 2. Test autoregressive generation
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=10)
    print(f"Prompt + Generated token IDs: {generated.tolist()}")
