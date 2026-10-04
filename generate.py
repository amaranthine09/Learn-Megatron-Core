"""Autoregressive text generation and token sampling operations."""

from typing import Optional
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate_tokens(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Autoregressive token generation supporting temperature, Top-K, and Top-P (nucleus) sampling.

    Args:
        model: Autoregressive language model supporting forward(idx) -> (logits, loss).
        idx: (B, T) tensor of conditioning prompt tokens.
        max_new_tokens: Number of new tokens to generate.
        block_size: Context length limit of the model.
        temperature: Sampling temperature (higher = more random, lower = more deterministic).
        top_k: Optional top-k filtering cutoff.
        top_p: Optional nucleus (top-p) cumulative probability cutoff.

    Returns:
        idx: (B, T + max_new_tokens) tensor containing prompt and generated token sequence.
    """
    model.eval()

    for _ in range(max_new_tokens):
        # Crop context if it exceeds max block size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        # Forward pass to get logits for the sequence
        logits, _ = model(idx_cond)
        # Focus only on the last time step
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        # Optional Top-K sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # Optional Top-P (nucleus) sampling
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above top_p threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to keep the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float("Inf")

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample next token from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append sampled token to running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
