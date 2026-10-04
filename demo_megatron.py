"""Demonstration and Study Script for Megatron Core Tensor Parallelism.

Can be run in two ways:
1. Standalone simulated mode (no multi-process setup needed):
   python3 demo_megatron.py

2. Real multi-process distributed mode with Gloo backend (works on Mac / CPU):
   torchrun --nproc_per_node=2 demo_megatron.py
"""

import os
import torch
import torch.distributed as dist

from config import GPTConfig
from model import GPT
from megatron_core import (
    MegatronGPT,
    ColumnParallelLinear,
    RowParallelLinear,
    get_tp_rank,
    get_tp_world_size,
)


def run_distributed_test():
    """Runs when launched with torchrun or distributed environment."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize PyTorch distributed process group using 'gloo' (works on Mac/CPU!)
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    config = GPTConfig(
        vocab_size=1000,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0
    )

    # Each rank instantiates the Megatron model
    model = MegatronGPT(config)
    model.eval()

    # Generate identical input tokens on all ranks
    torch.manual_seed(42)
    dummy_input = torch.randint(0, config.vocab_size, (2, 8))
    dummy_targets = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        logits, loss = model(dummy_input, targets=dummy_targets)

    print(
        f"[Rank {rank}/{world_size}] Success! "
        f"Logits shape: {logits.shape}, Loss: {loss.item():.4f}"
    )

    if dist.is_initialized():
        dist.destroy_process_group()


def run_standalone_step_by_step_explanation():
    """Step-by-step educational demo showing how weights and tensors split and recombine."""
    print("=" * 70)
    print(" MEGATRON-LM TENSOR PARALLELISM: STEP-BY-STEP CONCEPT DEMO ")
    print("=" * 70)

    # 1. Understanding ColumnParallelLinear
    print("\n1. ColumnParallelLinear (Used for QKV projection & MLP FC1):")
    print("   Split weight W along columns (output features).")
    batch_size, seq_len, in_feat, out_feat = 1, 2, 4, 8
    X = torch.randn(batch_size, seq_len, in_feat)  # Identical input on both ranks

    # Full weight (non-parallel)
    W_full = torch.randn(out_feat, in_feat)
    Y_expected = torch.matmul(X, W_full.t())

    # Split into 2 ranks (TP=2)
    W_rank0 = W_full[: out_feat // 2, :]  # (4, 4)
    W_rank1 = W_full[out_feat // 2 :, :]  # (4, 4)

    Y_rank0 = torch.matmul(X, W_rank0.t())  # (1, 2, 4)
    Y_rank1 = torch.matmul(X, W_rank1.t())  # (1, 2, 4)

    # Concatenate across columns recovers the full output!
    Y_combined = torch.cat([Y_rank0, Y_rank1], dim=-1)
    diff = (Y_expected - Y_combined).abs().max().item()
    print(f"   - Full output shape: {Y_expected.shape}")
    print(f"   - Rank 0 slice shape: {Y_rank0.shape}, Rank 1 slice shape: {Y_rank1.shape}")
    print(f"   - Difference between single-GPU vs split: {diff:.6e} (Identical!)")

    # 2. Understanding RowParallelLinear
    print("\n2. RowParallelLinear (Used for Attention output proj & MLP FC2):")
    print("   Input is already split into [Y_rank0, Y_rank1].")
    print("   Split weight W along rows (input features).")
    W2_full = torch.randn(in_feat, out_feat)  # (4, 8)
    # Target full multiplication: Y_combined @ W2_full.t()
    Z_expected = torch.matmul(Y_combined, W2_full.t())

    # Split W2 along rows across the 2 ranks:
    W2_rank0 = W2_full[:, : out_feat // 2]  # (4, 4)
    W2_rank1 = W2_full[:, out_feat // 2 :]  # (4, 4)

    # Local partial dot products:
    Z_rank0 = torch.matmul(Y_rank0, W2_rank0.t())  # (1, 2, 4)
    Z_rank1 = torch.matmul(Y_rank1, W2_rank1.t())  # (1, 2, 4)

    # All-Reduce (Sum) across ranks!
    Z_all_reduced = Z_rank0 + Z_rank1
    diff2 = (Z_expected - Z_all_reduced).abs().max().item()
    print(f"   - Local multiplication on Rank 0 + Rank 1")
    print(f"   - All-Reduce (Sum) recovers full tensor: diff = {diff2:.6e} (Identical!)")

    # 3. Communication Summary
    print("\n3. Why Megatron is brilliant (Minimal Communication!):")
    print("   In each Transformer Block:")
    print("   ┌─────────────────────────────────────────────────────────────┐")
    print("   │ 1. Self-Attention:                                          │")
    print("   │    - QKV projection: ColumnParallel (0 communication)       │")
    print("   │    - Multi-Head Attention: Local heads (0 communication)    │")
    print("   │    - Projection: RowParallel (1 ALL-REDUCE sum)             │")
    print("   │                                                             │")
    print("   │ 2. Feed-Forward (MLP):                                      │")
    print("   │    - FC1: ColumnParallel (0 communication)                  │")
    print("   │    - GELU: Local elementwise (0 communication)              │")
    print("   │    - FC2: RowParallel (1 ALL-REDUCE sum)                    │")
    print("   └─────────────────────────────────────────────────────────────┘")
    print("   TOTAL FORWARD COMMUNICATIONS PER BLOCK = EXACTLY 2 ALL-REDUCES!")
    print("=" * 70)


if __name__ == "__main__":
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        run_distributed_test()
    else:
        run_standalone_step_by_step_explanation()
        print("\nTo test real multi-process distributed Megatron on your Mac (CPU), run:")
        print("  torchrun --nproc_per_node=2 demo_megatron.py\n")
