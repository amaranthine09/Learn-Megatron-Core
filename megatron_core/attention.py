"""Megatron Core Tensor Parallel Multi-Head Causal Self-Attention.

Architecture:
1. QKV Linear: ColumnParallelLinear
   - Splits heads evenly across TP ranks (n_head // tp_world_size per rank).
   - No communication during forward pass.
2. Local Attention:
   - Each rank computes scaled dot-product attention for its own subset of heads.
   - Absolutely zero communication required across ranks.
3. Output Linear: RowParallelLinear
   - Takes partitioned head outputs and projects to n_embd.
   - Executes 1 All-Reduce (Operator g) to sum contributions from all ranks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig
from megatron_core.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    get_tp_world_size,
    get_tp_rank,
)


class MegatronCausalSelfAttention(nn.Module):
    """Tensor-Parallel Causal Self-Attention block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.tp_world_size = get_tp_world_size()
        self.tp_rank = get_tp_rank()

        assert config.n_head % self.tp_world_size == 0, (
            f"n_head ({config.n_head}) must be divisible by tp_world_size ({self.tp_world_size})"
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.num_heads_per_partition = config.n_head // self.tp_world_size
        self.dropout = config.dropout

        # Column-parallel QKV projection: projects to (3 * n_embd // tp_world_size)
        self.c_attn = ColumnParallelLinear(
            in_features=config.n_embd,
            out_features=3 * config.n_embd,
            bias=config.bias,
            gather_output=False,
        )

        # Row-parallel output projection: takes partitioned head output and reduces to full n_embd
        self.c_proj = RowParallelLinear(
            in_features=config.n_embd,
            out_features=config.n_embd,
            bias=config.bias,
            input_is_parallel=True,
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # 1. Column-parallel QKV projection: (B, T, 3 * (num_heads_per_partition * head_dim))
        qkv = self.c_attn(x)
        partition_dim = self.num_heads_per_partition * self.head_dim
        q, k, v = qkv.split(partition_dim, dim=2)

        # Reshape to (B, num_heads_per_partition, T, head_dim)
        q = q.view(B, T, self.num_heads_per_partition, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads_per_partition, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads_per_partition, self.head_dim).transpose(1, 2)

        # 2. Local Causal Attention for this rank's heads (No communication!)
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True
        )

        # Concatenate local heads: (B, T, num_heads_per_partition * head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, partition_dim)

        # 3. Row-parallel output projection (Contains the ONLY All-Reduce in the attention block)
        y = self.resid_dropout(self.c_proj(y))
        return y
