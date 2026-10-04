"""Megatron Core Tensor Parallel Feed-Forward Network (MLP).

Architecture:
1. First Linear (c_fc): ColumnParallelLinear
   - Expands dimension from n_embd -> (4 * n_embd) // tp_world_size.
   - No communication during forward pass.
2. Activation: GELU
   - Applied elementwise independently on each local partition.
   - Absolutely zero communication needed.
3. Second Linear (c_proj): RowParallelLinear
   - Projects from (4 * n_embd) // tp_world_size -> n_embd.
   - Executes 1 All-Reduce (Operator g) to sum contributions across all TP ranks.
"""

import torch
import torch.nn as nn

from config import GPTConfig
from megatron_core.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)


class MegatronMLP(nn.Module):
    """Tensor-Parallel Feed-Forward Network."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # Column-parallel expands hidden dim
        self.c_fc = ColumnParallelLinear(
            in_features=config.n_embd,
            out_features=4 * config.n_embd,
            bias=config.bias,
            gather_output=False,
        )
        self.gelu = nn.GELU(approximate="tanh")
        # Row-parallel projects back and executes the single all-reduce
        self.c_proj = RowParallelLinear(
            in_features=4 * config.n_embd,
            out_features=config.n_embd,
            bias=config.bias,
            input_is_parallel=True,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Column-parallel projection (local)
        x = self.c_fc(x)
        # 2. Elementwise activation (local)
        x = self.gelu(x)
        # 3. Row-parallel projection with forward All-Reduce
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
