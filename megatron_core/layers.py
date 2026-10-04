"""Megatron Core Tensor Parallel layers: ColumnParallelLinear, RowParallelLinear, and VocabParallelEmbedding."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from megatron_core.mappings import (
    copy_to_model_parallel_region,
    reduce_from_model_parallel_region,
)


def get_tp_world_size() -> int:
    """Return Tensor Parallel world size."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_tp_rank() -> int:
    """Return Tensor Parallel rank."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.

    The linear layer is defined as: Y = X * W + b.
    The weight matrix W is split along its output dimension (columns):
    W = [W_1, W_2, ..., W_p]

    Forward:
        - X is replicated across all ranks.
        - Pass X through Operator f (copy_to_model_parallel_region).
        - Compute Y_i = X * W_i + b_i locally on rank i.
        - Output Y_i is a slice of Y along the last dimension.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        self.tp_world_size = get_tp_world_size()
        self.tp_rank = get_tp_rank()

        assert out_features % self.tp_world_size == 0, (
            f"out_features ({out_features}) must be divisible by tp_world_size ({self.tp_world_size})"
        )
        self.out_features_per_partition = out_features // self.tp_world_size

        # Weight shape: (out_features_per_partition, in_features)
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Standard normal init matching GPT
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Operator f: identity in forward, all-reduce in backward
        input_parallel = copy_to_model_parallel_region(x)

        # Matrix multiply: (B, T, in_features) @ (in_features, out_features_per_partition)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)

        if self.gather_output and self.tp_world_size > 1:
            # Optional: gather outputs across ranks if requested
            from megatron_core.mappings import gather_from_model_parallel_region
            return gather_from_model_parallel_region(output_parallel)

        return output_parallel


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.

    The linear layer is defined as: Y = X * W + b.
    The weight matrix W is split along its input dimension (rows):
    W = [W_1; W_2; ...; W_p]

    Forward:
        - Input X_i is partitioned along its last dimension across ranks.
        - Compute Y_i = X_i * W_i locally on rank i.
        - Pass through Operator g (reduce_from_model_parallel_region), which executes
          an all-reduce (sum) across all TP ranks: Y = sum_i(X_i * W_i).
        - Add bias *after* all-reduce to avoid redundant bias summation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel

        self.tp_world_size = get_tp_world_size()
        self.tp_rank = get_tp_rank()

        assert in_features % self.tp_world_size == 0, (
            f"in_features ({in_features}) must be divisible by tp_world_size ({self.tp_world_size})"
        )
        self.in_features_per_partition = in_features // self.tp_world_size

        # Weight shape: (out_features, in_features_per_partition)
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel and self.tp_world_size > 1:
            from megatron_core.mappings import scatter_to_model_parallel_region
            input_parallel = scatter_to_model_parallel_region(x)
        else:
            input_parallel = x

        # Local matrix multiply without bias: (B, T, in_partition) @ (in_partition, out_features)
        output_parallel = F.linear(input_parallel, self.weight, None)

        # Operator g: all-reduce sum in forward, identity in backward
        output = reduce_from_model_parallel_region(output_parallel)

        # Add bias after all-reduce
        if self.bias is not None:
            output = output + self.bias

        return output


class VocabParallelEmbedding(nn.Module):
    """
    Embedding parallelized in vocabulary dimension.

    Each TP rank stores a slice of the vocabulary:
    [vocab_start_index, vocab_end_index).
    Tokens falling outside this range are masked to zero embedding,
    then an all-reduce (sum) gathers the full embedding across ranks.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.tp_world_size = get_tp_world_size()
        self.tp_rank = get_tp_rank()

        # Partition vocab evenly across ranks
        self.vocab_per_partition = math.ceil(num_embeddings / self.tp_world_size)
        self.vocab_start = self.tp_rank * self.vocab_per_partition
        self.vocab_end = min(self.vocab_start + self.vocab_per_partition, num_embeddings)

        self.weight = nn.Parameter(
            torch.empty(self.vocab_per_partition, embedding_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_world_size == 1:
            return F.embedding(x, self.weight)

        # Create mask of tokens handled by this rank
        mask = (x >= self.vocab_start) & (x < self.vocab_end)
        # Shift index to local table range (clamp tokens outside to 0 to avoid index error)
        local_x = torch.clamp(x - self.vocab_start, 0, self.vocab_per_partition - 1)

        # Embed and zero-out tokens not owned by this rank
        output_parallel = F.embedding(local_x, self.weight)
        output_parallel = output_parallel * mask.unsqueeze(-1)

        # All-reduce sum across TP ranks
        return reduce_from_model_parallel_region(output_parallel)
