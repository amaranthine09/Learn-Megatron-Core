"""Megatron Core: Tensor Parallelism implementation for GPT architectures."""

from megatron_core.mappings import (
    copy_to_model_parallel_region,
    reduce_from_model_parallel_region,
    scatter_to_model_parallel_region,
    gather_from_model_parallel_region,
)
from megatron_core.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    get_tp_rank,
    get_tp_world_size,
)
from megatron_core.attention import MegatronCausalSelfAttention
from megatron_core.mlp import MegatronMLP
from megatron_core.block import MegatronTransformerBlock
from megatron_core.model import MegatronGPT

__all__ = [
    "copy_to_model_parallel_region",
    "reduce_from_model_parallel_region",
    "scatter_to_model_parallel_region",
    "gather_from_model_parallel_region",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "get_tp_rank",
    "get_tp_world_size",
    "MegatronCausalSelfAttention",
    "MegatronMLP",
    "MegatronTransformerBlock",
    "MegatronGPT",
]
