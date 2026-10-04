"""Full Megatron Core Tensor Parallel GPT Model."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig
from megatron_core.layers import (
    ColumnParallelLinear,
    VocabParallelEmbedding,
    get_tp_world_size,
    get_tp_rank,
)
from megatron_core.block import MegatronTransformerBlock


class MegatronGPT(nn.Module):
    """
    GPT Language Model implemented with Megatron-LM 1D Tensor Parallelism.

    Key Parallelism Properties:
    - Token Embedding: VocabParallelEmbedding (vocab dimension split across TP ranks).
    - Position Embedding: Replicated across TP ranks.
    - Transformer Blocks:
        - Attention: ColumnParallel QKV -> Local Attention -> RowParallel Proj (All-Reduce).
        - MLP: ColumnParallel FC1 -> Local GELU -> RowParallel FC2 (All-Reduce).
    - LM Head: ColumnParallelLinear with output gathered for full logits.
    - LayerNorm: Replicated across TP ranks.
    """

    def __init__(self, config: GPTConfig, vocab_parallel: bool = True):
        super().__init__()
        self.config = config
        self.vocab_parallel = vocab_parallel
        self.tp_world_size = get_tp_world_size()
        self.tp_rank = get_tp_rank()

        # Word Token Embedding
        if vocab_parallel and self.tp_world_size > 1:
            wte = VocabParallelEmbedding(config.vocab_size, config.n_embd)
        else:
            wte = nn.Embedding(config.vocab_size, config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            wte=wte,
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([MegatronTransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, elementwise_affine=config.bias),
        ))

        # LM Head: column parallel with gathered output so full logits are returned
        if self.tp_world_size > 1:
            self.lm_head = ColumnParallelLinear(
                in_features=config.n_embd,
                out_features=config.vocab_size,
                bias=False,
                gather_output=True,
            )
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying if single process
        if self.tp_world_size == 1:
            self.transformer.wte.weight = self.lm_head.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        # Forward embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward through tensor-parallel transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final LayerNorm
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss
