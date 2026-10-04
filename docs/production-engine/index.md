# Megatron Core Production Engine Architecture
> **Composable ModuleSpec, Comm-Compute Overlap, FP8 Delayed Scaling & Distributed Checkpointing**

> **Reference**: *NVIDIA Megatron Core Architecture Specification & Developer Guide (2023–2025)*

> [!NOTE]
> **Prerequisites Refresher**:
> Before reading [Production Engine Architecture](/production-engine/), ensure you have mastered:
> 1. **Foundations ([Distributed Foundations & Interconnects](/foundations/))**: Hardware interconnects, NVLink vs InfiniBand, and autograd conjugate operators.
> 2. **3D Parallelism (Books 2, 4, 5)**: Composition of Tensor, Pipeline, and Distributed Optimizer Data Parallelism.
> 3. **Long Context & MoE (Books 3, 6)**: Sequence Parallelism, Ring Attention, and Expert Parallelism All-to-All communication.

---

## 1.1. The Evolution: Legacy Megatron-LM vs Megatron Core (M-Core)

Historically, NVIDIA's `Megatron-LM` repository was a monolithic research codebase:
- Model architectures (GPT, BERT, T5) were hardcoded directly with parallelism calls embedded into every class.
- Swapping an activation function, adding a new attention mechanism (like RoPE, ALiBi, or GQA), or plugging in a custom kernel required rewriting fundamental distributed primitives.

### 1.1.1 The Megatron Core (M-Core) Revolution:
Starting in 2023, NVIDIA completely refactored the system into **Megatron Core (M-Core)**:
1. **Decoupled Architecture & Parallelism**: Models are defined using declarative **Specifications (`spec.py`)**. You can configure a standard PyTorch module, a Tensor Parallel module, or a fused TransformerEngine module without modifying a single line of model code!
2. **Unified Parallelism Support**: Native composition of **TP, SP, PP, DP, CP, and EP** into a clean 5D parallel execution grid.
3. **Hardware Acceleration via Transformer Engine (TE)**: Direct integration of fused FP8 GEMMs, FlashAttention-3, and zero-overhead communication overlap.

---

## 1.2. Declarative Module Specifications in M-Core

In M-Core, a Transformer layer is not a fixed monolithic class. It is constructed declaratively from a **`TransformerConfig`** and a **`ModuleSpec`**:

```python
"""
Megatron Core Declarative Layer Specification and Configuration
Demonstrates the real megatron.core.transformer APIs for composing TP, SP, and TE layers.
"""
import torch
from dataclasses import dataclass
from typing import Optional

# ── 1. The Real Megatron Core TransformerConfig API ──
# from megatron.core.transformer.transformer_config import TransformerConfig
#
# A single centralized configuration object that defines model dimensions,
# precision, and every single parallelism strategy across the 5D grid:

from megatron.core.transformer.transformer_config import TransformerConfig

config = TransformerConfig(
    # Model Geometry
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    num_query_groups=8,              # Grouped-Query Attention (GQA): 8 KV heads
    ffn_hidden_size=14336,           # SwiGLU hidden dim (typically 3.5 * hidden_size)
    kv_channels=128,                 # head_dim = hidden_size / num_attention_heads
    
    # 5D Parallelism Strategy
    tensor_model_parallel_size=4,    # TP = 4
    pipeline_model_parallel_size=2,  # PP = 2
    sequence_parallel=True,          # Sequence Parallelism (SP) enabled!
    context_parallel_size=2,         # Context Parallelism (CP) for long context
    
    # Precision & Kernels
    bf16=True,                       # Native BF16 mixed-precision
    params_dtype=torch.bfloat16,
    pipeline_dtype=torch.bfloat16,
    
    # Activation Memory Management
    recompute_granularity='selective', # Selective activation recomputation ([Sequence Parallelism](/sequence-parallelism/))
    recompute_method='uniform',
    
    # Performance & Comm Overlap
    tp_comm_overlap=True,            # Micro-tiled Comm-Compute GEMM overlap
)


# ── 2. Declarative Module Specifications (ModuleSpec) ──
# from megatron.core.transformer.spec_utils import ModuleSpec
# from megatron.core.transformer.custom_layers.transformer_engine import (
#     TELinear, TEDotProductAttention, TENorm
# )
# from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
# from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
# from megatron.core.transformer.mlp import MLP, MLPSubmodules
# from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

# Option A: Standard PyTorch Native Spec (Runs with standard torch ops)
native_layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm=torch.nn.LayerNorm,
        self_attention=ModuleSpec(
            module=SelfAttention,
            submodules=SelfAttentionSubmodules(
                linear_qkv=ColumnParallelLinear,
                core_attention=DotProductAttention,
                linear_proj=RowParallelLinear,
            ),
        ),
        pre_mlp_layernorm=torch.nn.LayerNorm,
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
            ),
        ),
    ),
)

# Option B: NVIDIA Transformer Engine (TE) Spec (Hardware-Fused FP8 kernels)
te_layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm=TENorm,                       # Fused RMSNorm / LayerNorm
        self_attention=ModuleSpec(
            module=SelfAttention,
            submodules=SelfAttentionSubmodules(
                linear_qkv=TELinear,                  # Fused FP8 Column-Parallel GEMM
                core_attention=TEDotProductAttention, # Fused FlashAttention-3 FP8 kernel
                linear_proj=TELinear,                 # Fused FP8 Row-Parallel GEMM
            ),
        ),
        pre_mlp_layernorm=TENorm,
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TELinear,
                linear_fc2=TELinear,
            ),
        ),
    ),
)

# Building a layer from spec:
# layer = TransformerLayer(config=config, submodules=te_layer_spec.submodules)
# hidden_states = layer(hidden_states, attention_mask=causal_mask)
```

This declarative decoupling is what makes Megatron Core radically superior to legacy Megatron-LM: you can switch from native PyTorch debugging to NVIDIA Hopper/Blackwell hardware-fused FP8 kernels with **zero code modifications to the Transformer block logic**.

---

## 1.3. Communication-Computation Overlap (Comm-Compute Overlap)

In Books 2 and 3, we treated computation and communication as sequential steps:
$$\text{Time}_{\text{block}} = \text{Time}_{\text{GEMM}} + \text{Time}_{\text{All-Reduce}}$$

Even with high-speed NVLink, communication takes $15 - 25\%$ of each step's time.

### 1.3.1 The M-Core Solution: Micro-Tiling GEMMs
In Megatron Core, large GEMMs are split along the sequence or batch dimension into **independent tiles**:

```
                       Traditional Sequential Execution
                       
  CUDA Stream 0 (Compute): [       Full GEMM       ]
  CUDA Stream 1 (Comm):                             [   Full All-Reduce   ]
  Time ────────────────────────────────────────────────────────────────────────>
  
  
                       M-Core Comm-Compute Overlap
                       
  CUDA Stream 0 (Compute): [ GEMM Tile 0 ] [ GEMM Tile 1 ]
  CUDA Stream 1 (Comm):                    [ Comm Tile 0 ] [ Comm Tile 1 ]
  Time ────────────────────────────────────────────────────────────────────────>
```

### 1.3.2 The Overlap Mechanics:
1. Stream 0 executes **GEMM Tile 0**.
2. As soon as Tile 0 completes, Stream 1 launches the asynchronous **Reduce-Scatter / All-Reduce on Tile 0**!
3. Simultaneously, Stream 0 starts computing **GEMM Tile 1**!
4. By the time GEMM Tile 1 finishes, Tile 0's communication is already done!

**Observable Result**: Communication latency is almost completely hidden behind computation, achieving **$>90\%$ of theoretical peak GPU throughput (MFU)**!

```python
"""
Communication-Computation Overlap Implementation with Dual CUDA Streams.
Demonstrates the exact micro-tiling pipeline used in Megatron Core and Transformer Engine.
"""
import torch
import torch.distributed as dist

class PipelinedRowParallelLinearWithOverlap(torch.nn.Module):
    """
    Splits the Row-Parallel GEMM + All-Reduce into K micro-tiles.
    Overlaps GEMM(Tile k) on compute_stream with AllReduce(Tile k-1) on comm_stream.
    """
    def __init__(self, in_features_per_partition: int, out_features: int, num_tiles: int = 2):
        super().__init__()
        self.in_features = in_features_per_partition
        self.out_features = out_features
        self.num_tiles = num_tiles
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features_per_partition)
        )
        torch.nn.init.xavier_normal_(self.weight)
        
        # In real GPU execution, we create dedicated CUDA streams:
        # self.compute_stream = torch.cuda.current_stream()
        # self.comm_stream    = torch.cuda.Stream()

    def forward(self, x: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
        """
        x: [Batch, Seq_len, in_features_per_partition]
        Splits sequence dimension into `num_tiles` chunks.
        """
        # Split along sequence dimension into micro-tiles
        x_tiles = list(torch.chunk(x, chunks=self.num_tiles, dim=1))
        gemm_outputs = []
        comm_work_handles = []

        is_cuda = x.is_cuda and torch.cuda.is_available()
        compute_stream = torch.cuda.current_stream() if is_cuda else None
        comm_stream = torch.cuda.Stream() if is_cuda else None

        for k in range(self.num_tiles):
            # 1. Compute GEMM for Tile k on the Compute Stream
            # local_tile: [B, S/k, Out]
            local_tile = torch.nn.functional.linear(x_tiles[k], self.weight)
            gemm_outputs.append(local_tile)

            if is_cuda and comm_stream is not None:
                # Synchronize: comm_stream must wait for Tile k GEMM to finish
                comm_stream.wait_stream(compute_stream)
                
                # 2. Launch non-blocking All-Reduce for Tile k on Comm Stream
                with torch.cuda.stream(comm_stream):
                    work = dist.all_reduce(local_tile, group=group, async_op=True)
                    comm_work_handles.append(work)
            else:
                # CPU / Gloo fallback: synchronous or standard dist
                if dist.is_initialized():
                    dist.all_reduce(local_tile, group=group)

        # 3. Synchronize streams before passing output to downstream layers
        if is_cuda and comm_stream is not None:
            # Wait for all async collective handles
            for work in comm_work_handles:
                work.wait()
            # Main compute stream waits for all communication to complete
            compute_stream.wait_stream(comm_stream)

        # Concatenate the communicated tiles along sequence dimension
        return torch.cat(gemm_outputs, dim=1)


# ── In Megatron Core Production ──
# Megatron Core enables this transparently when using TransformerEngine specs:
# config.tp_comm_overlap = True
# 
# Transformer Engine internally partitions the GEMM into CUTLASS warp-specialized
# tiles and overlaps them directly with NCCL ring/NVLS collectives at the C++/CUDA kernel level!
```

---

