# FP8 Precision & Distributed Checkpointing
> **FP8 Delayed Scaling (E4M3/E5M2), Transformer Engine Integration, and Sharded State Dicts**

---

## 2.1. FP8 Mixed Precision Training

Modern accelerators (NVIDIA Hopper H100, Blackwell B200) feature **FP8 Tensor Cores**, doubling throughput over FP16:
- **FP16 / BF16**: $\sim 989\text{ TFLOPs}$ (H100 SXM)
- **FP8**: $\sim 1{,}978\text{ TFLOPs}$ (H100 SXM — **$2\times$ faster!**)

However, an 8-bit float has only 256 representable numbers! Naive FP8 training instantly causes overflow or underflow.

### 2.1.1 The Two FP8 Formats
IEEE 754 defines two distinct 8-bit formats:

```
1. E4M3 (1 sign bit, 4 exponent bits, 3 mantissa bits):
   [ S | E E E E | M M M ]
   - Higher precision (3 mantissa bits)
   - Smaller dynamic range (~ -448 to 448)
   - Used for: FORWARD PASS (Weights and Activations)

2. E5M2 (1 sign bit, 5 exponent bits, 2 mantissa bits):
   [ S | E E E E E | M M ]
   - Identical dynamic range to FP16 (5 exponent bits)
   - Lower precision (2 mantissa bits)
   - Used for: BACKWARD PASS (Gradients have wide dynamic range!)
```

---

### 2.1.2 Delayed Scaling Algorithm in M-Core
To prevent numerical clipping, tensors are scaled dynamically by a scaling factor $S$:
$$X_{fp8} = \text{clip}\left(\text{round}\left(X \times S\right)\right)$$

Calculating the optimal scale factor $S = \frac{\text{FP8\_MAX}}{\max(|X|)}$ requires scanning the entire tensor, which introduces memory synchronization stalls.

M-Core uses **Delayed Scaling**:
- It maintains a **history buffer** of the maximum absolute values ($\text{amax}$) over the last $N$ iterations (typically $N = 16$).
- The scale factor for the current step is computed using the **historical maximum**:
  $$S_t = \frac{\text{FP8\_MAX}}{\max(\text{history}_{t-1})}$$
- This removes all GPU stalls, allowing FP8 matrix multiplies to run at maximum hardware speed!

```python
"""
FP8 Delayed Scaling Mechanics & Transformer Engine Integration.
Demonstrates:
1. NVIDIA Transformer Engine recipe setup
2. Standalone Delayed Scaling history buffer implementation in pure PyTorch
"""
import torch

# ── 1. The Real Transformer Engine & Megatron Core FP8 API ──
# import transformer_engine.pytorch as te
# from transformer_engine.common.recipe import DelayedScaling, Format

# Define FP8 recipe with Delayed Scaling
# fp8_recipe = DelayedScaling(
#     margin=0,                      # Headroom margin (in bits)
#     interval=1,                    # How often to update scale factors (every step)
#     fp8_format=Format.HYBRID,      # E4M3 for forward, E5M2 for backward
#     amax_history_len=16,           # Circular buffer of 16 past iterations
#     amax_compute_algo="max",       # Use maximum value across history window
# )
#
# # Train inside the FP8 autocast context:
# with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
#     output = model(input_ids)
#     loss = loss_fn(output, labels)
# loss.backward()


# ── 2. Standalone Reference: Delayed Scaling Buffer from Scratch ──
class DelayedScalingTracker:
    """
    Simulates the Transformer Engine / M-Core Delayed Scaling mechanism.
    Avoids host-device stalls by using past amax values to compute current scale factor.
    """
    FP8_E4M3_MAX = 448.0   # Maximum representable value in FP8 E4M3
    FP8_E5M2_MAX = 57344.0 # Maximum representable value in FP8 E5M2

    def __init__(self, history_len: int = 16, margin: float = 0.0):
        self.history_len = history_len
        self.margin = margin
        self.history = torch.zeros(history_len)
        self.step_count = 0
        self.current_scale = 1.0

    def compute_scale(self) -> float:
        """Computes scale factor using the historical maximum amax from past steps."""
        if self.step_count == 0:
            return 1.0
        
        valid_steps = min(self.step_count, self.history_len)
        past_max_amax = self.history[:valid_steps].max().item()
        
        if past_max_amax <= 1e-12:
            return 1.0

        # Scale factor S such that: max_val * S <= FP8_MAX * 2^(-margin)
        target_max = self.FP8_E4M3_MAX * (2.0 ** (-self.margin))
        scale = target_max / past_max_amax
        return scale

    def quantize_and_record(self, tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        1. Quantizes tensor to simulated FP8 using DELAYED scale factor.
        2. Records the current tensor's amax into the history buffer for FUTURE steps.
        """
        # Step A: Use scale computed from PAST history
        scale = self.compute_scale()
        
        # Step B: Scale and clamp to simulated FP8 range
        scaled_tensor = tensor * scale
        clamped_tensor = torch.clamp(scaled_tensor, -self.FP8_E4M3_MAX, self.FP8_E4M3_MAX)
        
        # In real hardware, cast to torch.float8_e4m3fn
        # fp8_tensor = clamped_tensor.to(torch.float8_e4m3fn)

        # Step C: Record CURRENT step's amax for subsequent steps (asynchronous on GPU)
        current_amax = tensor.abs().max().detach()
        idx = self.step_count % self.history_len
        self.history[idx] = current_amax
        self.step_count += 1

        return clamped_tensor / scale, scale
```

---

## 2.2. Distributed Checkpointing: Sharded State Dicts

Saving a 70B or 405B parameter model to disk in a distributed cluster presents severe engineering challenges:
- In pure PyTorch, gathering the full model to Rank 0 to write `torch.save(model.state_dict())` causes an immediate Host CPU Out-Of-Memory crash.
- Saving raw per-GPU checkpoints tightly couples the saved file to the specific cluster configuration: if you train on 64 GPUs with $\text{TP}=8, \text{PP}=4$, you **cannot resume on 32 GPUs with $\text{TP}=4, \text{PP}=2$!**

### 2.2.1 Megatron Core Sharded State Dict:
M-Core implements **Fully Reshardable Distributed Checkpointing**:
1. Each GPU saves its local parameter shards with explicit global coordinate metadata:
   ```json
   {
     "param_name": "transformer.layers.0.mlp.c_fc.weight",
     "global_shape": [32768, 8192],
     "local_slice": [0, 4096, 0, 8192],
     "data_file": "shard_0001.bin"
   }
   ```
2. When loading the checkpoint on a completely different cluster topology (e.g., changing TP size from 8 to 2 or running on a single Mac CPU for evaluation), M-Core reads the global coordinate metadata, slices the files, and reconstructs the correct local tensors automatically!

```python
"""
Megatron Core Distributed Checkpointing (dist_checkpointing) & Resharding.
Demonstrates the real M-Core APIs for sharded state dict creation, saving, and topology resharding.
"""
import os
import torch
from dataclasses import dataclass
from typing import Dict, Any

# ── 1. The Real Megatron Core Checkpointing API ──
# from megatron.core import dist_checkpointing
# from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedObject
#
# # In each model layer (e.g. ColumnParallelLinear):
# def sharded_state_dict(self, prefix=''):
#     return {
#         f"{prefix}weight": ShardedTensor.from_rank_offsets(
#             key=f"{prefix}weight",
#             data=self.weight,
#             # Global unpartitioned shape: [out_features_global, in_features]
#             # Offset along partitioned dim (dim 0 for ColumnParallel):
#             (0, self.rank * self.out_features_per_partition, self.out_features_global)
#         )
#     }
#
# # Saving checkpoint across the entire cluster asynchronously:
# dist_checkpointing.save(
#     sharded_state_dict=model.sharded_state_dict(),
#     checkpoint_dir="/checkpoints/step_100000"
# )
#
# # Loading checkpoint on a completely different cluster (e.g. TP=2 instead of TP=8):
# # M-Core automatically matches ShardedTensor keys, reads the intersecting slices,
# # and performs parallel distributed scatter/gather to reassemble local shards!
# loaded_state_dict = dist_checkpointing.load(
#     sharded_state_dict=new_model.sharded_state_dict(),
#     checkpoint_dir="/checkpoints/step_100000"
# )
# new_model.load_state_dict(loaded_state_dict)


# ── 2. Standalone Reference: Resharding Engine from Scratch ──
@dataclass
class LocalShardMetadata:
    param_name: str
    global_shape: tuple
    shard_slice: tuple  # (dim0_start, dim0_end, dim1_start, dim1_end)


def reshard_tensor_2d(
    saved_shards: list[tuple[LocalShardMetadata, torch.Tensor]],
    target_metadata: LocalShardMetadata,
) -> torch.Tensor:
    """
    Reconstructs a target shard for a different parallelism topology
    by sampling and stitching intersecting regions from saved shards.
    """
    target_tensor = torch.zeros(
        target_metadata.shard_slice[1] - target_metadata.shard_slice[0],
        target_metadata.shard_slice[3] - target_metadata.shard_slice[2],
        dtype=saved_shards[0][1].dtype
    )
    
    t_r0, t_r1, t_c0, t_c1 = target_metadata.shard_slice

    for meta, data in saved_shards:
        s_r0, s_r1, s_c0, s_c1 = meta.shard_slice

        # Compute 2D intersection of source shard and target shard
        inter_r0 = max(t_r0, s_r0)
        inter_r1 = min(t_r1, s_r1)
        inter_c0 = max(t_c0, s_c0)
        inter_c1 = min(t_c1, s_c1)

        if inter_r1 > inter_r0 and inter_c1 > inter_c0:
            # Source slice
            src_slice = data[
                (inter_r0 - s_r0):(inter_r1 - s_r0),
                (inter_c0 - s_c0):(inter_c1 - s_c0)
            ]
            # Target destination
            target_tensor[
                (inter_r0 - t_r0):(inter_r1 - t_r0),
                (inter_c0 - t_c0):(inter_c1 - t_c0)
            ] = src_slice

    return target_tensor
```

---

## 2.3. The Complete Megatron Architecture Master Map

Here is how every single concept we studied across all 7 books connects in a production system:

```
+========================================================================================+
|                        THE 5D DISTRIBUTED PARALLELISM MATRIX                           |
+========================================================================================+
| Global Cluster Topology: N_total = TP × SP × PP × DP × CP × EP                         |
|                                                                                        |
|                       ┌────────────────────────────────────────┐                       |
|                       │      GLOBAL DATA BATCH (DP Dimension)  │                       |
|                       └───────────────────┬────────────────────┘                       |
|                                           ▼                                            |
|                       ┌────────────────────────────────────────┐                       |
|                       │   CONTEXT PARALLELISM (CP Ring Mesh)   │                       |
|                       │ Sequence S partitioned into S/CP blocks│                       |
|                       └───────────────────┬────────────────────┘                       |
|                                           ▼                                            |
|                       ┌────────────────────────────────────────┐                       |
|                       │    PIPELINE PARALLELISM (PP Stages)    │                       |
|                       │ Stage 0 ──P2P──► Stage 1 ──P2P──► ...  │                       |
|                       └───────────────────┬────────────────────┘                       |
|                                           ▼                                            |
|       ┌────────────────────────────────────────────────────────────────────────┐       |
|       │                INTRA-NODE HIGH-SPEED NVLINK DOMAIN (TP / SP)           │       |
|       │                                                                        │       |
|       │  Sequence Parallel Layernorm/RMSNorm [S / (TP * CP), B, H]             │       |
|       │         │ (All-Gather along sequence dimension S)                      │       |
|       │         ▼                                                              │       |
|       │  Column-Parallel QKV Projection: W_qkv sharded into TP partitions      │       |
|       │         │ FlashAttention-3 Ring Attention (CP over NVLink / IB)        │       |
|       │         ▼                                                              │       |
|       │  Row-Parallel Projection: W_proj sharded into TP partitions            │       |
|       │         │ (Reduce-Scatter along sequence dimension S)                  │       |
|       │         ▼                                                              │       |
|       │  Sequence Parallel Dropout & Residual Addition                         │       |
|       └───────────────────────────────────┬────────────────────────────────────┘       |
|                                           ▼                                            |
|                       ┌────────────────────────────────────────┐                       |
|                       │   EXPERT PARALLELISM (EP All-to-All)   │                       |
|                       │ MoE Tokens routed to expert-owning GPUs│                       |
|                       └───────────────────┬────────────────────┘                       |
|                                           ▼                                            |
|                       ┌────────────────────────────────────────┐                       |
|                       │     DISTRIBUTED OPTIMIZER (ZeRO-2 DP)  │                       |
|                       │  Optimizer states sharded over DP group│                       |
|                       └────────────────────────────────────────┘                       |
+========================================================================================+
```

```
+========================================================================================+
|                              MEGATRON CORE MASTER ARCHITECTURE                         |
+========================================================================================+
|                                                                                        |
|  1. MODEL ARCHITECTURE: Transformer Config & Spec ([Production Engine Architecture](/production-engine/))                             |
|     • Modular specs for Attention, MLP, LayerNorm, and RoPE                            |
|     • FP8 E4M3 Forward / E5M2 Backward with Delayed Scaling ([Production Engine Architecture](/production-engine/))                   |
|                                                                                        |
|  2. DATA PARALLEL & MEMORY OPTIMIZATION:                                               |
|     • Megatron Distributed Optimizer (ZeRO-1 & ZeRO-2) ([Distributed Optimizer & ZeRO-2](/distributed-optimizer/))                        |
|     • Gradients Reduce-Scattered over DP Group during Backprop                         |
|     • Optimizer States sharded: 16 bytes/param reduced to (2 + 14/D)                   |
|                                                                                        |
|  3. INTRA-NODE HIGH BANDWIDTH (NVLink / NVSwitch):                                     |
|     • Tensor Parallelism (TP): Column-Parallel -> Row-Parallel GEMMs ([1D Tensor Parallelism](/tensor-parallelism/))          |
|     • Sequence Parallelism (SP): S/N sharding over LayerNorm & Dropout ([Sequence Parallelism](/sequence-parallelism/))        |
|     • Comm-Compute Overlap: Pipelined GEMM micro-tiles with All-Reduce ([Production Engine Architecture](/production-engine/))       |
|     • Selective Activation Recomputation: Checkpoint GEMMs, recompute Softmax ([Sequence Parallelism](/sequence-parallelism/)) |
|                                                                                        |
|  4. INTER-NODE NETWORK (InfiniBand / RoCE):                                            |
|     • Pipeline Parallelism (PP): 1F1B & Interleaved 1F1B Schedules ([Pipeline Parallelism & DualPipe](/pipeline-parallelism/))            |
|     • Non-blocking P2P communication (isend / irecv) between stages ([Pipeline Parallelism & DualPipe](/pipeline-parallelism/))           |
|                                                                                        |
|  5. LONG CONTEXT & SPARSITY SCALING:                                                   |
|     • Context Parallelism (CP): Ring Attention for 1M+ token sequences ([Context Parallelism & MoE](/context-parallelism/))        |
|     • Mixture of Experts (MoE): All-to-All token dispatching and routing ([Context Parallelism & MoE](/context-parallelism/))      |
|                                                                                        |
+========================================================================================+
```

---

