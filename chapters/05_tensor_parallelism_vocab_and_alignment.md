# Vocabulary Parallelism & Tensor Core Alignment
> **VocabParallelEmbedding, Parallel Cross-Entropy, GQA/MQA Sharding, and Numerical Traces**

---

## 2.1. Vocabulary Parallelism & Parallel Cross Entropy

When training language models with massive vocabularies (`V = 32,000` to `256,000`), the token embedding table and final language modeling head become major memory bottlenecks:

> `Params = V * H`

For `V = 128,000` and `H = 8,192`, the embedding matrix alone consumes **2 GB in FP16**. More critically, computing logits produces a massive activation tensor:

> `Logits Memory = B * T * V * 2 bytes`

For `B = 8, T = 4,096, V = 128,000`, the logits tensor is **`8.4 GB` per microbatch**, often triggering Out-Of-Memory (OOM) errors!

### 2.1.1 `VocabParallelEmbedding`
We partition the vocabulary dimension across the N ranks:

> `Vocab Partition Size = ceil(V / N)`

Rank i holds token IDs in range:

> `[start_idx_i, end_idx_i) = [i * (V / N), (i+1) * (V / N))`

```
Token Input Tensor: [ token_id = 45 ]
Rank 0 holds vocab [0 .. 31]:   Lookup gives ZERO vector
Rank 1 holds vocab [32 .. 63]:  Lookup gives Embedding Vector E_45
Rank 2 holds vocab [64 .. 95]:  Lookup gives ZERO vector
Rank 3 holds vocab [96 .. 127]: Lookup gives ZERO vector

All-Reduce (SUM) across all ranks -> All ranks obtain E_45!
```

```python
"""
VocabParallelEmbedding Implementation.
Shards embedding table along vocab dimension across tensor parallel ranks.
"""
import torch
import torch.nn as nn
import torch.distributed as dist

class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, tp_group=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_group = tp_group
        
        self.tp_world_size = dist.get_world_size(tp_group) if dist.is_initialized() else 1
        self.tp_rank = dist.get_rank(tp_group) if dist.is_initialized() else 0
        
        # Partition vocab size
        self.vocab_start_index, self.vocab_end_index, self.padded_vocab_size = self._get_vocab_range()
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        
        # Local embedding weight slice (aligned to Tensor Core boundaries)
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def _get_vocab_range(self):
        # NVIDIA Tensor Core alignment: GEMM outer dimensions must be multiples
        # of 64 or 128 elements to maximize systolic array throughput and avoid unaligned P2P/All-Reduce.
        alignment = self.tp_world_size * 64
        padded_vocab_size = int(math.ceil(self.num_embeddings / alignment)) * alignment
        per_partition = padded_vocab_size // self.tp_world_size
        start = self.tp_rank * per_partition
        end = start + per_partition
        return start, end, padded_vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Build mask for tokens belonging to this rank's partition
        input_mask = (input_ids < self.vocab_start_index) | (input_ids >= self.vocab_end_index)
        # Shift input to local index range [0, num_embeddings_per_partition)
        local_ids = input_ids - self.vocab_start_index
        local_ids[input_mask] = 0  # Dummy index to prevent out-of-bounds error
        
        # Local lookup
        output_parallel = nn.functional.embedding(local_ids, self.weight)
        # Mask out embeddings for tokens that do NOT belong to this rank
        output_parallel[input_mask] = 0.0
        
        # All-Reduce (SUM) across TP ranks to reassemble full embeddings
        if self.tp_world_size > 1 and dist.is_initialized():
            dist.all_reduce(output_parallel, group=self.tp_group, op=dist.ReduceOp.SUM)
            
        return output_parallel
```

---

### 2.1.2 `ParallelCrossEntropyLoss`: Softmax Without Gathering Logits

In standard PyTorch, cross-entropy is:

> `Loss = -log ((e^z_target / sum(j=1)^V e^z_j)) = -z_target + log (sum(j=1)^V e^z_j)`

In Megatron, each rank i only computes logits for its local slice of vocab V_i:

> `z_local in shape [B * T * (V / N)]`

Megatron computes the loss in parallel using **three lightweight All-Reduces on scalars**:

```
Step 1: Local Max
  Each rank finds: m_i = max(z_local)
  All-Reduce (MAX) gives global max: M = max_i(m_i)

Step 2: Local Exp Sum
  Each rank computes: s_i = sum(exp(z_local - M))
  All-Reduce (SUM) gives global denominator: S = sum_i(s_i)

Step 3: Target Logit Extraction
  Rank holding target token extracts z_target (others have 0).
  All-Reduce (SUM) shares z_target across ranks.

Final Loss:
  Loss = - (z_target - M) + log(S)
```

**Memory Saved**: The full `(B * T * V)` tensor is **NEVER materialized in GPU memory**!

```python
"""
ParallelCrossEntropyLoss Implementation.
Computes cross-entropy loss over partitioned logits without gathering the full [B, S, V] tensor.
"""
class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, tp_group=None):
        # vocab_parallel_logits: [B, S, V/N]
        # target: [B, S]
        tp_world_size = dist.get_world_size(tp_group) if dist.is_initialized() else 1
        tp_rank = dist.get_rank(tp_group) if dist.is_initialized() else 0

        # Step 1: Global Maximum Logit for numerical stability
        local_max = torch.max(vocab_parallel_logits, dim=-1, keepdim=True)[0]
        global_max = local_max.clone()
        if tp_world_size > 1 and dist.is_initialized():
            dist.all_reduce(global_max, group=tp_group, op=dist.ReduceOp.MAX)

        # Step 2: Sum of Exponentials
        shifted_logits = vocab_parallel_logits - global_max
        local_sum_exp = torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True)
        global_sum_exp = local_sum_exp.clone()
        if tp_world_size > 1 and dist.is_initialized():
            dist.all_reduce(global_sum_exp, group=tp_group, op=dist.ReduceOp.SUM)

        # Step 3: Extract target logit
        per_partition = vocab_parallel_logits.size(-1)
        vocab_start = tp_rank * per_partition
        vocab_end = vocab_start + per_partition
        
        target_mask = (target >= vocab_start) & (target < vocab_end)
        local_target = target.clone() - vocab_start
        local_target[~target_mask] = 0

        # Gather target logit from local partition
        target_logit = torch.gather(vocab_parallel_logits, dim=-1, index=local_target.unsqueeze(-1)).squeeze(-1)
        target_logit[~target_mask] = 0.0

        if tp_world_size > 1 and dist.is_initialized():
            dist.all_reduce(target_logit, group=tp_group, op=dist.ReduceOp.SUM)

        # Loss = log(sum(exp(z - max))) + max - target_logit
        loss = torch.log(global_sum_exp.squeeze(-1)) + global_max.squeeze(-1) - target_logit

        # Save tensors for backward pass
        ctx.save_for_backward(shifted_logits, global_sum_exp, target_mask, local_target)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        shifted_logits, global_sum_exp, target_mask, local_target = ctx.saved_tensors
        # Softmax probabilities: exp(z - max) / sum(exp(z - max))
        softmax = torch.exp(shifted_logits) / global_sum_exp
        
        # dL / dz = softmax - 1 (for target token)
        grad_input = softmax
        if target_mask.any():
            grad_input.scatter_add_(-1, local_target.unsqueeze(-1), -torch.ones_like(grad_input))
            
        grad_input.mul_(grad_output.unsqueeze(-1))
        return grad_input, None, None

class ParallelCrossEntropyLoss(nn.Module):
    def __init__(self, tp_group=None):
        super().__init__()
        self.tp_group = tp_group

    def forward(self, vocab_parallel_logits, target):
        return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, self.tp_group).mean()
```

---

## 2.2. Numerical Subtleties & Bias Handling in `RowParallelLinear`

A subtle bug in tensor-parallel implementations occurs in bias handling:

> `Y = X W + b`

In RowParallelLinear:

> `Y = (sum(i=1)^N X_i W_i) + b`

If each rank computes `Z_i = X_i W_i + b`, and then performs `All-Reduce(SUM)`, the resulting tensor will be:

> `sum(i=1)^N (X_i W_i + b) = (sum(i=1)^N X_i W_i) + N * b`

The bias is added N times!

### 2.2.1 The Two Correct Fixes:
1. **Divide bias by N**: Add `b/N` before All-Reduce (introduces floating-point roundoff error).
2. **Add bias AFTER All-Reduce (Megatron's method)**:
   ```python
   # Compute GEMM without bias
   output_parallel = F.linear(input_parallel, self.weight, None)
   # All-reduce partial sums
   output = reduce_from_model_parallel_region(output_parallel)
   # Add bias once to the final sum
   if self.bias is not None:
       output = output + self.bias
   ```

---

## 2.3. Grouped Query Attention (GQA) & Multi-Query Attention (MQA) with TP

Modern LLMs (Llama 3, Mistral, Qwen) use **Grouped Query Attention (GQA)** instead of standard Multi-Head Attention (MHA).

In GQA, query heads and key/value heads have different counts:
- Query heads: h_Q (e.g., 32)
- Key/Value heads: h_KV (e.g., 8)
- **Group size**: `G = h_Q / h_KV` (e.g., 4)

Each KV head is shared across G query heads.

### 2.3.1 Why GQA Changes the Tensor Parallel Constraint:

In standard MHA, any N that evenly divides h works:

> `h % N = 0`

In GQA, we must also ensure KV heads can be partitioned evenly:

> `h_KV % N = 0`

If `h_KV = 8` and `N = 4`, each rank holds `h_Q / N = 8` query heads and `h_KV / N = 2` KV heads.

### 2.3.2 The Constraint Matrix:

| Config | h_Q | h_KV | Max TP | Per-Rank Q Heads | Per-Rank KV Heads |
|---|---|---|---|---|---|
| **MHA** | 32 | 32 | 8 (if node has 8) | 4 | 4 |
| **GQA (G=4)** | 32 | 8 | 8 | 4 | 1 |
| **MQA** | 32 | 1 | **1** (Cannot TP!) | 32 | 1 |

> [!IMPORTANT]
> **Multi-Query Attention (MQA)** has exactly **1 KV head**. You **cannot** apply Tensor Parallelism to MQA attention without special all-gather tricks, because there's no KV head to shard! This is why modern production models use GQA (minimum `h_KV = N`) rather than MQA.

---

## 2.4. Weight Initialization Scaling for Tensor Parallelism

A subtle but critical detail: **weight initialization must be adjusted for TP.**

In a standard single-GPU model, residual projections (attention output projection and MLP FC2) are typically initialized with reduced variance to prevent residual activation blow-up:

> `Std(W_out) = (0.02 / (sqrt(2 * L)))`

Where L is the number of transformer layers (the 2L accounts for 2 residual connections per block).

In Megatron with Tensor Parallelism, `RowParallelLinear` projects from `(4H / N) -> H` instead of `4H -> H`.
The **fan-in** is `4H/N` rather than 4H, which changes the He/Xavier initialization normalization.

Megatron handles this by scaling with respect to the **full un-partitioned fan-in**:
```python
# In ColumnParallelLinear / RowParallelLinear __init__:
# WRONG (uses partitioned fan-in — gives wrong variance):
std = math.sqrt(2.0 / in_features_per_partition)

# CORRECT (Megatron always scales w.r.t. the FULL global model fan-in):
std = math.sqrt(2.0 / (in_features_global * num_layers_factor))
```

Not doing this correctly causes the activation variance to diverge by a factor of `sqrt(N)` across GPUs!

---

## 2.5. Worked Numerical Example: Exact Tensor Shapes at Every Step (TP=2)

Let's trace the complete forward pass of one Transformer block with exact tensor shapes:

**Model Config**: `B=2, S=4, H=8, h=4, d_head=2, TP=2`

```
INPUT: X [B=2, S=4, H=8]  <-- Identical on both Rank 0 and Rank 1
         │
         ├── Operator f (Identity)
         │
         ▼ RANK 0                               ▼ RANK 1
   W_QKV [8, 12] (heads 0,1)            W_QKV [8, 12] (heads 2,3)
   Output QKV [2, 4, 12]                Output QKV [2, 4, 12]
   Q0 [2, 4, 4], K0 [2, 4, 4]          Q1 [2, 4, 4], K1 [2, 4, 4]
   V0 [2, 4, 4]                         V1 [2, 4, 4]
         │                                      │
   AttnScores0 = Q0 @ K0.T              AttnScores1 = Q1 @ K1.T
   [2, h_local=2, S=4, S=4]             [2, h_local=2, S=4, S=4]
         │                                      │
   AttnOut0 = Softmax @ V0              AttnOut1 = Softmax @ V1
   [2, h_local=2, 4, 2]                 [2, h_local=2, 4, 2]
         │                                      │
   AttnOut0.reshape [2, 4, 4]           AttnOut1.reshape [2, 4, 4]
         │                                      │
   W_proj0 [4, 8]                       W_proj1 [4, 8]
   Z_proj0 = AttnOut0 @ W_proj0         Z_proj1 = AttnOut1 @ W_proj1
   [2, 4, 8]                            [2, 4, 8]
         │                                      │
         └──────── Operator g: All-Reduce (SUM) ┘
                             │
              Y_attn = Z_proj0 + Z_proj1 [2, 4, 8]  <-- Full H=8 on both ranks!
```

This shows concretely how the RowParallelLinear output projections `Z_proj0 [2,4,8]` and `Z_proj1 [2,4,8]` are **summed** via All-Reduce to yield the final output. Neither rank holds a partial slice of H; both receive the full `H=8` dimensional vector.

---

## 2.6. Summary & Checklist

| Component | Parallelism Type | Local Dimension | Forward Communication | Backward Communication |
|---|---|---|---|---|
| **Token Embedding** | Vocab Parallel | `V/N * H` | All-Reduce (SUM) | Identity |
| **QKV Projection** | Column Parallel | `H * (3H/N)` | 0 | All-Reduce (SUM) |
| **Self-Attention** | Head Partitioned | `h/N heads` | 0 | 0 |
| **Attention Proj** | Row Parallel | `(H/N) * H` | All-Reduce (SUM) | Identity |
| **MLP FC1** | Column Parallel | `H * (4H/N)` | 0 | All-Reduce (SUM) |
| **MLP GELU** | Local Elementwise | `4H/N` | 0 | 0 |
| **MLP FC2** | Row Parallel | `(4H/N) * H` | All-Reduce (SUM) | Identity |
| **LM Head** | Column Parallel | `H * V/N` | 0 (or All-Gather) | All-Reduce (SUM) |

---

