# Tensor Parallelism Implementation & Checklist
> **Complete Standalone PyTorch Implementation, Autograd Operators f/g, and Production Checklist**

---

## 3.1. Complete Standalone PyTorch Implementation: Megatron TP Layers from Scratch

Here is the complete, self-contained implementation of Megatron's Tensor Parallel building blocks using pure PyTorch. You can study, copy, and run this code directly:

```python
"""
Megatron 1D Tensor Parallelism: Complete PyTorch Implementation.
Includes:
- Autograd Conjugate Operators f and g
- ColumnParallelLinear
- RowParallelLinear
- Parallel Causal Self-Attention
- Parallel MLP
- Full MegatronTransformerBlock
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# ── 1. Conjugate Autograd Operators f and g ──

class _CopyToModelParallelRegion(torch.autograd.Function):
    """
    Operator f:
    Forward: Pass input unmodified (Identity).
    Backward: All-Reduce SUM across TP ranks (aggregates partitioned gradients).
    """
    @staticmethod
    def forward(ctx, input_, group=None):
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_initialized() and dist.get_world_size(ctx.group) > 1:
            dist.all_reduce(grad_output, group=ctx.group, op=dist.ReduceOp.SUM)
        return grad_output, None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """
    Operator g:
    Forward: All-Reduce SUM across TP ranks (combines partial row-parallel outputs).
    Backward: Pass upstream gradient unmodified (Identity).
    """
    @staticmethod
    def forward(ctx, input_, group=None):
        if dist.is_initialized() and dist.get_world_size(group) > 1:
            dist.all_reduce(input_, group=group, op=dist.ReduceOp.SUM)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def copy_to_model_parallel_region(input_, group=None):
    return _CopyToModelParallelRegion.apply(input_, group)

def reduce_from_model_parallel_region(input_, group=None):
    return _ReduceFromModelParallelRegion.apply(input_, group)


# ── 2. ColumnParallelLinear ──

class ColumnParallelLinear(nn.Module):
    """
    Splits weight along columns (out_features dimension):
    W = [W_1, W_2, ..., W_N], where each W_i has shape [in_features, out_features / N].
    No forward communication. Backward applies All-Reduce SUM via operator f.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, tp_group=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_group = tp_group
        
        self.tp_world_size = dist.get_world_size(tp_group) if dist.is_initialized() else 1
        self.tp_rank = dist.get_rank(tp_group) if dist.is_initialized() else 0
        
        assert out_features % self.tp_world_size == 0, "out_features must be divisible by TP size"
        self.out_features_per_partition = out_features // self.tp_world_size
        
        # Local weight shard: [out_features_per_partition, in_features]
        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Megatron initialization scales w.r.t unpartitioned dimensions
        std = math.sqrt(2.0 / self.in_features)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Operator f: identity forward, all-reduce backward
        input_parallel = copy_to_model_parallel_region(x, self.tp_group)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        return output_parallel


# ── 3. RowParallelLinear ──

class RowParallelLinear(nn.Module):
    """
    Splits weight along rows (in_features dimension):
    W = [W_1^T, W_2^T, ..., W_N^T]^T, where each W_i has shape [in_features / N, out_features].
    Forward applies All-Reduce SUM via operator g.
    Crucial: bias is added AFTER All-Reduce to prevent N-times duplication!
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, init_scale: float = 1.0, tp_group=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_scale = init_scale
        self.tp_group = tp_group
        
        self.tp_world_size = dist.get_world_size(tp_group) if dist.is_initialized() else 1
        self.tp_rank = dist.get_rank(tp_group) if dist.is_initialized() else 0
        
        assert in_features % self.tp_world_size == 0, "in_features must be divisible by TP size"
        self.in_features_per_partition = in_features // self.tp_world_size
        
        # Local weight shard: [out_features, in_features_per_partition]
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_partition))
        
        if bias:
            # Bias is replicated across all ranks
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Scaled w.r.t global full fan-in and residual depth scaling factor:
        # std = sqrt(2 / d_model) * init_scale (where init_scale = 1 / sqrt(2 * num_layers))
        std = math.sqrt(2.0 / self.in_features) * self.init_scale
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Matrix multiply local slice WITHOUT bias
        output_parallel = F.linear(x, self.weight, None)
        # Operator g: All-Reduce partial sums
        output_ = reduce_from_model_parallel_region(output_parallel, self.tp_group)
        # Add bias ONCE to the final gathered sum
        if self.bias is not None:
            output_ = output_ + self.bias
        return output_


# ── 4. Megatron Parallel MLP ──

class ParallelMLP(nn.Module):
    """
    ColumnParallelLinear (H -> 4H) -> GELU -> RowParallelLinear (4H -> H).
    Only 1 All-Reduce in forward pass!
    """
    def __init__(self, hidden_size: int, ffn_hidden_size: int, init_scale: float = 1.0, tp_group=None):
        super().__init__()
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, ffn_hidden_size, bias=True, tp_group=tp_group)
        self.activation = nn.GELU()
        self.dense_4h_to_h = RowParallelLinear(ffn_hidden_size, hidden_size, bias=True, init_scale=init_scale, tp_group=tp_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H]
        h = self.dense_h_to_4h(x)    # [B, S, 4H/N] (no comm)
        h = self.activation(h)        # [B, S, 4H/N] (local)
        out = self.dense_4h_to_h(h)   # [B, S, H] (1 All-Reduce)
        return out


# ── 5. Megatron Parallel Self-Attention ──

class ParallelSelfAttention(nn.Module):
    """
    Multi-Head Attention with partitioned heads.
    QKV projection is ColumnParallel (3H/N).
    Output projection is RowParallel (H/N -> H) with residual scaling init_scale.
    Only 1 All-Reduce in forward pass!
    """
    def __init__(self, hidden_size: int, num_heads: int, init_scale: float = 1.0, attention_dropout_prob: float = 0.0, tp_group=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.tp_group = tp_group
        
        tp_world_size = dist.get_world_size(tp_group) if dist.is_initialized() else 1
        assert num_heads % tp_world_size == 0, f"num_heads ({num_heads}) must be divisible by TP size ({tp_world_size})"
        self.num_local_heads = num_heads // tp_world_size
        self.head_dim = hidden_size // num_heads

        # ColumnParallel project to Q, K, V simultaneously: 3 * (num_local_heads * head_dim)
        self.query_key_value = ColumnParallelLinear(
            hidden_size, 3 * hidden_size, bias=True, tp_group=tp_group
        )
        # Optional attention dropout
        self.attention_dropout = nn.Dropout(attention_dropout_prob) if attention_dropout_prob > 0.0 else None

        # RowParallel project back to hidden_size with 1/sqrt(2L) residual scaling
        self.dense = RowParallelLinear(
            hidden_size, hidden_size, bias=True, init_scale=init_scale, tp_group=tp_group
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        # [B, S, 3 * num_local_heads * head_dim]
        qkv = self.query_key_value(x)
        
        # Real Megatron Core QKV Layout:
        # Reshape to separate the 3 projections and unbind along dim=2:
        qkv = qkv.reshape(B, S, 3, self.num_local_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each is [B, S, num_local_heads, head_dim]
        
        # Transpose to multi-head batch format: [B, num_local_heads, S, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention on local heads (zero comm!)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Robust Causal Mask: use torch.finfo(dtype).min to guarantee no NaN/underflow across FP16/BF16/FP32
        mask_value = torch.finfo(scores.dtype).min
        causal_mask = torch.triu(
            torch.full((S, S), mask_value, dtype=scores.dtype, device=x.device), diagonal=1
        )
        scores = scores + causal_mask
        probs = F.softmax(scores, dim=-1)
        if self.attention_dropout is not None:
            probs = self.attention_dropout(probs)
        
        # Context: [B, num_local_heads, S, head_dim] -> [B, S, num_local_heads * head_dim]
        context = torch.matmul(probs, v).transpose(1, 2).reshape(B, S, -1)
        
        # RowParallel output projection (1 All-Reduce across TP ranks)
        output = self.dense(context)
        return output


# ── 6. Full Megatron Transformer Block ──

class MegatronTransformerBlock(nn.Module):
    """
    Standard Pre-LN Transformer Block with Megatron 1D TP.
    Total communications: exactly 2 All-Reduces per layer!
    Residual scaling 1 / sqrt(2 * num_layers) applied to both residual projections.
    """
    def __init__(self, hidden_size: int, num_heads: int, ffn_hidden_size: int, num_layers: int = 1, tp_group=None):
        super().__init__()
        # Megatron-LM scaling rule: variance of residual stream preserved by scaling output projections
        init_scale = 1.0 / math.sqrt(2.0 * num_layers)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attention = ParallelSelfAttention(hidden_size, num_heads, init_scale=init_scale, tp_group=tp_group)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = ParallelMLP(hidden_size, ffn_hidden_size, init_scale=init_scale, tp_group=tp_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention with residual connection
        norm_x = self.input_layernorm(x)
        attn_out = self.self_attention(norm_x)
        x = x + attn_out

        # Pre-LN MLP with residual connection
        norm_x = self.post_attention_layernorm(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out
        return x
```

### 3.1.1 Deep Line-by-Line Pedagogical Breakdown of the TP Architecture:

#### 3.1.1.1 Why `F.linear` Shape is `[out_features, in_features]`
In PyTorch, the linear layer function `torch.nn.functional.linear(input, weight, bias)` implements:
$$y = x W^T + b$$
Because PyTorch transposes the weight matrix during computation, the memory layout of `weight` is stored as `[out_features, in_features]`:
- In **`ColumnParallelLinear`**: We partition `out_features`. Therefore, local weight has shape:
  $$\left[\frac{\text{out\_features}}{N}, \text{in\_features}\right]$$
- In **`RowParallelLinear`**: We partition `in_features`. Therefore, local weight has shape:
  $$\left[\text{out\_features}, \frac{\text{in\_features}}{N}\right]$$

---

#### 3.1.1.2 The Autograd Conjugate Pairing in Code
Look at where the autograd functions are placed:
- In **`ColumnParallelLinear.forward`**:
  ```python
  input_parallel = copy_to_model_parallel_region(x, self.tp_group) # Operator f
  output_parallel = F.linear(input_parallel, self.weight, self.bias)
  ```
  `copy_to_model_parallel_region` is an **identity** in forward. But it plants a hook in PyTorch's backward graph! When PyTorch runs backpropagation, it automatically triggers `dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)`. You never have to manually write backward communication logic!

- In **`RowParallelLinear.forward`**:
  ```python
  output_parallel = F.linear(x, self.weight, None) # NO bias!
  output_ = reduce_from_model_parallel_region(output_parallel, self.tp_group) # Operator g
  if self.bias is not None:
      output_ = output_ + self.bias
  ```
  `reduce_from_model_parallel_region` triggers an immediate **forward All-Reduce** to sum the partial dot-products across GPUs. During backpropagation, it returns the incoming gradient without modification.

---

#### 3.1.1.3 Why the Bias Timing is a Life-or-Death Detail
Notice that in `RowParallelLinear`, we explicitly pass `None` as the bias to `F.linear`, and only add `self.bias` **after** the All-Reduce:
```python
# CRITICAL:
output_parallel = F.linear(x, self.weight, None)  # 1. Multiply without bias
output_ = reduce_from_model_parallel_region(...)  # 2. All-Reduce partial sums
if self.bias is not None:
    output_ = output_ + self.bias                 # 3. Add bias ONCE to the true sum
```

What would happen if you added the bias during `F.linear(x, self.weight, self.bias)` before the All-Reduce?
Each of the $N$ GPUs would add its local bias $b$. Then, the All-Reduce would sum all $N$ outputs:
$$\text{Summed Output} = \left(\sum_{i=1}^N x_i W_i\right) + \mathbf{N \times b}$$
The bias would be added **$N$ times!** For $N = 8$, the bias term is multiplied by 8, instantly blowing up activations, destabilizing LayerNorm, and causing gradient explosion (NaN loss) within 10 iterations!

---

#### 3.1.1.4 The Megatron Attention Factoring Efficiency
Notice how `ParallelSelfAttention` fuses the Q, K, and V projections:
```python
self.query_key_value = ColumnParallelLinear(
    hidden_size, 3 * hidden_size, bias=True, tp_group=tp_group
)
```
- In naive attention, you would call `linear_q(x)`, `linear_k(x)`, and `linear_v(x)` separately.
- In Megatron, a single Column-Parallel GEMM produces the concatenated $[Q, K, V]$ tensor of shape $[B, S, 3 \times h_{local} \times d_{head}]$.
- Because all heads are partitioned evenly across ranks ($h_{local} = h / N$), each GPU computes attention for its assigned heads using standard scaled dot-product attention in local HBM.
- **Zero communication occurs during the entire attention score calculation and softmax!** Communication only occurs once, at the very end of the output projection `RowParallelLinear`.

---

## 3.2. Common Bugs & Gotchas in Tensor Parallelism

| Bug / Pitfall | Physical Symptom | Underlying Root Cause | Battle-Tested Fix |
|---|---|---|---|
| **$N \times b$ Bias Failure** | Loss explodes to NaN in $<10$ steps | Passing `bias` directly to `F.linear` in `RowParallelLinear` before the All-Reduce | Set `bias=None` in `RowParallelLinear`'s GEMM; add `bias` explicitly **after** `dist.all_reduce()` |
| **GQA Head Imbalance** | `AssertionError: num_heads % TP != 0` | Attempting to shard 8 KV heads across $\text{TP}=16$ | In Grouped-Query Attention, $h_{\text{KV}}$ must be an integer multiple of $N$; use $\text{TP} \le h_{\text{KV}}$ |
| **Vocab Parallel Striding Error** | `RuntimeError: input and target shapes do not match` | Target class indices not offset to local partition $[0, V/N)$ | Mask unowned targets with `-100` and subtract `vocab_start_index` before cross-entropy |
| **Duplicate RNG Seeds** | All TP ranks sample identical dropout masks | Using the same random seed across all TP ranks | Use Megatron's `CudaRNGStatesTracker` with distinct per-rank seeds for dropout |
| **Inconsistent Weight Init** | Output differs across TP ranks on identical inputs | Column/Row parallel weights initialized with different random states | Seed identically for replicated weights, use rank-dependent offsets for partitioned weights |

---

## 3.3. Runnable Checklist & Verification

To verify the complete 1D Tensor Parallelism module locally on 2 CPU workers:

```bash
# Verify ColumnParallelLinear, RowParallelLinear, and VocabParallelEmbedding
torchrun --nproc_per_node=2 demo_megatron.py
```

**Key Execution Checks**:
1. Confirm that `tp_size=2` partitions `hidden_size=1024` into two local slices of `512`.
2. Confirm that forward loss matches the unpartitioned reference model within floating-point tolerance ($\Delta < 1 \times 10^{-5}$).
3. Verify that backward gradients across both ranks are synchronized via the backward All-Reduce of operator $f$.

In **[Sequence Parallelism](/sequence-parallelism/)**, we will study **Sequence Parallelism (SP)** from Megatron-LM v3: how to eliminate activation memory redundancy in LayerNorm and Dropout with zero extra communication!

