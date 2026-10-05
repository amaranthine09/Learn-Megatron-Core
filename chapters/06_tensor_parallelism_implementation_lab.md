# Tensor Parallelism: API Reference & Common Bugs
> **Megatron-Core ColumnParallelLinear, RowParallelLinear, and VocabParallelEmbedding API**

---

## 3.1. Megatron-Core Tensor Parallel API

The production implementations live in `megatron.core.tensor_parallel.layers`. The key building blocks are:

### 3.1.1 ColumnParallelLinear

Splits the weight matrix along columns (output features). No forward communication; backward requires All-Reduce via operator `f`.

```python
from megatron.core.tensor_parallel.layers import ColumnParallelLinear

# in_features: full input hidden size H
# out_features: full output size (e.g. 4H for MLP up-projection)
# gather_output=False keeps outputs sharded (S/N) for chained RowParallel
col_linear = ColumnParallelLinear(
    input_size=hidden_size,
    output_size=ffn_hidden_size,
    bias=True,
    gather_output=False,        # Chain directly into RowParallelLinear
    init_method=init_method,
    skip_bias_add=False,
)

# Forward: input [B, S, H] → output [B, S, ffn_hidden_size/TP]
output, bias = col_linear(input_tensor)
```

**Shape invariant**: Each TP rank holds weight of shape `[out_features/TP, in_features]`.

---

### 3.1.2 RowParallelLinear

Splits the weight matrix along rows (input features). Forward requires All-Reduce via operator `g`. Bias is **always** added after the All-Reduce to avoid `N *` bias duplication.

```python
from megatron.core.tensor_parallel.layers import RowParallelLinear

row_linear = RowParallelLinear(
    input_size=ffn_hidden_size,
    output_size=hidden_size,
    bias=True,
    input_is_parallel=True,     # Input already sharded from ColumnParallel
    init_method=output_layer_init_method,
    skip_bias_add=False,
)

# Forward: input [B, S, ffn_hidden_size/TP] → output [B, S, H] (all-reduced)
output, bias = row_linear(input_tensor)
```

> [!IMPORTANT]
> **Bias Timing**: Megatron's `RowParallelLinear` internally passes `bias=None` to `F.linear`, fires `dist.all_reduce`, and **then** adds the bias. If the bias is added inside the GEMM before the All-Reduce, it gets summed N times across TP ranks, blowing up activations within ~10 training steps.

---

### 3.1.3 VocabParallelEmbedding

Shards the vocabulary embedding table across TP ranks. Each rank owns a contiguous vocabulary slice `[vocab_start, vocab_end)`.

```python
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding

vocab_embedding = VocabParallelEmbedding(
    num_embeddings=vocab_size,      # Full vocabulary size
    embedding_dim=hidden_size,
    init_method=init_method,
)

# Forward: token_ids [B, S] → embeddings [B, S, H]
# Tokens outside local shard are zeroed; an All-Reduce recovers full output.
embeddings = vocab_embedding(token_ids)
```

**Vocabulary offset handling**: Tokens in `[vocab_start, vocab_end)` are remapped to `[0, vocab_size/TP)`. Tokens outside are set to zero before the All-Reduce aggregation.

---

### 3.1.4 Fused QKV Projection (ParallelAttention Pattern)

Megatron fuses Q, K, and V into a single `ColumnParallelLinear` to reduce kernel launches:

```python
from megatron.core.tensor_parallel.layers import ColumnParallelLinear

# Single fused QKV projection: H → 3H (or [H, H_kv, H_kv] for GQA)
self.query_key_value = ColumnParallelLinear(
    input_size=hidden_size,
    output_size=3 * hidden_size,    # or (num_heads + 2 * num_kv_heads) * head_dim
    bias=True,
    gather_output=False,
    init_method=init_method,
)
```

Each TP rank computes attention for `num_heads / TP` local heads entirely in HBM — zero communication during QK^T and softmax. The only inter-GPU communication is the output `RowParallelLinear` All-Reduce.

---

### 3.1.5 CudaRNGStatesTracker for Dropout

Megatron uses a custom RNG tracker to ensure TP ranks sample **different** dropout masks (since each rank processes different head partitions):

```python
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

# In transformer forward pass:
with get_cuda_rng_tracker().fork():
    # Each TP rank uses an independent RNG state seeded at init
    x = F.dropout(x, p=dropout_prob, training=self.training)
```

> [!WARNING]
> If all TP ranks use the same RNG seed for dropout, they sample identical masks. Since each rank applies dropout to different head partitions, the masks should be statistically independent — using a shared seed introduces hidden correlations that reduce regularization effectiveness.

---

## 3.2. Common Bugs & Gotchas in Tensor Parallelism

| Bug / Pitfall | Physical Symptom | Underlying Root Cause | Battle-Tested Fix |
|---|---|---|---|
| **`N * b` Bias Failure** | Loss explodes to NaN in `<10` steps | Passing `bias` directly to `F.linear` in `RowParallelLinear` before the All-Reduce | Set `bias=None` in `RowParallelLinear`'s GEMM; add `bias` explicitly **after** `dist.all_reduce()` |
| **GQA Head Imbalance** | `AssertionError: num_heads % TP != 0` | Attempting to shard 8 KV heads across `TP=16` | In Grouped-Query Attention, h_KV must be an integer multiple of N; use `TP <= h_KV` |
| **Vocab Parallel Striding Error** | `RuntimeError: input and target shapes do not match` | Target class indices not offset to local partition `[0, V/N)` | Mask unowned targets with `-100` and subtract `vocab_start_index` before cross-entropy |
| **Duplicate RNG Seeds** | All TP ranks sample identical dropout masks | Using the same random seed across all TP ranks | Use Megatron's `CudaRNGStatesTracker` with distinct per-rank seeds for dropout |
| **Inconsistent Weight Init** | Output differs across TP ranks on identical inputs | Column/Row parallel weights initialized with different random states | Seed identically for replicated weights, use rank-dependent offsets for partitioned weights |

---

## 3.3. Summary & What's Next

In **[Sequence Parallelism](/sequence-parallelism/)**, we will study **Sequence Parallelism (SP)** from Megatron-LM v3: how to eliminate activation memory redundancy in LayerNorm and Dropout with zero extra communication!
