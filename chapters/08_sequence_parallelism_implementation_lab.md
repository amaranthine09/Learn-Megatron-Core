# Sequence Parallelism: API Reference & Memory Economics
> **Megatron-Core SP Autograd Mappings, Memory Analysis, and Common Bugs**

---

## 2.1. Sequence Parallel Autograd Mappings (megatron.core)

The Sequence Parallel operators live in `megatron.core.tensor_parallel.mappings`. They are the conjugate pair that replaces All-Reduce with Reduce-Scatter + All-Gather:

```python
# megatron/core/tensor_parallel/mappings.py

class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """
    Forward: Reduce-scatter along sequence dimension (dim 0 in [S, B, H] layout).
    Backward: All-gather along sequence dimension.
    Used at the OUTPUT of RowParallelLinear.
    """
    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)

class _AllGatherFromSequenceParallelRegion(torch.autograd.Function):
    """
    Forward: All-gather along sequence dimension (dim 0 in [S, B, H] layout).
    Backward: Reduce-scatter along sequence dimension.
    Used at the INPUT of ColumnParallelLinear.
    """
    @staticmethod
    def forward(ctx, input_, need_to_all_gather=True):
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce_scatter_along_first_dim(grad_output), None
```

### 2.1.1 The Conjugate Duality in Sequence Space

Notice the symmetry between forward and backward passes:

> `Forward: Reduce-Scatter <=> Backward: All-Gather`
> `Forward: All-Gather     <=> Backward: Reduce-Scatter`

Why does this mathematical conjugate relationship exist?
- When a forward operation **scatters** data to N GPUs, each GPU receives a `(1 / N)`-th slice. In the backward pass, each GPU computes a gradient for its local slice. To reconstruct the gradient with respect to the original unscattered input, the gradients must be **gathered** back together.
- When a forward operation **gathers** data from N GPUs to form a full tensor, every GPU receives a copy of the full tensor. In the backward pass, every GPU computes a gradient on the full tensor. To reconstruct the gradient with respect to each local input slice, the gradients must be **summed across ranks and scattered**.

---

## 2.2. How SP Integrates with TP Layers

In Megatron-Core v3+, the SP operators wrap the existing TP layers with zero extra communication cost:

```
                    [S/N, B, H]  ← Sequence-parallel shard (LayerNorm, Dropout)
                         │
         AllGather (forward) / ReduceScatter (backward)
                         │
                    [S, B, H]    ← Full sequence for ColumnParallelLinear
                         │
              ColumnParallelLinear (no comm forward)
                         │
                    [S, B, H/N]  ← TP-sharded output
                         │
              RowParallelLinear (All-Reduce → replaced by ReduceScatter)
                         │
         ReduceScatter (forward) / AllGather (backward)
                         │
                    [S/N, B, H]  ← Returns to sequence-parallel layout
```

The key Megatron-Core configuration flag that enables this:

```python
from megatron.core.transformer.transformer_config import TransformerConfig

config = TransformerConfig(
    tensor_model_parallel_size=8,
    sequence_parallel=True,      # Enables SP: replaces TP All-Reduce with RS+AG
    ...
)
```

> [!IMPORTANT]
> When `sequence_parallel=True`, `RowParallelLinear` internally switches from `dist.all_reduce` to `dist.reduce_scatter`, and `ColumnParallelLinear` prefixes inputs with `dist.all_gather`. No code changes are needed in the model definition — `TransformerConfig` controls the behavior.

---

## 2.3. Why `use_reentrant=False` in `torch.utils.checkpoint` is Mandatory

In Megatron with SP, activation checkpointing is configured via:

```python
from megatron.core.transformer.transformer_block import TransformerBlock

# TransformerBlock respects the config flag:
config = TransformerConfig(
    recompute_granularity='selective',   # or 'full'
    recompute_method='uniform',
    recompute_num_layers=1,
    ...
)
```

Internally, Megatron calls:
```python
torch.utils.checkpoint.checkpoint(
    forward_fn,
    *args,
    use_reentrant=False,    # MANDATORY for distributed autograd functions
)
```

In legacy PyTorch (< 2.0), activation checkpointing was **reentrant** — it re-invoked `torch.autograd.backward()` recursively inside a separate autograd engine instance. **The failure mode**: reentrant checkpointing does **NOT** work with custom autograd functions that perform collective communication (`dist.reduce_scatter`, `dist.all_gather`), causing deadlocks or silent gradient leaks.

Modern PyTorch (≥ 2.0) `use_reentrant=False` records the forward pass normally, stashes the inputs, and re-executes within the exact same autograd tape during backward, seamlessly supporting distributed operators and non-blocking CUDA streams.

---

## 2.4. Memory Economics Comparison

| Strategy | LayerNorm Activation Memory | Dropout Activation Memory | Communication per Block | Compute Overhead |
|---|---|---|---|---|
| **Pure TP (v1)** | Full `B * S * H` | Full `B * S * H` | 2 All-Reduces | 0% |
| **TP + Full Recomp** | Minimal | Minimal | 2 All-Reduces | +33% |
| **TP + Sequence Parallel (v3)** | `((B * S * H) / N)` | `((B * S * H) / N)` | 2 RS + 2 AG (Identical!) | 0% |
| **TP + SP + Selective Recomp** | **Minimum possible** | **Minimum possible** | **Identical!** | **`< 3%`** |

**Communication volume equivalence**:

> `2 * ((N-1) / N) * S (All-Reduce) = ((N-1) / N) * S (Reduce-Scatter) + ((N-1) / N) * S (All-Gather)`

Sequence Parallelism achieves a `(1 / N)` reduction in activation memory for all sequence-length-dependent operations (LayerNorm, Dropout) at zero additional communication cost.

---

## 2.5. Common Bugs & Gotchas in Sequence Parallelism

| Bug / Pitfall | Physical Symptom | Underlying Root Cause | Battle-Tested Fix |
|---|---|---|---|
| **Reentrant Autograd Deadlock** | Distributed job hangs during backward pass | Using `torch.utils.checkpoint` with legacy `use_reentrant=True` alongside collective functions | Always pass `use_reentrant=False` in PyTorch `>= 2.0` |
| **Indivisible Sequence Length** | `AssertionError: S % TP != 0` | Sequence length `S=4,097` not evenly divisible by TP world size `N=8` | Pad input sequence length to the nearest multiple of N before embedding |
| **All-Gather Buffer Overwrite** | Silent gradient corruption | Output buffer in `_AllGather` points to an in-place mutated activation | Ensure `tensor_list = [torch.empty_like(...) ...]` allocates fresh non-overlapping memory |
| **Double Sharding Trap** | Tensor shapes shrink to `S / N^2` | Applying Sequence Parallel Reduce-Scatter to an already sequence-parallel tensor | Strictly apply Reduce-Scatter only at the exit of `RowParallelLinear` |
| **LayerNorm Comm Leaks** | Extremely slow LayerNorm forward | Calling collective communication inside LayerNorm | LayerNorm operates purely on the last dimension H; ensure zero collectives are executed inside normalization |

---

## 2.6. Summary & What's Next

In **[Pipeline Parallelism & 1F1B Schedules](/pipeline-parallelism/)**, we expand beyond a single node: **Pipeline Parallelism (PP)**, the 1F1B schedule, and managing the pipeline bubble.
