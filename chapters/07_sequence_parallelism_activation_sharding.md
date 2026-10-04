# Sequence Parallelism & Dynamic Activation Management
> **Zero-Overhead Activation Sharding, Selective Recomputation, and Memory Economics**

> **Reference Paper**: *Reducing Activation Recomputation in Large Transformer Models* (Korthikanti et al., NVIDIA 2022, [arXiv:2205.05198](https://arxiv.org/abs/2205.05198))

---

## 1.1. Activation Memory Scaling Limits in Pure Tensor-Parallel Transformers

In pure Tensor Parallelism (TP), the weights and intermediate GEMM activations of the **Attention** and **MLP** blocks are partitioned across $N$ GPUs.

However, consider the components of a Transformer block outside of Attention and MLP:
- **LayerNorm 1 & LayerNorm 2**
- **Residual Dropout 1 & Residual Dropout 2**
- **Residual Additions**

```
            Megatron v1/v2 Architecture Bottleneck
            
           Input X: [B, S, H]  <--- REPLICATED on all N GPUs!
                 │
            [LayerNorm]        <--- REPLICATED computation & memory!
                 │
         [TP Attention]        <--- Sharded across N GPUs
                 │
          (All-Reduce)         <--- Re-assembles full [B, S, H]
                 │
             [Dropout]         <--- REPLICATED on all N GPUs!
                 │
                 +             <--- REPLICATED residual add!
                 │
            [LayerNorm]        <--- REPLICATED computation & memory!
                 │
             [TP MLP]          <--- Sharded across N GPUs
                 │
          (All-Reduce)         <--- Re-assembles full [B, S, H]
                 │
             [Dropout]         <--- REPLICATED on all N GPUs!
                 │
                 +             <--- REPLICATED residual add!
```

### 1.1.1 The Math of Replicated Activations: Where Does Memory Go?
For a Transformer model with hidden size $H$, sequence length $S$, batch size $B$, and $L$ layers:
- Each LayerNorm must store its normalized input for backpropagation: $2 \times (B \times S \times H)$ elements.
- Each Dropout requires an activation bitmask: $(B \times S \times H)$ bytes.
- Residual connections must store their inputs to compute gradient additions: $(B \times S \times H)$ elements.

Even with $\text{TP} = 8$, **none of this memory was sharded in Megatron v1/v2**. Every single GPU stored the exact same full $(B \times S \times H)$ activation tensors!

As sequence length $S$ grew from $1{,}024$ to $4{,}096$ and beyond, activation memory exploded to over **$75\%$ of total GPU VRAM**, forcing practitioners to use slow, brute-force activation recomputation (checkpointing) which wasted $>30\%$ of total GPU compute time.

---

## 1.2. Megatron Sequence Parallelism: The Decomposition Insight

Korthikanti et al. (2022) noticed an elegant algebraic equivalence in collective communications:

$$\mathbf{\text{All-Reduce}}(X) \equiv \mathbf{\text{All-Gather}}\Big(\mathbf{\text{Reduce-Scatter}}(X)\Big)$$

Recall from [Distributed Foundations & Interconnects](/foundations/) that Ring All-Reduce is physically executed in two successive phases:
1. **Scatter-Reduce**: Takes full tensors from all ranks, sums them, and leaves rank $i$ holding a reduced $\frac{1}{N}$-th shard.
2. **All-Gather**: Collects the reduced shards from all ranks and concatenates them back into a full tensor.

In pure Tensor Parallelism ([1D Tensor Parallelism](/tensor-parallelism/)), immediately after `RowParallelLinear`, we executed an All-Reduce to reconstruct the full $[B, S, H]$ tensor for LayerNorm.

**Megatron Sequence Parallelism asks a foundational question:**
> *Why reconstruct the full sequence $[B, S, H]$ before LayerNorm?*
> LayerNorm, Dropout, and Residual Additions are **completely independent across the sequence dimension $S$!*

$$\text{LayerNorm}(x_{1..S}) = \Big[\text{LayerNorm}(x_1), \text{LayerNorm}(x_2), \dots, \text{LayerNorm}(x_S)\Big]$$

Because LayerNorm computes the mean and variance across the **hidden dimension $H$ for each token individually**, **tokens at different sequence positions never communicate with each other!**
Whether a GPU holds $S$ tokens or $\frac{S}{N}$ tokens, the LayerNorm output for each token is mathematically identical.

---

## 1.3. The Sequence Parallelism Architecture

Instead of executing an All-Reduce at the end of Attention and an All-Reduce at the end of MLP, Sequence Parallelism **splits the All-Reduce across the layers**:

1. At the end of `RowParallelLinear`, we execute a **`Reduce-Scatter`** along the sequence dimension $S$.
   - Output shape on each GPU: $\left[B, \frac{S}{N}, H\right]$
2. **LayerNorm, Dropout, and Residual additions** run on sequence shards of size $\frac{S}{N}$!
   - Activation memory on each GPU drops by a factor of $N$!
3. Before entering the next `ColumnParallelLinear`, we execute an **`All-Gather`** along the sequence dimension to reconstruct the full $[B, S, H]$ sequence for the GEMM.

```
                   Sequence Parallel Transformer Block
                   
           Input Shard: [B, S/N, H]  <--- Sharded across GPUs!
                     │
                [LayerNorm]          <--- Memory is partitioned (S/N)!
                     │
                 All-Gather (Along Sequence Dim S)
                     │
           Full Sequence: [B, S, H]
                     │
               ColumnParallel QKV
                     │
              Local Multi-Head Attn
                     │
               RowParallel Proj
                     │
                Reduce-Scatter (Along Sequence Dim S)
                     │
           Output Shard: [B, S/N, H]
                     │
                 [Dropout]           <--- Memory is partitioned (S/N)!
                     │
                     + (Residual)    <--- Memory is partitioned (S/N)!
                     │
                [LayerNorm]          <--- Memory is partitioned (S/N)!
                     │
                 All-Gather (Along Sequence Dim S)
                     │
           Full Sequence: [B, S, H]
                     │
               ColumnParallel FC1
                     │
                  Local GELU
                     │
               RowParallel FC2
                     │
                Reduce-Scatter (Along Sequence Dim S)
                     │
           Output Shard: [B, S/N, H]
```

---

## 1.4. The Zero Communication Overhead Proof

Does Sequence Parallelism add communication latency? **Mathematically, no!**

Let $M = B \times S \times H$ be the total tensor volume in bytes, and $N$ be the Tensor Parallel size.

### 1.4.1 Communication in Pure Tensor Parallelism:
Each block performs 2 All-Reduces in the forward pass.
Recall from [Distributed Foundations & Interconnects](/foundations/) that the communication volume of a Ring All-Reduce is:
$$\text{Vol}_{\text{All-Reduce}} = 2 \left(\frac{N - 1}{N}\right) M$$

$$\text{Total Forward Comm}_{\text{TP}} = 2 \times \left[ 2 \left(\frac{N - 1}{N}\right) M \right] = \mathbf{4 \left(\frac{N - 1}{N}\right) M}$$

---

### 1.4.2 Communication in Tensor Parallelism + Sequence Parallelism:
In Sequence Parallelism, each block replaces:
- 1 All-Reduce $\longrightarrow$ 1 Reduce-Scatter + 1 All-Gather

Recall from [Distributed Foundations & Interconnects](/foundations/):
$$\text{Vol}_{\text{Reduce-Scatter}} = \left(\frac{N - 1}{N}\right) M$$
$$\text{Vol}_{\text{All-Gather}} = \left(\frac{N - 1}{N}\right) M$$

Summing both operations:
$$\text{Vol}_{\text{Reduce-Scatter}} + \text{Vol}_{\text{All-Gather}} = \left(\frac{N - 1}{N}\right) M + \left(\frac{N - 1}{N}\right) M = 2 \left(\frac{N - 1}{N}\right) M$$

$$\text{Total Forward Comm}_{\text{TP+SP}} = 2 \times \left[ 2 \left(\frac{N - 1}{N}\right) M \right] = \mathbf{4 \left(\frac{N - 1}{N}\right) M}$$

### 1.4.3 The Fundamental Equivalence Theorem:
$$\mathbf{\text{Comm Volume}}(\text{TP}) \equiv \mathbf{\text{Comm Volume}}(\text{TP} + \text{SP})$$

Sequence Parallelism introduces **EXACTLY ZERO additional communication bytes**, while slashing activation memory for all non-tensor-parallel layers by a factor of $N$!

---

## 1.5. The Comprehensive Activation Memory Breakdown

To quantify the breakthrough, let us examine the exact analytic equations derived by Korthikanti et al. for the activation memory per Transformer layer (in bytes, assuming 16-bit precision where each element is 2 bytes).

Let:
- $S$: Sequence length
- $B$: Batch size
- $H$: Hidden dimension
- $a$: Number of attention heads
- $N$: Tensor Parallel size

### 1.5.1 Standard Transformer (Single GPU, No Parallelism)
$$\text{Mem}_{\text{standard}} = S \cdot B \cdot H \cdot \left(34 + 5 \cdot \frac{a \cdot S}{H}\right) \text{ bytes}$$

Where:
- $34 \cdot S \cdot B \cdot H$ comes from GEMMs, LayerNorms, Dropouts, and Residuals.
- $5 \cdot \frac{a \cdot S^2 \cdot B}{H}$ comes from the quadratic attention matrices ($Q K^T$ scores, softmax probabilities, and attention dropout).

---

### 1.5.2 Tensor Parallelism Alone (Megatron-LM v1 & v2)
$$\text{Mem}_{\text{TP}} = S \cdot B \cdot H \cdot \left(\mathbf{10} + \frac{24}{N} + 5 \cdot \frac{a \cdot S}{N \cdot H}\right) \text{ bytes}$$

Notice that the **$10 \cdot S \cdot B \cdot H$ term is NOT divided by $N$!**
- The $10 \cdot S \cdot B \cdot H$ corresponds to the two LayerNorms ($2 \times 2 = 4$), two Dropouts ($2 \times 2 = 4$), and Residual additions ($2$).
- Even with $N = 8$, this constant non-sharded memory term dominates, choking the GPU at sequence lengths $S \ge 4{,}096$.

---

### 1.5.3 Tensor Parallelism + Sequence Parallelism (Megatron-LM v3)
$$\text{Mem}_{\text{TP+SP}} = S \cdot B \cdot H \cdot \left(\frac{\mathbf{34}}{N} + 5 \cdot \frac{a \cdot S}{N \cdot H}\right) \text{ bytes}$$

**Every single linear term is now divided by $N$!**
The non-sharded $10 \cdot S \cdot B \cdot H$ barrier is completely demolished.

---

### 1.5.4 Selective Activation Recomputation: Eliminating the Quadratic Term
Look at the remaining term:
$$5 \cdot \frac{a \cdot S^2 \cdot B}{N} \text{ bytes}$$
This term scales **quadratically with sequence length $S^2$**. At $S = 32{,}768$ or $128{,}000$, this quadratic term overwhelms all GPU memory, regardless of $N$.

### 1.5.5 The Selective Recomputation Insight:
Korthikanti et al. asked: *What creates this quadratic term?*
- The attention score matrix: $Q K^T$ ($B \cdot a \cdot S^2$)
- The softmax probabilities ($B \cdot a \cdot S^2$)
- The attention dropout mask ($B \cdot a \cdot S^2$)

These operations require **tiny compute FLOPs** (they are cheap elementwise exponential and multiplication operations), but consume **gigantic memory**!
Conversely, the QKV projections and MLP projections require **huge compute FLOPs ($95\%$ of all step FLOPs)**, but produce relatively small activations $\mathcal{O}(S \cdot H)$.

> **The Selective Recomputation Rule**:
> Store the inputs to the expensive GEMMs. Discard ONLY the cheap quadratic attention operations (Softmax, Attention Dropout), and recompute them on-the-fly during backpropagation!

When using **TP + SP + Selective Recomputation**, the quadratic term is completely removed:
$$\text{Mem}_{\text{TP+SP+Selective}} = \mathbf{\frac{34}{N} \cdot S \cdot B \cdot H \text{ bytes}}$$

```
Activation Memory for a 70B Model at Sequence Length 8,192:
  - Standard TP (v1):                     ~ 38.4 GB / GPU  (OOM on 40GB/80GB with optimizer!)
  - TP + Full Activation Checkpointing:   ~ 12.1 GB / GPU  (+33% FLOP compute slowdown!)
  - TP + SP + Selective Recomputation:    ~ 4.8 GB / GPU   (Only ~2-4% FLOP overhead!)
```

### 1.5.6 Activation Offloading (CPU/NVMe) vs. Selective Recomputation

A frequent architectural question in long-context training is: *Why not offload activations to Host CPU RAM over PCIe instead of recomputing them?*

```
┌────────────────────────┬──────────────────────┬──────────────────────┬───────────────────────────────┐
│ Strategy               │ Memory Overhead      │ Compute Penalty      │ Interconnect Requirement      │
├────────────────────────┼──────────────────────┼──────────────────────┼───────────────────────────────┤
│ **No Checkpointing**   │ Baseline ($100\%$)   │ $0\%$ extra FLOPs    │ None (Severe OOM Risk)        │
│ **CPU Offloading**     │ Minimal in HBM       │ $0\%$ extra FLOPs    │ Choked by PCIe (64 GB/s)      │
│ **Full Checkpointing** │ $-70\%$ HBM usage    │ $+33\%$ extra FLOPs  │ Zero external I/O             │
│ **Selective Recomp**   │ $\mathbf{-50\%}$ HBM │ $\mathbf{<3\%}$ FLOPs│ $\mathbf{Zero\ external\ I/O}$│
└────────────────────────┴──────────────────────┴──────────────────────┴───────────────────────────────┘
```

1. **The PCIe I/O Bottleneck**:
   - Modern HBM3 memory bandwidth is $3.35\text{ TB/s}$.
   - PCIe Gen 5 $\times 16$ interconnect bandwidth is only $64\text{ GB/s}$ ($\mathbf{\approx 52\times\text{ slower}}$).
   - Offloading an activation tensor to host CPU memory and reading it back during the backward pass introduces immense latency. The GPU's Tensor Cores spend up to $60\%$ of their time stalled waiting for DMA memory transfers.
2. **The Selective Recomputation Victory**:
   - Because modern Tensor Cores deliver nearly $1{,}000\text{ TFLOPS}$ of compute throughput, re-executing cheap elementwise attention operations (Softmax, Dropout) takes a few microseconds—vastly faster than shipping tensors across the PCIe bus!
   - Selective Recomputation is compute-optimal and memory-optimal for sequence lengths up to $64\text{k}$.

> [!NOTE]
> **Why This Matters**:
> In LLM pretraining, never use CPU activation offloading unless you have zero other options. Selective Activation Recomputation delivers the exact same memory relief with zero I/O transfer latency and $<3\%$ compute overhead.

---

**Code — Selective activation recomputation with `torch.utils.checkpoint`:**

```python
import torch
import torch.utils.checkpoint as checkpoint

class SelectiveRecomputeAttention(torch.nn.Module):
    """
    Wraps the 'expensive to store but cheap to recompute' quadratic operations
    (softmax, dropout) inside checkpoint so their activations are NOT stored.

    The GEMM inputs (Q, K, V after projection) ARE stored because:
    - Their memory is O(S*H), much smaller than O(S^2)
    - The QKV GEMMs are expensive to recompute (we don't want to pay that cost)

    This is the SELECTIVE strategy: only checkpoint what is cheap to recompute!
    """

    def _attention_core(self, Q, K, V, scale, dropout_p):
        """The QUADRATIC part: softmax(QK^T/sqrt(d)) @ V. This is CHECKPOINTED."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale      # O(S^2): NOT stored
        probs  = torch.nn.functional.softmax(scores, dim=-1)       # O(S^2): NOT stored
        if dropout_p > 0:
            probs = torch.nn.functional.dropout(probs, p=dropout_p)  # O(S^2): NOT stored
        return torch.matmul(probs, V)

    def forward(self, Q, K, V, scale, dropout_p):
        # torch.utils.checkpoint.checkpoint re-runs _attention_core during backward
        # INSTEAD of storing the O(S^2) softmax and dropout tensors!
        #
        # use_reentrant=False is required in PyTorch >= 2.0 for correct behavior
        # with autograd, custom backward, and nested checkpoints.
        return checkpoint.checkpoint(
            self._attention_core,
            Q, K, V, scale, dropout_p,
            use_reentrant=False,
        )


# In Megatron Core the real API:
# from megatron.core.transformer.dot_product_attention import DotProductAttention
# The config flag that enables this is:
# transformer_config.recompute_granularity = 'selective'   # NOT 'full'
# transformer_config.recompute_method = 'uniform'
#
# Internally Megatron uses flash_attn_varlen_func or core_attention
# wrapped in tensor_parallel.checkpoint() when selective recompute is on.
```

### 1.5.7 Deep Line-by-Line Pedagogical Breakdown: `SelectiveRecomputeAttention`

1. **Lines 250–257 (`_attention_core` Definition):**
   - Implements the pure quadratic attention operations: `scores = torch.matmul(Q, K^T) * scale`, followed by `softmax` and optional `dropout`.
   - In standard backprop, the entire $[B, h, S, S]$ probability matrix must be saved in GPU memory to compute $\frac{\partial \mathcal{L}}{\partial \text{scores}} = P \odot (\frac{\partial \mathcal{L}}{\partial P} - \sum P \odot \frac{\partial \mathcal{L}}{\partial P})$.
   - By isolating this function into an isolated sub-graph, we can checkpoint it independently from the linear GEMMs.
2. **Lines 264–268 (`checkpoint.checkpoint(..., use_reentrant=False)`):**
   - PyTorch evaluates `_attention_core` in the forward pass, outputs the attended values $O = P \cdot V$, but **does NOT save the internal $P$ tensor to autograd's activation tape**.
   - Instead, only the inputs $(Q, K, V)$ are saved. Because $(Q, K, V)$ each have size $B \times S \times H$, their combined storage is $\mathcal{O}(S \cdot H)$—vastly smaller than the $\mathcal{O}(S^2)$ attention matrix.
   - During backward propagation, PyTorch re-executes `_attention_core` on-the-fly using the stashed $(Q, K, V)$, computes the attention probabilities, executes backprop through them, and instantly discards them from memory.

---

